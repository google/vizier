# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

"""GP-Bandit using a Flax model and a TFP Gaussian Process.

A Python implementation of Google Vizier's GP-Bandit algorithm.
"""
# pylint: disable=logging-fstring-interpolation, g-long-lambda

import copy
import datetime
import functools
import json
import random
from typing import Optional, Sequence, Union

from absl import logging
import attr
import chex
import jax
from jax import numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.algorithms.designers.gp import output_warpers
from vizier._src.algorithms.optimizers import eagle_strategy as es
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax.models import tuned_gp_models
from vizier._src.jax.optimizers import optimizers
from vizier.pyvizier import converters
from vizier.utils import json_utils

tfd = tfp.distributions


@attr.define(auto_attribs=False)
class VizierGPBandit(vza.Designer):
  """GP-Bandit using a Flax model.

  Attributes:
    problem: Must be a flat study with a single metric.
    acquisition_optimizer: Typically either a designer wrapped as an optimizer
      or a batched optimizer (like Eagle).
    metadata_ns: Metadata namespace that this designer writes to.
    use_trust_region: Uses trust region.
    ard_optimizer: An optimizer which should return a batch of hyperparameters
      to be ensembled.
    num_acquisition_optimizer_evaluations: Number of evaluations the optimizer
      can perform to find the optimal acquisition trials.
    acquisition_batch_size: The number of trials to evaluate in each acquisition
      iteration.
    num_seed_trials: If greater than zero, first trial is the center of the
      search space. Afterwards, uses quasirandom until this number of trials are
      observed.
    rng: If not set, uses random numbers.
  """

  _problem: vz.ProblemStatement = attr.field(kw_only=False)
  _acquisition_optimizer: Union[
      vza.GradientFreeOptimizer, vb.VectorizedOptimizer
  ] = attr.field(kw_only=False,
                 factory=lambda: VizierGPBandit.default_acquisition_optimizer)
  _metadata_ns: str = attr.field(default='oss_gp_bandit', kw_only=True)
  _trials: list[vz.Trial] = attr.field(factory=list, init=False)
  _ard_optimizer: optimizers.Optimizer = attr.field(
      factory=lambda: VizierGPBandit.default_ard_optimizer, kw_only=True)
  _use_trust_region: bool = attr.field(default=True, kw_only=True)
  _num_seed_trials: int = attr.field(default=1, kw_only=True)
  _rng: chex.PRNGKey = attr.field(
      factory=lambda: jax.random.PRNGKey(random.getrandbits(32)), kw_only=True
  )

  # not an attr field.
  # Only one of these optimizers will be used.
  # `default_ard_optimizer` returns the best 5 parameter values for ensembling,
  # while `default_ard_optimizer_noensemble` returns only
  # the single best parameter value.
  default_ard_optimizer = optimizers.JaxoptLbfgsB(random_restarts=32, best_n=5)
  default_ard_optimizer_noensemble = optimizers.JaxoptLbfgsB(
      random_restarts=32, best_n=None
  )

  # not an attr field.
  default_acquisition_optimizer = vb.VectorizedOptimizer(
      strategy_factory=es.VectorizedEagleStrategyFactory())

  def __attrs_post_init__(self):
    # Extra validations
    if self._problem.search_space.is_conditional:
      raise ValueError(f'{type(self)} does not support conditional search.')
    elif len(self._problem.metric_information) != 1:
      raise ValueError(f'{type(self)} works with exactly one metric.')

    # Extra initializations.
    # Discrete parameters are continuified to account for their actual values.
    self._converter = converters.TrialToArrayConverter.from_study_config(
        self._problem,
        scale=True,
        pad_oovs=True,
        max_discrete_indices=0,
        flip_sign_for_minimization_metrics=True,
    )
    self._quasi_random_sampler = quasi_random.QuasiRandomDesigner(
        self._problem.search_space
    )

  def update(self, trials: vza.CompletedTrials) -> None:
    self._trials.extend(copy.deepcopy(trials.completed))

  @property
  def _metric_info(self) -> vz.MetricInformation:
    return self._problem.metric_information.item()

  def suggest(
      self, count: Optional[int] = None
  ) -> Sequence[vz.TrialSuggestion]:
    count = count or 1
    if len(self._trials) < self._num_seed_trials:
      seed_suggestions = []
      if not self._trials:
        # TODO: Should track number of pending suggestions
        # so we don't suggest the center more than once.
        features = self._converter.to_features([])  # to extract shape.
        # Scaled value of .5 corresponds to the center of the feasible range.
        parameters = self._converter.to_parameters(
            0.5 * np.ones([1, features.shape[1]])
        )[0]
        seed_suggestions.append(
            vz.TrialSuggestion(
                parameters, metadata=vz.Metadata({'seeded': 'center'})
            )
        )
      if (remaining_counts := count - len(seed_suggestions)) > 0:
        seed_suggestions.extend(
            self._quasi_random_sampler.suggest(remaining_counts)
        )
      return seed_suggestions

    begin_time = datetime.datetime.now()
    self._rng, rng = jax.random.split(self._rng, 2)
    # TrialToArrayConverter returns floating arrays.
    features, labels = self._converter.to_xy(self._trials)
    logging.info(
        'Transforming the labels of shape %s. Features has shape: %s',
        labels.shape,
        features.shape,
    )

    # Warp the output.
    labels = output_warpers.create_default_warper()(labels)

    labels = labels.reshape([-1])
    logging.info('Transformed the labels. Now has shape: %s', labels.shape)

    # Run ARD.
    setup = lambda rng: model.init(rng, features)['params']
    model, loss_fn = tuned_gp_models.VizierGaussianProcess.model_and_loss_fn(
        features, labels
    )
    constraints = sp.get_constraints(model.coroutine)
    loss_fn = jax.jit(loss_fn)

    logging.info('Optimizing the loss function...')
    optimal_params, _ = self._ard_optimizer(
        setup, loss_fn, rng, constraints=constraints
    )

    logging.info('Optimal parameters: %s', optimal_params)
    if self._ard_optimizer is not self.default_ard_optimizer_noensemble:
      loss_fn = jax.vmap(loss_fn)

    ard_all_losses = loss_fn(optimal_params)[0]
    logging.info('All losses: %s', ard_all_losses)
    ard_best_loss = ard_all_losses.flatten()[0].item()

    def predict_on_array_one_model(
        params: chex.ArrayTree, *, xs: chex.Array
    ) -> chex.ArrayTree:
      predictive, _ = model.apply(
          {'params': params},
          xs,
          features,
          labels,
          method=model.posterior,
          mutable=True,
      )
      return {'mean': predictive.mean(), 'stddev': predictive.stddev()}

    @jax.jit
    def predict_on_array(xs: chex.Array) -> chex.ArrayTree:
      predict_fn = functools.partial(predict_on_array_one_model, xs=xs)
      if self._ard_optimizer is self.default_ard_optimizer_noensemble:
        pp = predict_fn(
            optimal_params,
        )
        return {'mean': pp['mean'], 'stddev': pp['stddev']}
      predict_fn = jax.vmap(predict_fn)
      pp = predict_fn(
          optimal_params,
      )
      batched_normal = tfd.Normal(pp['mean'].T, pp['stddev'].T)
      mixture = tfd.MixtureSameFamily(
          tfd.Categorical(logits=jnp.ones(batched_normal.batch_shape[1])),
          batched_normal,
      )
      return {'mean': mixture.mean(), 'stddev': mixture.stddev()}

    # Optimize acquisition.
    tr = acquisitions.TrustRegion(features, self._converter.output_specs)

    @jax.jit
    def acquisition_on_array(xs):
      pred = predict_on_array(xs)
      ucb = pred['mean'] + 1.8 * pred['stddev']
      if self._use_trust_region and tr.trust_radius < 0.5:
        distance = tr.min_linf_distance(xs)
        # Due to output normalization, ucb can't be nearly as low as -1e12.
        # We use a bad value that decreases in the distance to trust region
        # so that acquisition optimizer can follow the gradient and escape
        # untrustred regions.
        return jnp.where(distance <= tr.trust_radius, ucb, -1e12 - distance)
      else:
        return ucb

    def acquisition_on_trials(trials: Sequence[vz.Trial]):
      array = self._converter.to_features(trials)
      jax_acquisitions = acquisition_on_array(array)
      return {'acquisition': jax_acquisitions}

    acquisition_problem = copy.deepcopy(self._problem)
    acquisition_problem.metric_information = [
        vz.MetricInformation(
            name='acquisition', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    ]
    logging.info('Optimizing acquisition...')

    # TODO: Change budget based on requested suggestion count.
    if isinstance(self._acquisition_optimizer, vb.VectorizedOptimizer):
      best_candidates = self._acquisition_optimizer.optimize(
          score_fn=acquisition_on_array,
          converter=self._converter,
          count=count,
          prior_trials=self._trials,
      )
    elif isinstance(self._acquisition_optimizer, vza.GradientFreeOptimizer):
      # Seed the optimizer with previous trials.
      best_candidates = self._acquisition_optimizer.optimize(
          score_fn=acquisition_on_trials,
          problem=acquisition_problem,
          count=count,
          seed_candidates=copy.deepcopy(self._trials),
      )

    # Convert best_candidates (in warped space) into suggestions (in unwarped
    # space); also append debug inforamtion like model predictions.
    logging.info('Converting the optimization result into suggestions...')
    optimal_features = self._converter.to_features(best_candidates)  # [N, D]
    # Make predictions (in the warped space).
    predictions = predict_on_array(optimal_features)  # event shape [N]
    predict_mean = predictions['mean'].reshape([-1])  # [N,]
    predict_stddev = predictions['stddev'].reshape([-1])  # [N,]
    logging.info(
        'Created predictions for the best candidates which were '
        f'converted to an array of shape: {optimal_features.shape}. '
        f'mean has shape {predict_mean.shape}. '
        f'stddev has shape {predict_stddev.shape}.'
    )
    # Create suggestions, injecting the predictions as metadata for
    # debugging needs.
    suggestions = []
    for i, candidate in enumerate(best_candidates):
      metadata = candidate.metadata.ns(self._metadata_ns).ns('devinfo')
      acquisition = candidate.final_measurement.metrics.get_value(
          'acquisition', 'nan'
      )
      metadata['prediction_in_warped_y_space'] = json.dumps(
          {
              'ard_all_losses': ard_all_losses,
              'ard_best_loss': ard_best_loss,
              'mean': predict_mean[i],
              'stddev': predict_stddev[i],
              'acquisiton': acquisition,
              'trust_radius': tr.trust_radius,
              'params': optimal_params,
          },
          cls=json_utils.NumpyEncoder,
      )
      metadata['time_spent'] = f'{datetime.datetime.now() - begin_time}'
      suggestions.append(
          vz.TrialSuggestion(candidate.parameters, metadata=candidate.metadata)
      )

    return suggestions
