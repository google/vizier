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
import json
import random
from typing import Optional, Sequence, Union, Callable, Any

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


@attr.define(slots=False)
class GPBanditAcquisitionBuilder(acquisitions.AcquisitionBuilder):
  """Acquisition/prediction builder for the GP Bandit class.

  This builder takes in a Jax/Flax model, along with its hparams, and builds
  the usable predictive metrics, as well as the acquisition.

  For example:

    acquisition_builder =
    GPBanditAcquisitionBuilder(acquisition_fn=acquisitions.UCB())
    acquisition_builder.build(
          problem_statement,
          model=model,
          state=state,
          features=features,
          labels=labels,
          converter=self._converter,
    )
    # Get the acquisition Callable.
    acq = acquisition_builder.acquisition_on_array
  """

  # Acquisition function that takes a TFP distribution and optional features
  # and labels.
  acquisition_fn: acquisitions.AcquisitionFunction = attr.field(
      factory=acquisitions.UCB, kw_only=True
  )
  use_trust_region: bool = attr.field(default=True, kw_only=True)

  def __attrs_post_init__(self):
    # Perform extra initializations.
    self._built = False

  def build(
      self,
      problem: vz.ProblemStatement,
      model: sp.StochasticProcessModel,
      state: chex.ArrayTree,
      features: chex.Array,
      labels: chex.Array,
      converter: converters.TrialToArrayConverter,
      use_vmap: bool = True,
  ) -> None:
    """Generates the predict and acquisition functions.

    Args:
      problem: See abstraction.
      model: See abstraction.
      state: See abstraction.
      features: See abstraction.
      labels: See abstraction.
      converter: TrialToArrayConverter for TrustRegion configuration.
      use_vmap: If True, applies Vmap across parameter ensembles.
    """

    def _predict_on_array_one_model(
        state: chex.ArrayTree, *, xs: chex.Array
    ) -> chex.ArrayTree:
      return model.apply(
          state,
          xs,
          features,
          labels,
          method=model.posterior_predictive,
      )

    # Vmaps and combines the predictive distribution over all models.
    def _get_predictive_dist(xs: chex.Array) -> tfd.Distribution:
      if not use_vmap:
        return _predict_on_array_one_model(state, xs=xs)

      def _predict_mean_and_stddev(state_: chex.ArrayTree) -> chex.ArrayTree:
        dist = _predict_on_array_one_model(state_, xs=xs)
        return {'mean': dist.mean(), 'stddev': dist.stddev()}  # pytype: disable=attribute-error  # numpy-scalars

      # Returns a dictionary with mean and stddev, of shape [M, N].
      # M is the size of the parameter ensemble and N is the number of points.
      pp = jax.vmap(_predict_mean_and_stddev)(state)
      batched_normal = tfd.Normal(pp['mean'].T, pp['stddev'].T)  # pytype: disable=attribute-error  # numpy-scalars
      return tfd.MixtureSameFamily(
          tfd.Categorical(logits=jnp.ones(batched_normal.batch_shape[1])),
          batched_normal,
      )

    # Could also allow injection of a prediction function that takes a
    # distribution and returns something else, or rename this to
    # 'predict_mean_and_stddev`, e.g.`
    @jax.jit
    def predict_on_array(xs: chex.Array) -> chex.ArrayTree:
      dist = _get_predictive_dist(xs)
      return {'mean': dist.mean(), 'stddev': dist.stddev()}

    self._predict_on_array = predict_on_array

    # Define acquisition.
    self._tr = acquisitions.TrustRegion(features, converter.output_specs)

    # This supports acquisition fns that do arbitrary computations with the
    # input distributions -- e.g. they could take samples or compute quantiles.
    @jax.jit
    def acquisition_on_array(xs):
      dist = _get_predictive_dist(xs)
      acquisition = self.acquisition_fn(dist, features, labels)
      if self.use_trust_region and self._tr.trust_radius < 0.5:
        distance = self._tr.min_linf_distance(xs)
        # Due to output normalization, acquisition can't be nearly as
        # low as -1e12.
        # We use a bad value that decreases in the distance to trust region
        # so that acquisition optimizer can follow the gradient and escape
        # untrusted regions.
        return jnp.where(
            distance <= self._tr.trust_radius, acquisition, -1e12 - distance
        )
      else:
        return acquisition

    self._acquisition_on_array = acquisition_on_array

    acquisition_problem = copy.deepcopy(problem)
    config = vz.MetricsConfig(
        metrics=[
            vz.MetricInformation(
                name='acquisition', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    acquisition_problem.metric_information = config
    self._acquisition_problem = acquisition_problem
    self._built = True

  @property
  def acquisition_problem(self) -> vz.ProblemStatement:
    if not self._built:
      raise ValueError('Acquisition must be built first via build().')
    return self._acquisition_problem

  @property
  def acquisition_on_array(self) -> Callable[[chex.Array], chex.Array]:
    if not self._built:
      raise ValueError('Acquisition must be built first via build().')
    return self._acquisition_on_array

  @property
  def predict_on_array(self) -> Callable[[chex.Array], chex.ArrayTree]:
    if not self._built:
      raise ValueError('Acquisition must be built first via build().')
    return self._predict_on_array

  @property
  def metadata_dict(self) -> dict[str, Any]:
    if not self._built:
      raise ValueError('Acquisition must be built first via build().')
    return {'trust_radius': self._tr.trust_radius}


@attr.define(auto_attribs=False)
class VizierGPBandit(vza.Designer):
  """GP-Bandit using a Flax model.

  Attributes:
    problem: Must be a flat study with a single metric.
    acquisition_optimizer: Typically either a designer wrapped as an optimizer
      or a batched optimizer (like Eagle).
    metadata_ns: Metadata namespace that this designer writes to.
    ard_optimizer: An optimizer which should return a batch of hyperparameters
      to be ensembled.
    use_trust_region: Uses trust region.
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
  _acquisition_builder: acquisitions.AcquisitionBuilder = attr.field(
      factory=GPBanditAcquisitionBuilder, kw_only=False
  )
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

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    self._trials.extend(copy.deepcopy(completed.trials))

  @property
  def _metric_info(self) -> vz.MetricInformation:
    return self._problem.metric_information.item()

  def suggest(
      self, count: Optional[int] = None
  ) -> Sequence[vz.TrialSuggestion]:
    count = count or 1
    if count > 1:
      logging.warning(
          'GAUSSIAN_PROCESS_BANDIT currently is not optimized for batched'
          ' suggestions. Suggestions in the batch are likely to be very'
          ' similar.'
      )
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

    @jax.jit
    def precompute_cholesky(params):
      return model.apply(
          {'params': params},
          features,
          labels,
          method=model.precompute_predictive,
          mutable=('predictive',),
      )

    if self._ard_optimizer is not self.default_ard_optimizer_noensemble:
      precompute_cholesky = jax.vmap(precompute_cholesky)

    # `pp_state` contains intermediates that are expensive to compute, depend
    # only on observed (not predictive) index points, and are needed to compute
    # the posterior predictive GP (i.e. the Cholesky factor of the kernel matrix
    # over observed index points). `pp_state` is passed to
    # `model.posterior_predictive` to avoid re-computing the intermediates
    # unnecessarily.
    _, pp_state = precompute_cholesky(optimal_params)
    state = {'params': optimal_params, **pp_state}

    self._acquisition_builder.build(
        self._problem,
        model=model,
        state=state,
        features=features,
        labels=labels,
        converter=self._converter,
    )
    acquisition_problem = self._acquisition_builder.acquisition_problem
    acquisition_on_array = self._acquisition_builder.acquisition_on_array
    predict_on_array = self._acquisition_builder.predict_on_array
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

      def acquisition_on_trials(trials: Sequence[vz.Trial]):
        array = self._converter.to_features(trials)
        jax_acquisitions = acquisition_on_array(array)
        return {'acquisition': jax_acquisitions}

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
    predictions = predict_on_array(optimal_features)  # event shape [N]
    # Make predictions (in the warped space), reshape and log shapes.
    for predict_key, predict_value in predictions.items():  # pytype: disable=attribute-error  # numpy-scalars
      predictions[predict_key] = predict_value.reshape([-1])  # [N,]  # pytype: disable=unsupported-operands  # numpy-scalars
    logging.info(
        'Created predictions for the best candidates which were '
        f'converted to an array of shape: {optimal_features.shape}. '
        f' with predictions { {k : v.shape for k,v in predictions.items() } }'  # pytype: disable=attribute-error  # numpy-scalars
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
              'acquisition': acquisition,
              'params': optimal_params,
          }
          | self._acquisition_builder.metadata_dict
          | {k: v[i] for k, v in predictions.items()},  # pytype: disable=attribute-error  # numpy-scalars
          cls=json_utils.NumpyEncoder,
      )
      metadata['time_spent'] = f'{datetime.datetime.now() - begin_time}'
      suggestions.append(
          vz.TrialSuggestion(candidate.parameters, metadata=candidate.metadata)
      )

    return suggestions
