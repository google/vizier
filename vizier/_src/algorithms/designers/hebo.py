# Copyright 2022 Google LLC.
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

"""Vizer Hebo using a flax GP."""
# pylint: disable=logging-fstring-interpolation, g-long-lambda

import copy
import datetime
import functools
import random
from typing import Optional, Sequence

from absl import logging
import attr
import chex
from flax.core import frozen_dict
import jax
from jax import numpy as jnp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers.google import yjt
from vizier._src.algorithms.evolution import nsga2
from vizier._src.algorithms.optimizers import designer_optimizer
from vizier._src.jax import optimizers
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax.models import hebo_gp_model
from vizier.pyvizier import converters

tfd = tfp.distributions


@attr.define(auto_attribs=False)
class VizierHebo(vza.Designer):
  """Vizier Hebo with a flax model."""
  _problem: vz.ProblemStatement = attr.field(kw_only=False)
  _metadata_ns: str = attr.field(default='hebo', kw_only=True)
  _trials: tuple[vz.Trial] = attr.field(factory=tuple, init=False)
  _ard_optimizer: optimizers.Optimizer[chex.ArrayTree] = attr.field(
      factory=lambda: VizierHebo.default_optimizer, kw_only=True)
  _num_seed_trials: int = attr.field(default=1, kw_only=True)
  _best_measure: float = attr.field(init=False, default=float('-inf'))
  _nsga_batch_size: int = attr.field(init=True, default=100)
  _nsga_num_evaluations: int = attr.field(init=True, default=15000)
  _rng: chex.PRNGKey = attr.field(
      factory=lambda: jax.random.PRNGKey(random.getrandbits(32)), kw_only=True)

  # TODO: Update to values to match official HEBO implementation.
  _kappa: float = attr.field(default=1.0)

  # not an attr field.
  default_optimizer = optimizers.OptaxTrainWithRandomRestarts(
      optax.adam(5e-3), epochs=100, verbose=False, random_restarts=50, best_n=5)

  def __attrs_post_init__(self):
    # Extra validations
    if self._problem.search_space.is_conditional:
      raise ValueError(f'{type(self)} does not support conditional search.')
    elif len(self._problem.metric_information) != 1:
      raise ValueError(f'{type(self)} works with exactly one metric.')

    # Extra initializations.
    # Discrete parameters are continuified to account for their actual values.
    self._converter = converters.TrialToArrayConverter.from_study_config(
        self._problem, max_discrete_indices=0, pad_oovs=True, scale=True)

    self._quasi_random_sampler = quasi_random.QuasiRandomDesigner(
        self._problem.search_space)

  def update(self, trials: vza.CompletedTrials) -> None:
    self._trials += tuple(trials.completed)
    # Update the best measure seen thus far to be used in acquisition function.
    for trial in trials.completed:
      trial_measure = list(trial.final_measurement.metrics.values())[0].value
      if trial_measure > self._best_measure:
        self._best_measure = trial_measure

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    count = count or 1
    if len(self._trials) < self._num_seed_trials:
      seed_suggestions = []
      if not self._trials:
        # TODO: Should track number of pending suggestions
        # so we don't suggest the center more than once.
        features = self._converter.to_features([])  # to extract shape.
        # Scaled value of .5 corresponds to the center of the feasible range.
        parameters = self._converter.to_parameters(
            .5 * np.ones([1, features.shape[1]]))[0]
        seed_suggestions.append(
            vz.TrialSuggestion(
                parameters, metadata=vz.Metadata({'seeded': 'center'})))
      if (remaining_counts := count - len(seed_suggestions)) > 0:
        seed_suggestions.extend(
            self._quasi_random_sampler.suggest(remaining_counts))
      return seed_suggestions
    self._rng, rng = jax.random.split(self._rng, 2)
    begin = datetime.datetime.now()
    # TrialToArrayConverter returns floating arrays.
    features, labels = self._converter.to_xy(self._trials)
    logging.info('Transforming the labels of shape %s. Features has shape: %s',
                 labels.shape, features.shape)

    # Warp the output.
    labels = yjt.optimal_transformation(labels)(labels)
    labels = labels.reshape([-1])
    logging.info('Transformed the labels. Now has shape: %s', labels.shape)

    # Define model.
    model, loss_fn = hebo_gp_model.VizierHeboGaussianProcess.model_and_loss_fn(
        features, labels)

    # Run ARD.
    setup = lambda rng: model.init(rng, features)['params']
    logging.info('Optimizing the loss function...')
    optimal_params = self._ard_optimizer(setup, loss_fn, rng)

    optimal_constrained_params = sp.ConstrainModelParametersFunction(
        model.coroutine)(
            optimal_params)
    logging.info('Optimal parameters: %s', optimal_constrained_params)

    def predict_on_array_one_model(params: chex.ArrayTree, *,
                                   xs: chex.Array) -> chex.ArrayTree:
      predictive, _ = model.apply({'params': params},
                                  xs,
                                  features,
                                  labels,
                                  method=model.posterior,
                                  mutable=True)
      return {'mean': predictive.mean(), 'stddev': predictive.stddev()}

    @jax.jit
    def predict_on_array(xs: chex.Array) -> chex.ArrayTree:
      pp = jax.vmap(functools.partial(predict_on_array_one_model,
                                      xs=xs))(optimal_params,)
      batched_normal = tfd.Normal(pp['mean'].T, pp['stddev'].T)
      mixture = tfd.MixtureSameFamily(
          tfd.Categorical(logits=jnp.ones(batched_normal.batch_shape[1])),
          batched_normal)
      return {'mean': mixture.mean(), 'stddev': mixture.stddev()}

    # Optimize acquisition.
    @jax.jit
    def acquisition_on_array(xs, normals):
      pred = predict_on_array(xs)

      # Generate noise to perturb posteriors.
      noise = jnp.power(
          jnp.std(labels),
          2) * optimal_constrained_params['observation_noise_variance']
      noise = jnp.float64(jnp.sqrt(2.0) * noise)

      # UCB + Perturbation
      ucb = pred['mean'] + self._kappa * pred['stddev']
      perturbed_ucb = ucb + noise * normals[0]

      # EI + Perturbation
      z0 = (pred['mean'] - self._best_measure) / pred['stddev']
      normal = tfd.Normal(0, 1)
      log_ei = jnp.log(pred['stddev'] * (z0 * normal.cdf(z0) + normal.prob(z0)))
      perturbed_log_ei = log_ei + noise * normals[1]

      # PI + Perturbation
      log_pi = jnp.log(normal.cdf(z0))
      perturbed_log_pi = log_pi + noise * normals[2]

      return perturbed_ucb, perturbed_log_ei, perturbed_log_pi

    def acquisition_on_trials(
        trials: Sequence[vz.Trial]) -> dict[str, np.ndarray]:
      array = self._converter.to_features(trials)
      self._rng, rng = jax.random.split(self._rng, 2)
      normals = jax.random.normal(key=rng, shape=(3, array.shape[0]))
      jax_acquisitions = acquisition_on_array(array, normals)
      print(f'\nacquisition_on_trials. jax_acquisitions: {jax_acquisitions}')
      return {
          'ucb': np.array(jax_acquisitions[0]),
          'log_ei': np.array(jax_acquisitions[1]),
          'log_pi': np.array(jax_acquisitions[2])
      }

    acquisition_problem = copy.deepcopy(self._problem)
    acquisition_problem.metric_information = [
        vz.MetricInformation(name='ucb', goal=vz.ObjectiveMetricGoal.MAXIMIZE),
        vz.MetricInformation(
            name='log_ei', goal=vz.ObjectiveMetricGoal.MAXIMIZE),
        vz.MetricInformation(
            name='log_pi', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    ]
    logging.info('Multi-objective optimizing acquisition using NSGA-II...')
    # TODO: Create vectorized implemenetation to expedite performance.
    # TODO: Update NSGA-II 'mutation' to SBX.
    # TODO: Fix NSGA-II number of mutated solution to match 'count'.
    # TODO: Choose new candidates by highest mean and highest std from
    # the Pareto frontier and the rest choose randomly.
    optimizer = designer_optimizer.DesignerAsOptimizer(
        nsga2.create_nsga2,
        batch_size=self._nsga_batch_size,
        num_evaluations=self._nsga_num_evaluations)
    new_candidates = optimizer.optimize(
        score_fn=acquisition_on_trials,
        problem=acquisition_problem,
        count=count)
    # Generate exactly 'count' offsprings by arbitrarily chosing the first ones
    # and repeats solutions if needed.
    if len(new_candidates) >= count:
      new_candidates = new_candidates[0:count]
    else:
      self._rng, rng = jax.random.split(self._rng, 2)
      new_candidates += list(
          np.array(new_candidates)[jax.random.choice(
              key=rng,
              a=np.arange(len(new_candidates)),
              size=count - len(new_candidates),
              replace=True)])
    # Make predictions (in the warped space).
    logging.info('Converting the optimization result into suggestions...')
    optimal_features = self._converter.to_features(new_candidates)  # [N, D]
    predictions = predict_on_array(optimal_features)  # event shape [N]
    predict_mean = predictions['mean'].reshape([-1])  # [N,]
    predict_stddev = predictions['stddev'].reshape([-1])  # [N,]
    logging.info(f'Created predictions for the best candidates which were '
                 f'converted to an array of shape: {optimal_features.shape}. '
                 f'mean has shape {predict_mean.shape}. '
                 f'stddev has shape {predict_stddev.shape}.')
    # Create suggestions, injecting the predictions as metadata for
    # debugging needs.
    suggestions = []
    for i, candidate in enumerate(new_candidates):
      metadata = candidate.metadata.ns(self._metadata_ns)
      acquisition = candidate.final_measurement.metrics.get_value(
          'acquisition', 'nan')
      metadata.ns('prediction_in_warped_y_space').update({
          'mean':
              f'{predict_mean[i]}',
          'stddev':
              f'{predict_stddev[i]}',
          'acquisiton':
              f'{acquisition}',
          # Unfreeze the FrozenDict because its repr is overelaborate.
          'params':
              f'{frozen_dict.unfreeze(frozen_dict.FrozenDict(optimal_constrained_params))}',
      })
      metadata.ns('timing').update(
          {'time': f'{datetime.datetime.now() - begin}'})
      suggestions.append(
          vz.TrialSuggestion(candidate.parameters, metadata=candidate.metadata))

    return suggestions
