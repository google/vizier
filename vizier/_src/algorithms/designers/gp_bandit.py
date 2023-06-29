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
import dataclasses
import datetime
import random
from typing import Optional, Sequence

from absl import logging
import attr
import equinox as eqx
import jax
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.algorithms.designers.gp import output_warpers
from vizier._src.algorithms.optimizers import eagle_strategy as es
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import gp_bandit_utils
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier._src.jax.models import tuned_gp_models
from vizier.jax import optimizers
from vizier.pyvizier import converters
from vizier.pyvizier.converters import feature_mapper
from vizier.pyvizier.converters import padding
from vizier.utils import profiler


def _experimental_override_allowed(fun):
  """No-op. Marks functions that can be easily overriden for experimentation."""
  return fun


_GPBanditState = tuple[sp.UniformEnsemblePredictive, types.ModelData]


@attr.define(auto_attribs=False)
class VizierGPBandit(vza.Designer, vza.Predictor):
  """GP-Bandit using a Flax model.

  A minimal example of creating this designer:
  problem = vz.ProblemStatement(...)  # Configure a minimal problem statement.
  designer = VizierGPBandit(problem)

  Optionally set other attributes to change the defaults, e.g.:
  problem = vz.ProblemStatement(...)  # Configure a minimal problem statement.
  designer = VizierGPBandit(problem, use_trust_region=False)

  Attributes:
    problem: Must be a flat study with a single metric.
    acquisition_optimizer: Typically either a designer wrapped as an optimizer
      or a batched optimizer (like Eagle).
    ard_optimizer: An optimizer which should return a batch of hyperparameters
      to be ensembled.
    num_seed_trials: If greater than zero, first trial is the center of the
      search space. Afterwards, uses quasirandom until this number of trials are
      observed.
    acquisition_function: acquisition function to use.
    use_trust_region: Uses trust region to constrain initial exploration.
    rng: If not set, uses random numbers.
    metadata_ns: Metadata namespace that this designer writes to.
  """

  _problem: vz.ProblemStatement = attr.field(kw_only=False)
  _acquisition_optimizer_factory: vb.VectorizedOptimizerFactory = attr.field(
      kw_only=True,
      factory=lambda: VizierGPBandit.default_acquisition_optimizer_factory,
  )
  _ard_optimizer: optimizers.Optimizer[types.ParameterDict] = attr.field(
      factory=optimizers.default_optimizer,
      kw_only=True,
  )
  _num_seed_trials: int = attr.field(default=1, kw_only=True)
  _acquisition_function: acquisitions.AcquisitionFunction = attr.field(
      factory=acquisitions.UCB, kw_only=True
  )
  # Whether to pad all inputs, and what type of schedule to use. This is to
  # ensure fewer JIT compilation passes. (Default implies no padding.)
  _padding_schedule: padding.PaddingSchedule = attr.field(
      factory=padding.PaddingSchedule, kw_only=True
  )
  _use_trust_region: bool = attr.field(default=True, kw_only=True)
  _rng: jax.random.KeyArray = attr.field(
      factory=lambda: jax.random.PRNGKey(random.getrandbits(32)), kw_only=True
  )
  _metadata_ns: str = attr.field(
      default='oss_gp_bandit', kw_only=True, init=False
  )
  _ensemble_size: Optional[int] = attr.field(default=1, kw_only=True)

  # ------------------------------------------------------------------
  # Internal attributes which should not be set by callers.
  # ------------------------------------------------------------------
  _trials: list[vz.Trial] = attr.field(factory=list, init=False)
  # The number of trials that have been incorporated
  # into the designer state (Cholesky decomposition, ARD).
  _incorporated_trials_count: int = attr.field(
      default=0, kw_only=True, init=False
  )
  _acquisition_optimizer: vb.VectorizedOptimizer = attr.field(init=False)
  _output_warper_pipeline: output_warpers.OutputWarperPipeline = attr.field(
      init=False
  )
  # TODO: Remove this.
  _feature_mapper: Optional[
      feature_mapper.ContinuousCategoricalFeatureMapper
  ] = attr.field(init=False, default=None)
  _last_computed_state: _GPBanditState = attr.field(init=False)

  default_acquisition_optimizer_factory = vb.VectorizedOptimizerFactory(
      strategy_factory=es.VectorizedEagleStrategyFactory()
  )

  def __attrs_post_init__(self):
    # Extra validations
    if self._problem.search_space.is_conditional:
      raise ValueError(f'{type(self)} does not support conditional search.')
    elif len(self._problem.metric_information) != 1:
      raise ValueError(f'{type(self)} works with exactly one metric.')
    # Extra initializations.
    # Discrete parameters are continuified to account for their actual values.
    self._converter = converters.TrialToModelInputConverter.from_problem(
        self._problem,
        scale=True,
        max_discrete_indices=0,
        flip_sign_for_minimization_metrics=True,
        padding_schedule=self._padding_schedule,
    )
    self._quasi_random_sampler = quasi_random.QuasiRandomDesigner(
        self._problem.search_space,
        seed=int(jax.random.randint(self._rng, [], 0, 2**16)),
    )
    self._output_warper_pipeline = output_warpers.create_default_warper()

    # TODO: Get rid of this when Vectorized optimizers operate on CACV.
    self._one_hot_converter = (
        converters.TrialToArrayConverter.from_study_config(
            self._problem,
            scale=True,
            pad_oovs=True,
            max_discrete_indices=0,
            flip_sign_for_minimization_metrics=True,
        )
    )
    self._padded_one_hot_converter = (
        converters.PaddedTrialToArrayConverter.from_study_config(
            self._problem,
            scale=True,
            pad_oovs=True,
            padding_schedule=self._padding_schedule,
            max_discrete_indices=0,
            flip_sign_for_minimization_metrics=True,
        )
    )
    self._feature_mapper = feature_mapper.ContinuousCategoricalFeatureMapper(
        self._one_hot_converter
    )
    self._acquisition_optimizer = self._acquisition_optimizer_factory(
        self._padded_one_hot_converter
    )
    acquisition_problem = copy.deepcopy(self._problem)
    if isinstance(
        self._acquisition_function, acquisitions.MultiAcquisitionFunction
    ):
      acquisition_config = vz.MetricsConfig()
      for k in self._acquisition_function.acquisition_fns.keys():
        acquisition_config.append(
            vz.MetricInformation(
                name=k,
                goal=vz.ObjectiveMetricGoal.MAXIMIZE,
            )
        )
    else:
      acquisition_config = vz.MetricsConfig(
          metrics=[
              vz.MetricInformation(
                  name='acquisition', goal=vz.ObjectiveMetricGoal.MAXIMIZE
              )
          ]
      )

    acquisition_problem.metric_information = acquisition_config
    self._acquisition_problem = acquisition_problem

  def _get_coroutine(self, features) -> sp.ModelCoroutine:
    return tuned_gp_models.VizierGaussianProcess.build_model(features).coroutine

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    """Update the list of completed trials."""
    del all_active
    self._trials.extend(copy.deepcopy(completed.trials))

  @property
  def _metric_info(self) -> vz.MetricInformation:
    return self._problem.metric_information.item()

  # TODO: Check the latency of `_generate_seed_trials` and look
  # into reducing it.
  @profiler.record_runtime
  def _generate_seed_trials(self, count: int) -> Sequence[vz.TrialSuggestion]:
    """Generate seed trials.

    The first seed trial is chosen as the search space center, the rest of the
    seed trials are chosen quasi-randomly.

    Arguments:
      count: The number of seed trials.

    Returns:
      The seed trials.
    """
    seed_suggestions = []
    if not self._trials:
      # TODO: Should track number of pending suggestions
      # so we don't suggest the center more than once.
      features = self._converter.to_features([])  # to extract shape.
      # NOTE: The code below assumes that a scaled value of 0.5 corresponds
      #   to the center of the feasible range.  This is true, but only by
      #   accident; ideally, we should get the center from the converters.
      parameters = self._converter.to_parameters(
          types.ModelInput(
              continuous=self._padding_schedule.pad_features(
                  0.5 * np.ones([1, features.continuous.shape[1]])
              ),
              categorical=self._padding_schedule.pad_features(
                  np.zeros(
                      [1, features.categorical.shape[1]], dtype=types.INT_DTYPE
                  )
              ),
          )
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

  @_experimental_override_allowed
  def _warp_labels(self, labels: types.Array) -> types.Array:
    """Subclasses can override this method for experiments."""
    labels = self._output_warper_pipeline.warp(labels)
    return labels.reshape([-1])

  @profiler.record_runtime
  def _trials_to_data(self, trials: Sequence[vz.Trial]) -> types.ModelData:
    """Convert trials to scaled features and warped labels."""
    model_data = self._converter.to_xy(trials)
    logging.info(
        'Transforming the labels of shape %s. Features has shape: %s',
        model_data.labels.padded_array.shape,
        types.ContinuousAndCategorical(
            model_data.features.continuous.padded_array.shape,
            model_data.features.categorical.padded_array.shape,
        ),
    )

    # Warp the output.
    unpad_labels = np.asarray(model_data.labels.unpad())
    warped_labels = self._warp_labels(unpad_labels)

    labels = types.PaddedArray.from_array(
        warped_labels,
        model_data.labels.padded_array.shape,
        fill_value=model_data.labels.fill_value,
    )
    logging.info('Transformed the labels. Now has shape: %s', labels.shape)
    return types.ModelData(model_data.features, labels)

  @_experimental_override_allowed
  def _optimize_params(self, data: types.ModelData) -> types.ParameterDict:
    # Convert trials to Numpy arrays. Labels are warped.
    coroutine = self._get_coroutine(data.features)
    self._rng, ard_rng = jax.random.split(self._rng, 2)
    model = sp.CoroutineWithData(coroutine, data)

    logging.info('CoroutineWithData: %s', model)

    best_params, _ = self._ard_optimizer(
        model.setup,
        model.loss_with_aux,
        ard_rng,
        constraints=model.constraints(),
        best_n=self._ensemble_size or 1,
    )
    return best_params

  @profiler.record_runtime
  def _update_state(self, data: types.ModelData) -> _GPBanditState:
    """Compute the designer's state.

    Args:
      data: Data to go into GP.

    Returns:
      GPBanditState object containing the designer's state.

    1. Convert trials to features and labels.
    2. Perform ARD to find best model parameters.
    3. Pre-compute the Cholesky decomposition.

    If no new trials were added since last call, no update will occur.
    """
    if len(self._trials) == self._incorporated_trials_count:
      # If there's no change in the number of completed trials, don't update
      # state. The assumption is that trials can't be removed.
      return self._last_computed_state
    self._incorporated_trials_count = len(self._trials)

    coroutine = self._get_coroutine(data.features)
    best_params = self._optimize_params(data)
    best_models = sp.StochasticProcessWithCoroutine(coroutine, best_params)
    # Logging for debugging purposes.
    logging.info(
        'Best models: %s', eqx.tree_pformat(best_models, short_arrays=False)
    )
    predictive = sp.UniformEnsemblePredictive(
        eqx.filter_jit(best_models.precompute_predictive)(data)
    )
    self._last_computed_state = (predictive, data)
    return self._last_computed_state

  @_experimental_override_allowed
  @profiler.record_runtime
  def _optimize_acquisition(
      self, scoring_fn: acquisitions.BayesianScoringFunction, count: int
  ) -> list[vz.Trial]:
    start_time = datetime.datetime.now()
    # Set up optimizer and run
    seed_features = vb.trials_to_sorted_array(
        self._trials, self._padded_one_hot_converter
    )
    seed_features_unpad = vb.trials_to_sorted_array(
        self._trials, self._one_hot_converter
    )
    acq_rng, self._rng = jax.random.split(self._rng)

    # TODO: Remove this when Vectorized Optimizer works on CACV.
    cacpa = self._converter.to_features(self._trials)
    one_hot_to_modelinput = gp_bandit_utils.make_one_hot_to_modelinput_fn(
        seed_features_unpad, self._feature_mapper, cacpa
    )

    score = lambda xs: scoring_fn.score(one_hot_to_modelinput(xs))
    score_with_aux = lambda xs: scoring_fn.score_with_aux(
        one_hot_to_modelinput(xs)
    )

    best_candidates: vb.VectorizedStrategyResults = eqx.filter_jit(
        self._acquisition_optimizer
    )(
        score,
        prior_features=seed_features,
        count=count,
        seed=acq_rng,
        score_with_aux_fn=score_with_aux,
    )

    # TODO: Move the logging into `VectorizedOptimizer`.
    logging.info(
        (
            'Optimization completed. Duration: %s. Evaluations: %s. Best'
            ' Results: %s'
        ),
        datetime.datetime.now() - start_time,
        (
            (
                self._acquisition_optimizer.max_evaluations
                // self._acquisition_optimizer.suggestion_batch_size
            )
            * self._acquisition_optimizer.suggestion_batch_size
        ),
        best_candidates,
    )
    optimal_features = best_candidates.features
    best_candidates = dataclasses.replace(
        best_candidates, features=optimal_features
    )

    # Convert best_candidates (in scaled space) into suggestions (in unscaled
    # space); also append debug information like model predictions.
    logging.info('Converting the optimization result into suggestions...')
    return vb.best_candidates_to_trials(
        best_candidates, self._one_hot_converter
    )  # [N, D]

  @profiler.record_runtime
  def suggest(self, count: int = 1) -> Sequence[vz.TrialSuggestion]:
    logging.info('Suggest called with count=%d', count)
    if count > 1:
      logging.warning(
          'GAUSSIAN_PROCESS_BANDIT currently is not optimized for batched'
          ' suggestions. Suggestions in the batch are likely to be very'
          ' similar.'
      )
    if len(self._trials) < self._num_seed_trials:
      return self._generate_seed_trials(count)

    suggest_start_time = datetime.datetime.now()
    logging.info('Updating the designer state based on trials...')
    predictive, data = self._update_state(self._trials_to_data(self._trials))

    # Define acquisition function.
    scoring_fn = acquisitions.BayesianScoringFunction(
        predictive,
        data,
        self._acquisition_function,
        acquisitions.TrustRegion(data.features)
        if self._use_trust_region
        else None,
    )
    logging.info('Optimizing acquisition: %s', scoring_fn)
    best_trials = self._optimize_acquisition(scoring_fn, count)

    suggestions = []
    for t in best_trials:
      metadata = t.metadata.ns(self._metadata_ns).ns('devinfo')
      metadata['time_spent'] = f'{datetime.datetime.now() - suggest_start_time}'
      suggestions.append(
          vz.TrialSuggestion(parameters=t.parameters, metadata=t.metadata)
      )
    return suggestions

  @profiler.record_runtime
  def predict(
      self,
      trials: Sequence[vz.TrialSuggestion],
      rng: Optional[jax.random.KeyArray] = None,
      num_samples: Optional[int] = None,
  ) -> vza.Prediction:
    """Returns the mean and stddev for any given trials.

    The method performs sampling of the warped GP model, unwarp the samples and
    compute the empirical mean and stadard deviation as an apprixmation.

    Arguments:
      trials: The trials where the predictions will be made.
      rng: The sampling random key used for approximation.
      num_samples: The number of samples used for the approximation.

    Returns:
      The predictions in the specified trials.
    """
    if rng is None:
      rng = jax.random.PRNGKey(0)
    if num_samples is None:
      num_samples = 1000

    predictive, _ = self._update_state(self._trials_to_data(self._trials))

    xs = self._converter.to_features(trials)
    xs = types.ModelInput(
        continuous=xs.continuous.replace_fill_value(0.0),
        categorical=xs.categorical.replace_fill_value(0),
    )

    samples = eqx.filter_jit(acquisitions.sample_from_predictive)(
        predictive, xs, num_samples, key=rng
    )  # (num_samples, batch_size)
    samples = samples[
        :, ~(xs.continuous.is_missing[0] | xs.categorical.is_missing[0])
    ]
    unwarped_samples = None
    # TODO: vectorize output warping.
    for i in range(samples.shape[0]):
      unwarp_samples_ = self._output_warper_pipeline.unwarp(
          samples[i][..., np.newaxis]
      ).reshape(-1)
      if unwarped_samples is not None:
        unwarped_samples = np.vstack([unwarp_samples_, unwarped_samples])
      else:
        unwarped_samples = unwarp_samples_

    mean = np.mean(unwarped_samples, axis=0)
    stddev = np.std(unwarped_samples, axis=0)
    return vza.Prediction(mean=mean, stddev=stddev)
