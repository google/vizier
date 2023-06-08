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
from typing import Any, Optional, Sequence, Union

from absl import logging
import attr
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
from vizier._src.jax import predictive_fns
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier._src.jax.models import tuned_gp_models
from vizier.jax import optimizers
from vizier.pyvizier import converters
from vizier.pyvizier.converters import feature_mapper
from vizier.pyvizier.converters import padding
from vizier.utils import json_utils
from vizier.utils import profiler


# Define top-level JIT-ed versions of some library functions. See
# `gp_bandit_utils.py` for function documentation.
jit_precompute_cholesky = profiler.record_runtime(
    jax.jit(gp_bandit_utils.precompute_cholesky, static_argnames=('model',)),
    name_prefix='VizierGPBandit',
    name='precompute_cholesky',
)

jit_vmap_precompute_cholesky = profiler.record_runtime(
    jax.jit(
        jax.vmap(gp_bandit_utils.precompute_cholesky, in_axes=(None, None, 0)),
        static_argnames=('model',),
    ),
    name_prefix='VizierGPBandit',
    name='vmap_precompute_cholesky',
)

jit_optimize_acquisition = profiler.record_runtime(
    jax.jit(
        gp_bandit_utils.optimize_acquisition,
        static_argnames=('count', 'model', 'use_vmap', 'mapper'),
    ),
    name_prefix='VizierGPBandit',
    name='optimize_acquisition',
)


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
    acquisition_builder: An acquisition builder instance specifying the
      acqusition function to use.
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
  _use_categorical_kernel: bool = attr.field(default=False, kw_only=True)
  # Whether to pad all inputs, and what type of schedule to use. This is to
  # ensure fewer JIT compilation passes.
  _padding_schedule: Optional[padding.PaddingSchedule] = attr.field(
      default=None, kw_only=True
  )
  _use_trust_region: bool = attr.field(default=True, kw_only=True)
  _rng: jax.random.KeyArray = attr.field(
      factory=lambda: jax.random.PRNGKey(random.getrandbits(32)), kw_only=True
  )
  _metadata_ns: str = attr.field(
      default='oss_gp_bandit', kw_only=True, init=False
  )

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
  _feature_mapper: Optional[
      feature_mapper.ContinuousCategoricalFeatureMapper
  ] = attr.field(init=False, default=None)
  _last_computed_state: types.GPState = attr.field(init=False)

  default_acquisition_optimizer_factory = vb.VectorizedOptimizerFactory(
      strategy_factory=es.VectorizedEagleStrategyFactory()
  )

  def __attrs_post_init__(self):
    # Extra validations
    if self._problem.search_space.is_conditional:
      raise ValueError(f'{type(self)} does not support conditional search.')
    elif len(self._problem.metric_information) != 1:
      raise ValueError(f'{type(self)} works with exactly one metric.')
    if self._use_categorical_kernel and self._padding_schedule:
      raise ValueError(
          f'{type(self)} does not support padding with categorical kernel.'
          f' use_categorical_kernel = {self._use_categorical_kernel},'
          f' padding_schedule = {self._padding_schedule}.'
      )
    # Extra initializations.
    # Discrete parameters are continuified to account for their actual values.
    if self._padding_schedule:
      self._converter = (
          converters.PaddedTrialToArrayConverter.from_study_config(
              self._problem,
              scale=True,
              pad_oovs=True,
              padding_schedule=self._padding_schedule,
              max_discrete_indices=0,
              flip_sign_for_minimization_metrics=True,
          )
      )
    else:
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
    self._output_warper_pipeline = output_warpers.create_default_warper()
    if self._use_categorical_kernel:
      self._feature_mapper = feature_mapper.ContinuousCategoricalFeatureMapper(
          self._converter
      )
    self._acquisition_optimizer = self._acquisition_optimizer_factory(
        self._converter
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

  def _build_model(self, features):
    if self._use_categorical_kernel:
      return tuned_gp_models.VizierGaussianProcessWithCategorical.build_model(
          features
      )
    return tuned_gp_models.VizierGaussianProcess.build_model(features)

  @property
  def _use_vmap(self):
    """Returns whether ensemble of parameters is used which requires vmap."""
    # Derived classes of `optimizers.Optimizer` have a `best_n` property that
    # indicates whether the optimizer trains an ensemble of parameters and
    # therefore whether vmap should be used.
    # If the optimizer has a `best_n` property with value greater than 1, vmap
    # is used.
    return getattr(self._ard_optimizer, 'best_n', -1) > 1

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
  @profiler.record_runtime(
      name_prefix='VizierGPBandit', name='generate_seed_trials'
  )
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

  @profiler.record_runtime(
      name_prefix='VizierGPBandit', name='convert_trials_to_arrays'
  )
  def _convert_trials_to_arrays(
      self, trials: Sequence[vz.Trial]
  ) -> tuple[
      Union[types.MaybePaddedArray, types.ContinuousAndCategoricalArray],
      types.MaybePaddedArray,
  ]:
    """Convert trials to scaled features and warped labels."""
    features, pre_labels = self._converter.to_xy(self._trials)
    logging.info(
        'Transforming the labels of shape %s. Features has shape: %s',
        pre_labels.shape,
        features.shape,
    )
    # Map to continuous and categorical.
    if self._use_categorical_kernel:
      features = self._feature_mapper.map(features)

    labels = pre_labels
    if self._padding_schedule:
      # Labels coming from the converter will have shape [P, 1], where P >= T
      # the number of trials.
      padded_labels = pre_labels.padded_array
      # Unpad labels to have shape [T, 1].
      # This is because the warper may take into account all labels, which may
      # be NaN.
      labels = padded_labels[: len(self._trials)]

    # Warp the output.
    labels = self._output_warper_pipeline.warp(labels)
    labels = labels.reshape([-1])

    # Pad back
    if self._padding_schedule:
      # Pad back to shape [P, 1].
      padded_labels = np.concatenate(
          [labels, padded_labels[len(self._trials) :, 0]], axis=0
      )
      labels = padding.PaddedArray(
          padded_array=padded_labels, is_missing=pre_labels.is_missing
      )
    logging.info('Transformed the labels. Now has shape: %s', labels.shape)
    return features, labels

  @profiler.record_runtime(
      name_prefix='VizierGPBandit', name='ard', also_log=True
  )
  def _find_best_model_params(
      self,
      model: sp.StochasticProcessModel,
      loss_fn: optimizers.LossFunction,
      data: types.StochasticProcessModelData,
      seed: jax.random.KeyArray,
  ) -> tuple[types.ParameterDict, Any]:
    """Perform ARD on the current model to find best model parameters."""
    # Run ARD.
    setup = functools.partial(
        jax.jit(
            gp_bandit_utils.stochastic_process_model_setup,
            static_argnames=('model',),
        ),
        model=model,
        data=data,
    )
    constraints = sp.get_constraints(model)

    logging.info('Optimizing the loss function...')

    # The ARD optimizers JIT the train step/loop internally.
    return self._ard_optimizer(setup, loss_fn, seed, constraints=constraints)

  @profiler.record_runtime(name_prefix='VizierGPBandit', name='compute_state')
  def _compute_state(
      self,
  ) -> tuple[types.GPState, Optional[acquisitions.TrustRegion]]:
    """Compute the designer's state.

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
    # Convert trials to Numpy arrays. Labels are warped.
    features, labels = self._convert_trials_to_arrays(self._trials)

    dimension_is_missing = None
    label_is_missing = None
    if self._padding_schedule:
      dimension_is_missing = features.is_missing[1]
      label_is_missing = labels.is_missing[0]
      features = features.padded_array
      labels = labels.padded_array
    # Update the model.
    # TODO: Add support for PaddedArrays instead of passing in
    # masks.
    data = types.StochasticProcessModelData(
        features=features,
        labels=labels,
        label_is_missing=label_is_missing,
        dimension_is_missing=dimension_is_missing,
    )
    model = self._build_model(features)
    # TODO: Avoid retracing vmapped loss when loss function API is
    # redesigned.
    loss_fn = functools.partial(
        jax.jit(
            gp_bandit_utils.stochastic_process_model_loss_fn,
            static_argnames=('model', 'normalize'),
        ),
        model=model,
        data=data,
        # For SGD, normalize the loss so we can use the same learning rate
        # regardless of the number of examples (see
        # `OptaxTrainWithRandomRestarts` docstring).
        normalize=isinstance(
            self._ard_optimizer, optimizers.OptaxTrainWithRandomRestarts
        ),
    )
    self._rng, ard_rng = jax.random.split(self._rng, 2)
    best_model_params, metrics = self._find_best_model_params(
        model, loss_fn, data, ard_rng
    )
    # Logging for debugging purposes.
    logging.info('Best model parameters: %s', best_model_params)

    # ARD optimizer metrics dicts are assumed to have a 'loss' field.
    logging.info('All losses: %s', metrics['loss'])
    logging.info('ARD best loss: %s', np.min(metrics['loss']))

    if self._use_vmap:
      precompute_cholesky_fn = jit_vmap_precompute_cholesky
    else:
      precompute_cholesky_fn = jit_precompute_cholesky
    # `pp_state` contains intermediates that are expensive to compute, depend
    # only on observed (not predictive) index points, and are needed to compute
    # the posterior predictive GP (i.e. the Cholesky factor of the kernel matrix
    # over observed index points). `pp_state` is passed to
    # `model.posterior_predictive` to avoid re-computing the intermediates
    # unnecessarily.
    _, pp_state = precompute_cholesky_fn(model, data, best_model_params)
    model_state = {'params': best_model_params, **pp_state}

    trust_region = None
    if self._use_trust_region:
      trust_region = acquisitions.TrustRegion.build(
          self._converter.output_specs,
          data=data,
      )
    state = types.GPState(data=data, model_state=model_state), trust_region
    self._last_computed_state = state
    return state

  def _optimize_acquisition(
      self,
      count: int,
      state: types.GPState,
      trust_region: Optional[acquisitions.TrustRegion] = None,
  ) -> list[vz.Trial]:
    start_time = datetime.datetime.now()
    acq_rng, self._rng = jax.random.split(self._rng)

    prior_features = vb.trials_to_sorted_array(self._trials, self._converter)
    model = self._build_model(state.data.features)
    suggestions = jit_optimize_acquisition(
        count=count,
        model=model,
        acquisition_fn=self._acquisition_function,
        optimizer=self._acquisition_optimizer,
        prior_features=prior_features,
        state=state,
        seed=acq_rng,
        use_vmap=self._use_vmap,
        trust_region=trust_region,
        mapper=self._feature_mapper,
    )

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
        suggestions,
    )
    return vb.best_candidates_to_trials(suggestions, self._converter)

  @profiler.record_runtime(name_prefix='VizierGPBandit')
  def suggest(self, count: int = 1) -> Sequence[vz.TrialSuggestion]:
    logging.info('Suggest called with count=%d', count)
    if count > 1:
      logging.warning(
          'GAUSSIAN_PROCESS_BANDIT currently is not optimized for batched'
          ' suggestions. Suggestions in the batch are likely to be very'
          ' similar.'
      )
    suggest_start_time = datetime.datetime.now()
    if len(self._trials) < self._num_seed_trials:
      return self._generate_seed_trials(count)

    logging.info('Updating the designer state based on trials...')
    state, _ = self._compute_state()

    logging.info('Optimizing acquisition...')
    best_candidates = self._optimize_acquisition(count, state)

    # Convert best_candidates (in scaled space) into suggestions (in unscaled
    # space); also append debug information like model predictions.
    logging.info('Converting the optimization result into suggestions...')
    optimal_features = self._converter.to_features(best_candidates)  # [N, D]
    if self._padding_schedule:
      optimal_features = optimal_features.padded_array
    if self._use_categorical_kernel:
      optimal_features = self._feature_mapper.map(optimal_features)
    # Make predictions (in the warped space). [N]
    model = self._build_model(state.data.features)
    with profiler.record_runtime_context(
        'VizierGPBandit.predict_on_suggestions'
    ):
      predictions = jax.jit(
          predictive_fns.predict_on_array, static_argnames=('model', 'use_vmap')
      )(
          optimal_features,
          model=model,
          state=state,
          use_vmap=self._use_vmap,
      )
    predict_mean = predictions['mean']  # [N,]
    predict_stddev = predictions['stddev']  # [N,]
    # Possibly unpad predictions
    predict_mean = predict_mean[: len(self._trials)]
    predict_stddev = predict_stddev[: len(self._trials)]
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
              'mean': predict_mean[i],
              'stddev': predict_stddev[i],
              'acquisition': acquisition,
          },
          cls=json_utils.NumpyEncoder,
      )
      metadata['time_spent'] = f'{datetime.datetime.now() - suggest_start_time}'
      suggestions.append(
          vz.TrialSuggestion(candidate.parameters, metadata=candidate.metadata)
      )

    return suggestions

  @profiler.record_runtime(name_prefix='VizierGPBandit')
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

    state, _ = self._compute_state()
    model = self._build_model(state.data.features)
    xs = self._converter.to_features(trials)
    if self._padding_schedule:
      xs = xs.padded_array[: len(trials), ...]
    if self._use_categorical_kernel:
      xs = self._feature_mapper.map(xs)

    samples = jax.jit(
        predictive_fns.sample_on_array,
        static_argnames=('model', 'use_vmap', 'num_samples'),
    )(
        xs,
        num_samples=num_samples,
        key=rng,
        model=model,
        state=state,
        use_vmap=self._use_vmap,
    )  # (num_samples, batch_size)
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
