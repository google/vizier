# Copyright 2024 Google LLC.
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
import jax.numpy as jnp
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers import scalarization
from vizier._src.algorithms.designers.gp import acquisitions as acq_lib
from vizier._src.algorithms.designers.gp import gp_models
from vizier._src.algorithms.designers.gp import output_warpers
from vizier._src.algorithms.optimizers import eagle_strategy as es
from vizier._src.algorithms.optimizers import lbfgsb_optimizer as lo
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier.jax import optimizers
from vizier.pyvizier import converters
from vizier.pyvizier.converters import padding
from vizier.utils import profiler

default_acquisition_optimizer_factory = vb.VectorizedOptimizerFactory(
    strategy_factory=es.VectorizedEagleStrategyFactory(
        eagle_config=es.EagleStrategyConfig()
    ),
    max_evaluations=75_000,
    suggestion_batch_size=25,
)

default_scoring_function_factory = acq_lib.bayesian_scoring_function_factory(
    lambda _: acq_lib.UCB()
)


def _experimental_override_allowed(fun):
  """No-op.

  Marks functions that can be easily overridden for experimentation.

  Args:
    fun:

  Returns:
    fun:
  """
  return fun


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
    ard_random_restarts: The number of random initializations to run GP
      hyper-parameter optimization with.
    num_seed_trials: If greater than zero, first trial is the center of the
      search space. Afterwards, uses quasirandom until this number of trials are
      observed.
    scoring_function_factory: Callable that returns the scoring function to use.
    use_trust_region: Uses trust region to constrain initial exploration.
    rng: If not set, uses random numbers.
    metadata_ns: Metadata namespace that this designer writes to.
  """

  _problem: vz.ProblemStatement = attr.field(kw_only=False)
  _acquisition_optimizer_factory: vb.VectorizedOptimizerFactory = attr.field(
      kw_only=True,
      factory=lambda: default_acquisition_optimizer_factory,
  )
  _ard_optimizer: optimizers.Optimizer[types.ParameterDict] = attr.field(
      factory=optimizers.default_optimizer,
      kw_only=True,
  )
  _ard_random_restarts: int = attr.field(
      default=optimizers.DEFAULT_RANDOM_RESTARTS, kw_only=True
  )
  _num_seed_trials: int = attr.field(default=1, kw_only=True)
  _linear_coef: float = attr.field(default=0.0, kw_only=True)
  _scoring_function_factory: acq_lib.ScoringFunctionFactory = attr.field(
      factory=lambda: default_scoring_function_factory,
      kw_only=True,
  )
  _scoring_function_is_parallel: bool = attr.field(default=False, kw_only=True)
  # Whether to pad all inputs, and what type of schedule to use. This is to
  # ensure fewer JIT compilation passes. (Default implies no padding.)
  _padding_schedule: padding.PaddingSchedule = attr.field(
      factory=padding.PaddingSchedule, kw_only=True
  )
  _use_trust_region: bool = attr.field(default=True, kw_only=True)
  _rng: jax.Array = attr.field(
      factory=lambda: jax.random.PRNGKey(random.getrandbits(32)), kw_only=True
  )
  _metadata_ns: str = attr.field(
      default='oss_gp_bandit', kw_only=True, init=False
  )
  _ensemble_size: Optional[int] = attr.field(default=1, kw_only=True)
  _output_warper: output_warpers.OutputWarper = attr.field(
      factory=output_warpers.create_default_warper, kw_only=True
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

  _last_computed_gp: gp_models.GPState = attr.field(init=False)

  # The prior GP used in transfer learning. `last_computed_gp` is trained
  # on the residuals of `_prior_gp`, if one is trained.
  _prior_gp: Optional[gp_models.GPState] = attr.field(init=False, default=None)

  def __attrs_post_init__(self):
    # Extra validations
    if self._problem.search_space.is_conditional:
      raise ValueError(f'{type(self)} does not support conditional search.')
    if self._problem.search_space.num_parameters() == 0:
      raise ValueError(
          'SearchSpace should contain at least one parameter config.'
      )

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

    self._acquisition_optimizer = self._acquisition_optimizer_factory(
        self._converter
    )
    self._acquisition_problem = copy.deepcopy(self._problem)
    empty_data = types.ModelData(
        features=self._converter.to_features([]),
        labels=types.PaddedArray.as_padded(
            np.zeros((0, len(self._problem.metric_information)))
        ),
    )

    # Additional validations
    coroutine = gp_models.get_vizier_gp_coroutine(empty_data)
    params = sp.CoroutineWithData(coroutine, empty_data).setup(self._rng)
    model = sp.StochasticProcessWithCoroutine(coroutine, params)
    predictive = sp.UniformEnsemblePredictive(
        eqx.filter_jit(model.precompute_predictive)(empty_data)
    )
    scoring_fn = self._scoring_function_factory(
        empty_data, predictive, self._use_trust_region
    )
    if (
        isinstance(scoring_fn, acq_lib.MaxValueEntropySearch)
        and self._ensemble_size > 1
    ):
      raise ValueError(
          'MaxValueEntropySearch is not supported with ensemble '
          'size greater than one.'
      )

    acquisition_function = getattr(scoring_fn, 'acquisition_fn', None)
    self._acquisition_problem.metric_information = vz.MetricsConfig()
    if isinstance(acquisition_function, acq_lib.MultiAcquisitionFunction):
      for k in acquisition_function.acquisition_fns.keys():
        metric = vz.MetricInformation(k, goal=vz.ObjectiveMetricGoal.MAXIMIZE)
        self._acquisition_problem.metric_information.append(metric)
    else:
      metric = vz.MetricInformation(
          'acquisition', goal=vz.ObjectiveMetricGoal.MAXIMIZE
      )
      self._acquisition_problem.metric_information.append(metric)

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    """Update the list of completed trials."""
    del all_active
    self._trials.extend(copy.deepcopy(completed.trials))

  def set_priors(self, prior_studies: Sequence[vza.CompletedTrials]) -> None:
    """Updates the list of prior studies for transfer learning.

    Each element is treated as a new prior study, and will be stacked in order
    received - i.e. the first entry is for the first GP, the second entry is for
    the GP trained on the residuals of the first GP, etc.

    See section 3.3 of https://dl.acm.org/doi/10.1145/3097983.3098043 for more
    information, or see `gp/gp_models.py` and `gp/transfer_learning.py`

    Transfer learning is resilient to bad priors.

    Multiple calls are permitted, but unadvised. Each call will trigger
    retraining of the prior GPs - on only the state provided to `set_priors`.
    State is not incrementally updated.

    TODO: Decide on whether this method should become part of an
    interface.

    Args:
      prior_studies: A list of lists of completed trials, with one list per
        prior study. The designer will train a prior GP for each list of prior
        trials (for each `CompletedStudy` entry), in the order received.
    """
    self._rng, ard_rng = jax.random.split(self._rng)
    prior_data = [
        self._trials_to_data(prior_study.trials)
        for prior_study in prior_studies
    ]
    self._prior_gp = self._train_prior_gp(priors=prior_data, ard_rng=ard_rng)

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
      continuous = self._padding_schedule.pad_features(
          0.5 * np.ones([1, features.continuous.shape[1]])
      )
      categorical = self._padding_schedule.pad_features(
          np.zeros([1, features.categorical.shape[1]], dtype=types.INT_DTYPE)
      )
      model_input = types.ModelInput(continuous, categorical)
      parameters = self._converter.to_parameters(model_input)[0]
      suggestion = vz.TrialSuggestion(
          parameters, metadata=vz.Metadata({'seeded': 'center'})
      )
      seed_suggestions.append(suggestion)
    with profiler.timeit('quasi_random_sampler_seed_trials'):
      if (remaining_counts := count - len(seed_suggestions)) > 0:
        seed_suggestions.extend(
            self._quasi_random_sampler.suggest(remaining_counts)
        )
    return seed_suggestions

  @_experimental_override_allowed
  def _warp_labels(self, labels: types.Array) -> types.Array:
    """Subclasses can override this method for experiments."""
    return np.concatenate(
        [
            self._output_warper.warp(labels[:, i : i + 1])
            for i in range(labels.shape[1])
        ],
        axis=-1,
    )

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
  def _create_gp_spec(
      self, data: types.ModelData, ard_rng: jax.Array
  ) -> gp_models.GPTrainingSpec:
    """Overrideable creation of a training spec for a GP model."""
    return gp_models.GPTrainingSpec(
        ard_optimizer=self._ard_optimizer,
        ard_rng=ard_rng,
        coroutine=gp_models.get_vizier_gp_coroutine(
            data=data, linear_coef=self._linear_coef
        ),
        ensemble_size=self._ensemble_size,
        ard_random_restarts=self._ard_random_restarts,
    )

  @_experimental_override_allowed
  def _train_prior_gp(
      self,
      priors: Sequence[types.ModelData],
      ard_rng: jax.Array,
  ):
    """Trains a transfer-learning-enabled GP with prior studies.

    Args:
      priors: Data for each sequential prior to train for transfer learning.
        Assumed to be in order of training, i.e. element 0 is priors[0] is the
        first GP trained, and priors[1] trains a GP on the residuals of the GP
        trained on priors[0], and so on.
      ard_rng: RNG to do ARD to optimize GP parameters.

    Returns:
      A trained pre-computed ensemble GP.
    """
    ard_rngs = jax.random.split(ard_rng, len(priors))

    # Order `specs` in training order, i.e. `specs[0]` is trained first.
    specs = [
        self._create_gp_spec(prior_data, ard_rngs[i])
        for i, prior_data in enumerate(priors)
    ]

    # `train_gp` expects `specs` and `data` in training order, which is how
    # they were prepared above.
    return gp_models.train_gp(spec=specs, data=priors)

  @profiler.record_runtime
  def _update_gp(self, data: types.ModelData) -> gp_models.GPState:
    """Compute the designer's GP and caches the result. No-op without new data.

    Args:
      data: Data to go into GP.

    Returns:
      `GPState` object containing the trained GP.

    1. Convert trials to features and labels.
    2. Trains a pre-computed ensemble GP.

    If no new trials were added since last call, no update will occur.
    """
    if len(self._trials) == self._incorporated_trials_count:
      # If there's no change in the number of completed trials, don't update
      # state. The assumption is that trials can't be removed.
      return self._last_computed_gp
    self._incorporated_trials_count = len(self._trials)

    self._rng, ard_rng = jax.random.split(self._rng, 2)
    spec = self._create_gp_spec(data, ard_rng)
    if self._prior_gp:
      self._last_computed_gp = gp_models.train_stacked_residual_gp(
          base_gp=self._prior_gp, spec=spec, data=data
      )
    else:
      self._last_computed_gp = gp_models.train_gp(spec=spec, data=data)

    return self._last_computed_gp

  @_experimental_override_allowed
  @profiler.record_runtime
  def _optimize_acquisition(
      self, scoring_fn: acq_lib.BayesianScoringFunction, count: int
  ) -> list[vz.Trial]:
    jax.monitoring.record_event(
        '/vizier/jax/gp_bandit/optimize_acquisition/called'
    )
    # Set up optimizer and run
    seed_features = vb.trials_to_sorted_array(self._trials, self._converter)
    acq_rng, self._rng = jax.random.split(self._rng)

    score = scoring_fn.score
    score_with_aux = scoring_fn.score_with_aux

    n_parallel = None
    if self._scoring_function_is_parallel:
      n_parallel = count
      count = 1

    acquisition_optimizer = self._acquisition_optimizer
    if not isinstance(acquisition_optimizer, lo.LBFGSBOptimizer):
      acquisition_optimizer = eqx.filter_jit(acquisition_optimizer)
    best_candidates: vb.VectorizedStrategyResults = acquisition_optimizer(
        eqx.filter_jit(score),
        prior_features=seed_features,
        count=count,
        seed=acq_rng,
        score_with_aux_fn=eqx.filter_jit(score_with_aux),
        n_parallel=n_parallel,
    )

    best_candidates = dataclasses.replace(
        best_candidates, features=best_candidates.features
    )

    # Convert best_candidates (in scaled space) into suggestions (in unscaled
    # space); also append debug information like model predictions. Output shape
    # [N, D].
    logging.info('Converting the optimization result into suggestions...')
    return vb.best_candidates_to_trials(best_candidates, self._converter)

  @profiler.record_runtime
  def suggest(self, count: int = 1) -> Sequence[vz.TrialSuggestion]:
    logging.info('Suggest called with count=%d', count)
    if count > 1 and not self._scoring_function_is_parallel:
      logging.warning(
          'GAUSSIAN_PROCESS_BANDIT currently is not optimized for batched'
          ' suggestions. Suggestions in the batch are likely to be very'
          ' similar.'
      )
    if len(self._trials) < self._num_seed_trials:
      return self._generate_seed_trials(count)

    suggest_start_time = datetime.datetime.now()
    logging.info('Updating the designer state based on trials...')
    data = self._trials_to_data(self._trials)
    gp = self._update_gp(data)

    # Define acquisition function.
    scoring_fn = self._scoring_function_factory(
        data, gp, self._use_trust_region
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
  def sample(
      self,
      trials: Sequence[vz.TrialSuggestion],
      rng: Optional[jax.Array] = None,
      num_samples: Optional[int] = None,
  ) -> types.Array:
    """Returns unwarped samples from the model for any given trials.

    Arguments:
      trials: The trials where the predictions will be made.
      rng: The sampling random key.
      num_samples: The number of samples per trial.

    Returns:
      The samples in the specified trials. shape: (num_samples, num_trials)
    """
    if rng is None:
      rng = jax.random.PRNGKey(0)
    if num_samples is None:
      num_samples = 1000

    if not trials:
      return np.zeros((num_samples, 0))

    data = self._trials_to_data(self._trials)
    gp = self._update_gp(data)
    xs = self._converter.to_features(trials)
    xs = types.ModelInput(
        continuous=xs.continuous.replace_fill_value(0.0),
        categorical=xs.categorical.replace_fill_value(0),
    )
    samples = eqx.filter_jit(acq_lib.sample_from_predictive)(
        gp, xs, num_samples, key=rng
    )  # (num_samples, num_trials)
    # Scope the samples to non-padded only (there's a single padded dimension).
    samples = samples[
        :, ~(xs.continuous.is_missing[0] | xs.categorical.is_missing[0])
    ]
    unwarped_samples = None
    # TODO: vectorize output warping.
    for i in range(samples.shape[0]):
      unwarp_samples_ = self._output_warper.unwarp(
          samples[i][..., np.newaxis]
      ).reshape(-1)
      if unwarped_samples is not None:
        unwarped_samples = np.vstack([unwarp_samples_, unwarped_samples])
      else:
        unwarped_samples = unwarp_samples_

    return unwarped_samples  # pytype: disable=bad-return-type

  @profiler.record_runtime
  def predict(
      self,
      trials: Sequence[vz.TrialSuggestion],
      rng: Optional[jax.Array] = None,
      num_samples: Optional[int] = 1000,
  ) -> vza.Prediction:
    """Returns the mean and stddev for any given trials.

    The method performs sampling of the warped GP model, unwarp the samples and
    compute the empirical mean and standard deviation as an apprixmation.

    Arguments:
      trials: The trials where the predictions will be made.
      rng: The sampling random key used for approximation.
      num_samples: The number of samples used for the approximation.

    Returns:
      The predictions in the specified trials.
    """
    unwarped_samples = self.sample(trials, rng, num_samples)
    mean = np.mean(unwarped_samples, axis=0)
    stddev = np.std(unwarped_samples, axis=0)
    return vza.Prediction(mean=mean, stddev=stddev)

  @classmethod
  def from_problem(
      cls,
      problem: vz.ProblemStatement,
      seed: Optional[int] = None,
      num_scalarizations: int = 1000,
      reference_scaling: float = 0.01,
      **kwargs,
  ) -> 'VizierGPBandit':
    rng = jax.random.PRNGKey(seed or 0)
    # Linear coef is set to 1.0 as prior and uses VizierLinearGaussianProcess
    # which uses a sum of Matern and linear but ARD still tunes its amplitude.
    if problem.is_single_objective:
      return cls(problem, linear_coef=1.0, rng=rng, **kwargs)
    else:
      num_obj = len(problem.metric_information.of_type(vz.MetricType.OBJECTIVE))
      rng, weights_rng = jax.random.split(rng)
      weights = jnp.abs(
          jax.random.normal(weights_rng, shape=(num_scalarizations, num_obj))
      )

      def _scalarized_ucb(data: types.ModelData) -> acq_lib.AcquisitionFunction:
        scalarizer = scalarization.HyperVolumeScalarization(
            weights,
            acq_lib.get_reference_point(data.labels, scale=reference_scaling),
        )
        return acq_lib.ScalarizedAcquisition(
            acq_lib.UCB(),
            scalarizer,
            reduction_fn=lambda x: jnp.mean(x, axis=0),
        )

      scoring_function_factory = acq_lib.bayesian_scoring_function_factory(
          _scalarized_ucb
      )
      return cls(
          problem,
          linear_coef=1.0,
          scoring_function_factory=scoring_function_factory,
          scoring_function_is_parallel=True,
          use_trust_region=False,
          rng=rng,
          **kwargs,
      )
