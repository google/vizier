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
from typing import Sequence, Optional

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
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier._src.jax.models import tuned_gp_models
from vizier._src.jax.optimizers import optimizers
from vizier.pyvizier import converters
from vizier.utils import json_utils
from vizier.utils import profiler


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
  _acquisition_optimizer: vb.VectorizedOptimizer = attr.field(
      kw_only=True,
      factory=lambda: VizierGPBandit.default_acquisition_optimizer,
  )
  _ard_optimizer: optimizers.Optimizer[types.ParameterDict] = attr.field(
      factory=lambda: VizierGPBandit.default_ard_optimizer_noensemble,
      kw_only=True,
  )
  _num_seed_trials: int = attr.field(default=1, kw_only=True)
  _acquisition_builder: acquisitions.AcquisitionBuilder = attr.field(
      factory=acquisitions.GPBanditAcquisitionBuilder, kw_only=True
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
  # Numpy array representing the current trials' features.
  _features: types.Array = attr.field(init=False)
  # Numpy array representing the current recent trials' metrics.
  _labels: types.Array = attr.field(init=False)
  # The current designer state (Cholesky decomposition, model params).
  _state: types.ModelState = attr.field(init=False)
  # The current GP model.
  _model: sp.StochasticProcessModel = attr.field(init=False)
  _output_warper_pipeline: output_warpers.OutputWarperPipeline = attr.field(
      init=False
  )
  # ------------------------------------------------------------------
  # Below are class contants which are not attr fields.
  # ------------------------------------------------------------------
  # Only one of these optimizers will be used.
  # `default_ard_optimizer_ensemble` returns the best 5 parameter values for
  # ensembling, while `default_ard_optimizer_noensemble` returns only
  # the single best parameter value.
  default_ard_optimizer_ensemble = optimizers.JaxoptLbfgsB(
      random_restarts=8, best_n=5
  )
  default_ard_optimizer_noensemble = optimizers.JaxoptLbfgsB(
      random_restarts=4, best_n=1
  )
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
    self._output_warper_pipeline = output_warpers.create_default_warper()

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
  ) -> tuple[types.Array, types.Array]:
    """Convert trials to scaled features and warped labels."""
    features, labels = self._converter.to_xy(self._trials)
    logging.info(
        'Transforming the labels of shape %s. Features has shape: %s',
        labels.shape,
        features.shape,
    )
    # Warp the output.
    labels = self._output_warper_pipeline.warp(labels)
    labels = labels.reshape([-1])
    logging.info('Transformed the labels. Now has shape: %s', labels.shape)
    return features, labels

  @profiler.record_runtime(
      name_prefix='VizierGPBandit', name='ard', also_log=True
  )
  def _find_best_model_params(self) -> types.ParameterDict:
    """Perform ARD on the current model to find best model parameters."""
    self._model, loss_fn = (
        tuned_gp_models.VizierGaussianProcess.model_and_loss_fn(
            self._features, self._labels
        )
    )
    # Run ARD.
    setup = lambda rng: self._model.init(rng, self._features)['params']
    constraints = sp.get_constraints(self._model)
    ard_loss_fn = self._get_loss_fn(loss_fn)

    logging.info('Optimizing the loss function...')
    self._rng, ard_rng = jax.random.split(self._rng, 2)
    best_model_params, _ = self._ard_optimizer(
        setup, ard_loss_fn, ard_rng, constraints=constraints
    )
    return best_model_params

  def _get_loss_fn(self, loss_fn):
    if isinstance(self._ard_optimizer, optimizers.OptaxTrainWithRandomRestarts):

      def ard_loss_fn(x):
        loss_val, metrics = loss_fn(x)
        # For SGD, normalize the loss so we can use the same learning rate
        # regardless of the number of examples (see
        # `OptaxTrainWithRandomRestarts` docstring).
        return loss_val / self._features.shape[0], metrics

    else:
      ard_loss_fn = loss_fn
    return jax.jit(ard_loss_fn)

  @profiler.record_runtime(name_prefix='VizierGPBandit', name='compute_state')
  def _compute_state(self):
    """Compute the designer's state.

    1. Convert trials to features and labels.
    2. Perform ARD to find best model parameters.
    3. Pre-compute the Cholesky decomposition.

    If no new trials were added since last call, no update will occur.
    """
    if len(self._trials) == self._incorporated_trials_count:
      # If there's no change in the number of completed trials, don't update
      # state. The assumption is that trials can't be removed.
      return
    self._incorporated_trials_count = len(self._trials)
    # Convert trials to Numpy arrays. Labels are warped.
    self._features, self._labels = self._convert_trials_to_arrays(self._trials)
    # Update the model.
    self._model, loss_fn = (
        tuned_gp_models.VizierGaussianProcess.model_and_loss_fn(
            self._features, self._labels
        )
    )
    best_model_params = self._find_best_model_params()
    # Logging for debugging purposes.
    logging.info('Best model parameters: %s', best_model_params)
    ard_loss_fn = self._get_loss_fn(loss_fn)
    if self._use_vmap:
      ard_loss_fn = jax.vmap(ard_loss_fn)
    ard_all_losses = ard_loss_fn(best_model_params)[0]
    logging.info('All losses: %s', ard_all_losses)
    ard_best_loss = ard_all_losses.flatten()[0].item()
    logging.info('ARD best loss: %s', ard_best_loss)

    @profiler.record_runtime(
        name_prefix='VizierGPBandit', name='precompute_cholesky'
    )
    @jax.jit
    def precompute_cholesky(params):
      return self._model.apply(
          {'params': params},
          self._features,
          self._labels,
          method=self._model.precompute_predictive,
          mutable='predictive',
      )
    if self._use_vmap:
      precompute_cholesky = jax.vmap(precompute_cholesky)
    # `pp_state` contains intermediates that are expensive to compute, depend
    # only on observed (not predictive) index points, and are needed to compute
    # the posterior predictive GP (i.e. the Cholesky factor of the kernel matrix
    # over observed index points). `pp_state` is passed to
    # `model.posterior_predictive` to avoid re-computing the intermediates
    # unnecessarily.
    _, pp_state = precompute_cholesky(best_model_params)
    self._state = {'params': best_model_params, **pp_state}

    self._acquisition_builder.use_trust_region = self._use_trust_region

    with profiler.record_runtime_context(
        name='VizierGPBandit.acquisition_build', also_log=True
    ):
      self._acquisition_builder.build(
          problem=self._problem,
          model=self._model,
          state=self._state,
          features=self._features,
          labels=self._labels,
          converter=self._converter,
          use_vmap=self._use_vmap,
      )

  @profiler.record_runtime(
      name_prefix='VizierGPBandit', name='optimize_acquisition'
  )
  def _optimize_acquisition(self, count: int) -> list[vz.Trial]:
    """Optimize the acquisition function."""
    return self._acquisition_optimizer.optimize(
        score_fn=self._acquisition_builder.acquisition_on_array,
        converter=self._converter,
        count=count,
        prior_trials=self._trials,
    )

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
    self._compute_state()

    logging.info('Optimizing acquisition...')
    best_candidates = self._optimize_acquisition(count)

    # Convert best_candidates (in scaled space) into suggestions (in unscaled
    # space); also append debug information like model predictions.
    logging.info('Converting the optimization result into suggestions...')
    optimal_features = self._converter.to_features(best_candidates)  # [N, D]
    # Make predictions (in the warped space). [N]
    with profiler.record_runtime_context(
        'VizierGPBandit.predict_on_suggestions'
    ):
      predictions = self._acquisition_builder.predict_on_array(optimal_features)
    predict_mean = predictions['mean']  # [N,]
    predict_stddev = predictions['stddev']  # [N,]
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
              'mean': predict_mean[i],
              'stddev': predict_stddev[i],
              'acquisition': acquisition,
              'state': self._state,
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

    self._compute_state()
    xs = self._converter.to_features(trials)
    samples = self._acquisition_builder.sample_on_array(
        xs,
        num_samples=num_samples,
        key=rng,
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
