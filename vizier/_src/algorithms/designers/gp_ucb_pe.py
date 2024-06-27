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

"""Gaussian Process Bandit with Pure Exploration using a Flax model and a TFP Gaussian Process."""

# pylint: disable=logging-fstring-interpolation, g-long-lambda

import copy
import datetime
import random
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from absl import logging
import attr
import chex
import equinox as eqx
import jax
from jax import numpy as jnp
import jaxtyping as jt
import numpy as np
from tensorflow_probability.substrates import jax as tfp  # pylint: disable=g-importing-member
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
from vizier.jax import optimizers
from vizier.pyvizier import converters
from vizier.pyvizier.converters import padding
from vizier.utils import profiler


tfd = tfp.distributions


class UCBPEConfig(eqx.Module):
  """UCB-PE config parameters."""

  ucb_coefficient: jt.Float[jt.Array, ''] = eqx.field(
      default=1.8, converter=jnp.asarray
  )
  # A separate ucb coefficient defining the good region to explore.
  explore_region_ucb_coefficient: jt.Float[jt.Array, ''] = eqx.field(
      default=0.5, converter=jnp.asarray
  )
  # The constraint violation penalty is a linear function of the constraint
  # violation, whose slope is determined by this coefficient.
  cb_violation_penalty_coefficient: jt.Float[jt.Array, ''] = eqx.field(
      default=10.0, converter=jnp.asarray
  )
  # Probability of selecting the UCB acquisition function when there are no new
  # completed trials. No-op if `optimize_set_acquisition_for_exploration` below
  # is True.
  ucb_overwrite_probability: jt.Float[jt.Array, ''] = eqx.field(
      default=0.25, converter=jnp.asarray
  )
  # Probability of selecting the PE acquisition function when there are new
  # completed trials.
  pe_overwrite_probability: jt.Float[jt.Array, ''] = eqx.field(
      default=0.1, converter=jnp.asarray
  )
  # The same as `pe_overwrite_probability` but only applies when the noise is
  # estimated to be high.
  pe_overwrite_probability_in_high_noise: jt.Float[jt.Array, ''] = eqx.field(
      default=0.7, converter=jnp.asarray
  )
  # When the ratio between the estimated signal variance and the noise variance
  # is below this threshold, the designer considers the noise to be high and may
  # explore more aggressively. Default to 0.0 to disable this feature.
  signal_to_noise_threshold: jt.Float[jt.Array, ''] = eqx.field(
      default=0.0, converter=jnp.asarray
  )
  # Whether to optimize the set acquisition function for exploration.
  optimize_set_acquisition_for_exploration: bool = eqx.field(
      default=False, static=True
  )

  def __repr__(self):
    return eqx.tree_pformat(self, short_arrays=False)


# A dummy loss for ARD when there are no completed trials.
_DUMMY_LOSS = -1.0


def _has_new_completed_trials(
    completed_trials: Sequence[vz.Trial], active_trials: Sequence[vz.Trial]
) -> bool:
  """Returns True iff there are newer completed trials than active trials.

  Args:
    completed_trials: Completed trials.
    active_trials: Active trials.

  Returns:
    True if `completed_trials` is non-empty and:
      - `active_trials` is empty, or
      - The latest `completion_time` among `completed_trials` is
        later than the latest `creation_time` among `active_trials`.
    False: otherwise.
  """

  if not completed_trials:
    return False
  if not active_trials:
    return True

  completed_completion_times = [t.completion_time for t in completed_trials]
  active_creation_times = [t.creation_time for t in active_trials]

  if not all(completed_completion_times):
    raise ValueError('All completed trials must have completion times.')
  if not all(active_creation_times):
    raise ValueError('All active trials must have creation times.')

  return max(completed_completion_times) > max(active_creation_times)  # pytype:disable=unsupported-operands


def _compute_ucb_threshold(
    gprm: tfd.Distribution,
    is_missing: jt.Bool[jt.Array, ''],
    ucb_coefficient: jt.Float[jt.Array, ''],
) -> jax.Array:
  """Computes a threshold on UCB values.

  A promising evaluation point has UCB value no less than the threshold
  computed here. The threshold is the predicted mean of the feature array
  with the maximum UCB value among the points `gprm.index_points`.

  Args:
    gprm: A GP regression model for a set of predictive index points.
    is_missing: A 1-d boolean array indicating whether the corresponding
      predictive index points are missing.
    ucb_coefficient: The UCB coefficient.

  Returns:
    The predicted mean of the feature array with the maximum UCB among `xs`.
  """
  pred_mean = gprm.mean()
  ucb_values = jnp.where(
      is_missing, -jnp.inf, pred_mean + ucb_coefficient * gprm.stddev()
  )
  return pred_mean[jnp.argmax(ucb_values)]


# TODO: Use acquisitions.TrustRegion instead.
def _apply_trust_region(
    tr: acquisitions.TrustRegion, xs: types.ModelInput, acq_values: jax.Array
) -> jax.Array:
  """Applies the trust region to acquisition function values.

  Args:
   tr: Trust region.
   xs: Predictive index points.
   acq_values: Acquisition function values at predictive index points.

  Returns:
    Acquisition function values with trust region applied.
  """
  distance = tr.min_linf_distance(xs)
  # Due to output normalization, acquisition values can't be as low as -1e12.
  # We use a bad value that decreases in the distance to trust region so that
  # acquisition optimizer can follow the gradient and escape untrustred regions.
  return jnp.where(
      (distance < tr.trust_radius) | (tr.trust_radius > 0.5),
      acq_values,
      -1e12 - distance,
  )


def _apply_trust_region_to_set(
    tr: acquisitions.TrustRegion, xs: types.ModelInput, acq_values: jax.Array
) -> jax.Array:
  """Applies the trust region to a batch of set acquisition function values.

  Args:
   tr: Trust region.
   xs: A batch of predictive index point sets of a fixed size.
   acq_values: A batch of acquisition function values at predictive index point
     sets, shaped as [batch_size].

  Returns:
    Acquisition function values with trust region applied, shaped as
    [batch_size].
  """
  distance = tr.min_linf_distance(xs)  # [batch_size, index_point_set_size]
  # Due to output normalization, acquisition values can't be as low as -1e12.
  # We penalize the acquisition values by an amount that decreases in the
  # total distances to the trust region so that acquisition optimizer can follow
  # the gradient and escape untrustred regions.
  return acq_values + jnp.sum(
      ((distance > tr.trust_radius) & (tr.trust_radius <= 0.5))
      * (-1e12 - distance),
      axis=1,
  )


def _get_features_shape(
    features: types.ModelInput,
) -> types.ContinuousAndCategorical:
  """Gets the shapes of continuous/categorical features for logging."""
  return types.ContinuousAndCategorical(
      features.continuous.shape,
      features.categorical.shape,
  )


class UCBScoreFunction(eqx.Module):
  """Computes the UCB acquisition value.

  The UCB acquisition value is the sum of the predicted mean based on completed
  trials and the predicted standard deviation based on all trials, completed and
  pending (scaled by the UCB coefficient). This class follows the
  `acquisitions.ScoreFunction` protocol.

  Attributes:
    predictive: Predictive model with cached Cholesky conditioned on completed
      trials.
    predictive_all_features: Predictive model with cached Cholesky conditioned
      on completed and pending trials.
    ucb_coefficient: The UCB coefficient.
    trust_region: Trust region.
  """

  predictive: sp.UniformEnsemblePredictive
  predictive_all_features: sp.UniformEnsemblePredictive
  ucb_coefficient: jt.Float[jt.Array, '']
  trust_region: Optional[acquisitions.TrustRegion]

  def score(
      self, xs: types.ModelInput, seed: Optional[jax.Array] = None
  ) -> jax.Array:
    return self.score_with_aux(xs, seed=seed)[0]

  def aux(
      self, xs: types.ModelInput, seed: Optional[jax.Array] = None
  ) -> chex.ArrayTree:
    return self.score_with_aux(xs, seed=seed)[1]

  def score_with_aux(
      self, xs: types.ModelInput, seed: Optional[jax.Array] = None
  ) -> tuple[jax.Array, chex.ArrayTree]:
    del seed
    gprm = self.predictive.predict(xs)
    gprm_all_features = self.predictive_all_features.predict(xs)
    mean = gprm.mean()
    stddev_from_all = gprm_all_features.stddev()
    acq_values = mean + self.ucb_coefficient * stddev_from_all
    if self.trust_region is not None:
      acq_values = _apply_trust_region(self.trust_region, xs, acq_values)
    return acq_values, {
        'mean': mean,
        'stddev': gprm.stddev(),
        'stddev_from_all': stddev_from_all,
    }


class PEScoreFunction(eqx.Module):
  """Computes the Pure-Exploration acquisition value.

  The PE acquisition value is the predicted standard deviation (eq. (9)
  in https://arxiv.org/pdf/1304.5350) based on all completed and active trials,
  plus a penalty term that grows linearly in the amount of violation of the
  constraint `UCB(xs) >= threshold`. This class follows the
  `acquisitions.ScoreFunction` protocol.

  Attributes:
    predictive: Predictive model with cached Cholesky conditioned on completed
      trials.
    predictive_all_features: Predictive model with cached Cholesky conditioned
      on completed and pending trials.
    ucb_coefficient: The UCB coefficient used to compute the threshold.
    explore_ucb_coefficient: The UCB coefficient used for computing the UCB
      values on `xs`.
    penalty_coefficient: Multiplier on the constraint violation penalty.
    trust_region:

  Returns:
    The Pure-Exploration acquisition value.
  """

  predictive: sp.UniformEnsemblePredictive
  predictive_all_features: sp.UniformEnsemblePredictive
  ucb_coefficient: jt.Float[jt.Array, '']
  explore_ucb_coefficient: jt.Float[jt.Array, '']
  penalty_coefficient: jt.Float[jt.Array, '']
  trust_region: Optional[acquisitions.TrustRegion]

  def score(
      self, xs: types.ModelInput, seed: Optional[jax.Array] = None
  ) -> jax.Array:
    return self.score_with_aux(xs, seed=seed)[0]

  def aux(
      self, xs: types.ModelInput, seed: Optional[jax.Array] = None
  ) -> chex.ArrayTree:
    return self.score_with_aux(xs, seed=seed)[1]

  def score_with_aux(
      self, xs: types.ModelInput, seed: Optional[jax.Array] = None
  ) -> tuple[jax.Array, chex.ArrayTree]:
    del seed
    features = self.predictive_all_features.predictives.observed_data.features
    is_missing = (
        features.continuous.is_missing[0] | features.categorical.is_missing[0]
    )
    gprm_threshold = self.predictive.predict(features)
    threshold = _compute_ucb_threshold(
        gprm_threshold, is_missing, self.ucb_coefficient
    )
    gprm = self.predictive.predict(xs)
    mean = gprm.mean()
    stddev = gprm.stddev()
    explore_ucb = mean + stddev * self.explore_ucb_coefficient

    gprm_all = self.predictive_all_features.predict(xs)
    stddev_from_all = gprm_all.stddev()
    acq_values = stddev_from_all + self.penalty_coefficient * jnp.minimum(
        explore_ucb - threshold,
        0.0,
    )
    if self.trust_region is not None:
      acq_values = _apply_trust_region(self.trust_region, xs, acq_values)
    return acq_values, {
        'mean': mean,
        'stddev': stddev,
        'stddev_from_all': stddev_from_all,
    }


def _logdet(matrix: jax.Array):
  """Computes the log-determinant of a symmetric and positive-definite matrix.

  Args:
    matrix: A square matrix.

  Returns:
    The log-determinant of `matrix`. If `matrix` is not symmetric or not
    positive-definite, the result is invalid and may be -inf.
  """
  cholesky_matrix = jnp.linalg.cholesky(matrix)
  output = 2.0 * jnp.sum(jnp.log(jnp.linalg.diagonal(cholesky_matrix)), axis=-1)
  return jnp.where(jnp.isnan(output), -jnp.inf, output)


class SetPEScoreFunction(eqx.Module):
  """Computes the Pure-Exploration acquisition value over sets.

  The PE acquisition value over a set of points is the log-determinant of the
  predicted covariance matrix evaluated at the points (eq. (8) in
  https://arxiv.org/pdf/1304.5350) based on all completed and active trials,
  plus a penalty term that grows linearly in the amount of violation of the
  constraint `UCB(xs) >= threshold`. This class follows the
  `acquisitions.ScoreFunction` protocol.

  Attributes:
    predictive: Predictive model with cached Cholesky conditioned on completed
      trials.
    predictive_all_features: Predictive model with cached Cholesky conditioned
      on completed and pending trials.
    ucb_coefficient: The UCB coefficient used to compute the threshold.
    explore_ucb_coefficient: The UCB coefficient used for computing the UCB
      values on `xs`.
    penalty_coefficient: Multiplier on the constraint violation penalty.
    trust_region:

  Returns:
    The Pure-Exploration acquisition value.
  """

  predictive: sp.UniformEnsemblePredictive
  predictive_all_features: sp.UniformEnsemblePredictive
  ucb_coefficient: jt.Float[jt.Array, '']
  explore_ucb_coefficient: jt.Float[jt.Array, '']
  penalty_coefficient: jt.Float[jt.Array, '']
  trust_region: Optional[acquisitions.TrustRegion]

  def score(
      self, xs: types.ModelInput, seed: Optional[jax.Array] = None
  ) -> jax.Array:
    return self.score_with_aux(xs, seed=seed)[0]

  def aux(
      self, xs: types.ModelInput, seed: Optional[jax.Array] = None
  ) -> chex.ArrayTree:
    return self.score_with_aux(xs, seed=seed)[1]

  def score_with_aux(
      self, xs: types.ModelInput, seed: Optional[jax.Array] = None
  ) -> tuple[jax.Array, chex.ArrayTree]:
    del seed
    features = self.predictive_all_features.predictives.observed_data.features
    is_missing = (
        features.continuous.is_missing[0] | features.categorical.is_missing[0]
    )
    gprm_threshold = self.predictive.predict(features)
    threshold = _compute_ucb_threshold(
        gprm_threshold, is_missing, self.ucb_coefficient
    )
    gprm = self.predictive.predict(xs)
    mean = gprm.mean()
    stddev = gprm.stddev()
    explore_ucb = mean + stddev * self.explore_ucb_coefficient

    gprm_all = self.predictive_all_features.predict(xs)
    cov = gprm_all.covariance()
    acq_values = _logdet(cov) + self.penalty_coefficient * jnp.sum(
        jnp.minimum(
            explore_ucb - threshold,
            0.0,
        ),
        axis=1,
    )
    if self.trust_region is not None:
      acq_values = _apply_trust_region_to_set(self.trust_region, xs, acq_values)
    return acq_values, {
        'mean': mean,
        'stddev': stddev,
        'stddev_from_all': jnp.sqrt(jnp.diagonal(cov, axis1=1, axis2=2)),
    }


def default_ard_optimizer() -> optimizers.Optimizer[types.ParameterDict]:
  return optimizers.JaxoptScipyLbfgsB(
      options=optimizers.LbfgsBOptions(
          num_line_search_steps=20,
          tol=1e-5,
          maxiter=500,
      ),
      max_duration=datetime.timedelta(minutes=40),
  )


# TODO: Remove excess use of copy.deepcopy()
@attr.define(auto_attribs=False)
class VizierGPUCBPEBandit(vza.Designer):
  """GP_UCB_PE with a flax model.

  Attributes:
    problem: Must be a flat study with a single metric.
    acquisition_optimizer:
    metadata_ns: Metadata namespace that this designer writes to.
    use_trust_region: Uses trust region.
    ard_optimizer: An optimizer object, which should return a batch of
      hyperparameters to be ensembled.
    num_seed_trials: If greater than zero, first trial is the center of the
      search space. Afterwards, uses quasirandom until this number of trials are
      observed.
    rng: If not set, uses random numbers.
    clear_jax_cache: If True, every `suggest` call clears the Jax cache.
  """

  _problem: vz.ProblemStatement = attr.field(kw_only=False)
  _acquisition_optimizer_factory: Union[
      Callable[[Any], vza.GradientFreeOptimizer], vb.VectorizedOptimizerFactory
  ] = attr.field(
      kw_only=True,
      factory=lambda: VizierGPUCBPEBandit.default_acquisition_optimizer_factory,
  )
  _metadata_ns: str = attr.field(
      default='google_gp_ucb_pe_bandit', kw_only=True
  )
  _ensemble_size: Optional[int] = attr.field(default=1, kw_only=True)
  _all_completed_trials: list[vz.Trial] = attr.field(factory=list, init=False)
  _all_active_trials: Sequence[vz.Trial] = attr.field(factory=list, init=False)
  _ard_optimizer: optimizers.Optimizer[types.ParameterDict] = attr.field(
      factory=default_ard_optimizer,
      kw_only=True,
  )
  _ard_random_restarts: int = attr.field(default=4, kw_only=True)
  _use_trust_region: bool = attr.field(default=True, kw_only=True)
  _num_seed_trials: int = attr.field(default=1, kw_only=True)
  _config: UCBPEConfig = attr.field(
      factory=UCBPEConfig,
      kw_only=True,
  )
  _rng: jax.Array = attr.field(
      factory=lambda: jax.random.PRNGKey(random.getrandbits(32)), kw_only=True
  )
  _clear_jax_cache: bool = attr.field(default=False, kw_only=True)
  # Whether to pad all inputs, and what type of schedule to use. This is to
  # ensure fewer JIT compilation passes. (Default implies no padding.)
  # TODO: Check padding does not affect designer behavior.
  _padding_schedule: padding.PaddingSchedule = attr.field(
      factory=padding.PaddingSchedule, kw_only=True
  )

  default_acquisition_optimizer_factory = vb.VectorizedOptimizerFactory(
      strategy_factory=es.VectorizedEagleStrategyFactory(
          eagle_config=es.EagleStrategyConfig(
              visibility=3.6782451729470043,
              gravity=3.028167342024462,
              negative_gravity=0.03036267153343141,
              perturbation=0.23337470891647027,
              categorical_perturbation_factor=9.587350648631066,
              pure_categorical_perturbation_factor=28.636337967676518,
              prob_same_category_without_perturbation=0.9744882009359648,
              perturbation_lower_bound=7.376256294543107e-4,
              penalize_factor=0.7817632796830948,
              pool_size_exponent=2.0494446726436744,
              mutate_normalization_type=es.MutateNormalizationType.RANDOM,
              normalization_scale=1.9893618760239418,
              prior_trials_pool_pct=0.423499384081575,
          )
      ),
      max_evaluations=75000,
      suggestion_batch_size=25,
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
    qrs_seed, self._rng = jax.random.split(self._rng)
    self._quasi_random_sampler = quasi_random.QuasiRandomDesigner(
        self._problem.search_space,
        seed=int(jax.random.randint(qrs_seed, [], 0, 2**16)),
    )

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    self._all_completed_trials.extend(copy.deepcopy(completed.trials))
    self._all_active_trials = copy.deepcopy(all_active.trials)

  @property
  def _metric_info(self) -> vz.MetricInformation:
    return self._problem.metric_information.item()

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
    if (not self._all_completed_trials) and (not self._all_active_trials):
      features = self._converter.to_features([])  # to extract shape.
      # NOTE: The code below assumes that a scaled value of 0.5 corresponds
      # to the center of the feasible range. This is true, but only by accident;
      # ideally, we should get the center from the converters.
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
    if (remaining_counts := count - len(seed_suggestions)) > 0:
      quasi_suggestions = self._quasi_random_sampler.suggest(remaining_counts)
      seed_suggestions.extend(quasi_suggestions)
    return seed_suggestions

  @profiler.record_runtime(
      name_prefix='VizierGPUCBPEBandit',
      name='build_gp_model_and_optimize_parameters',
  )
  def _build_gp_model_and_optimize_parameters(
      self, data: types.ModelData, rng: jax.Array
  ) -> sp.StochasticProcessWithCoroutine:
    """Builds a GP model and optimizes parameters.

    Args:
      data: Observed features and labels.
      rng: A key for random number generation.

    Returns:
      A tuple of GP model and its parameters optimized over `data.features` and
      `data.labels`. If `data.features` is empty, the returned parameters are
      initial values picked by the GP model.
    """
    # TODO: Update to `VizierLinearGaussianProcess`.
    coroutine = tuned_gp_models.VizierGaussianProcess.build_model(
        data.features
    ).coroutine
    model = sp.CoroutineWithData(coroutine, data)

    if (data.features.continuous.padded_array.shape[0] == 0) and (
        data.features.categorical.padded_array.shape[0] == 0
    ):
      # This happens when `suggest` is called after the seed trials are
      # generated without any completed trials. In this case, the designer
      # uses the PE acquisition, but still needs a GP to do that. By using a
      # dummy loss here, the ARD optimizer is expected to return the initial
      # values it uses for the parameters.
      ard_loss_with_aux = lambda _: (_DUMMY_LOSS, dict())
    else:
      ard_loss_with_aux = model.loss_with_aux

    logging.info(
        'Optimizing the loss function on features with shape '
        f'{_get_features_shape(data.features)} and labels with shape '
        f'{data.labels.shape}...'
    )
    constraints = sp.get_constraints(model)
    rng, init_rng = jax.random.split(rng, 2)
    random_init_params = eqx.filter_jit(eqx.filter_vmap(model.setup))(
        jax.random.split(init_rng, self._ard_random_restarts)
    )
    fixed_init_params = {
        'signal_variance': jnp.array([0.039]),
        'observation_noise_variance': jnp.array([0.0039]),
        'continuous_length_scale_squared': jnp.array(
            [[1.0] * data.features.continuous.padded_array.shape[-1]]
        ),
        'categorical_length_scale_squared': jnp.array(
            [[1.0] * data.features.categorical.padded_array.shape[-1]]
        ),
    }
    best_n = self._ensemble_size or 1
    optimal_params, metrics = self._ard_optimizer(
        init_params=jax.tree.map(
            lambda x, y: jnp.concatenate([x, y]),
            fixed_init_params,
            random_init_params,
        ),
        loss_fn=ard_loss_with_aux,
        rng=rng,
        constraints=constraints,
        best_n=best_n,
    )
    # The `"loss"` field of the `metrics` output of ARD optimizers contains an
    # array of losses of shape `[num_steps, num_random_restarts]` (or
    # `[1, num_random_restarts]` if only the final loss is recorded).
    if jnp.any(metrics['loss'][-1, :].argsort()[:best_n] == 0):
      logging.info(
          'Parameters found by fixed initialization are among the best'
          f' {best_n} parameters.'
      )
    else:
      logging.info(
          f'The best {best_n} parameters were all found by random'
          ' initialization.'
      )

    logging.info('Optimal parameters: %s', optimal_params)
    return sp.StochasticProcessWithCoroutine(coroutine, optimal_params)

  def get_score_fn_on_trials(
      self, score_fn: Callable[[types.ModelInput], jax.Array]
  ) -> Callable[[Sequence[vz.Trial]], Mapping[str, jax.Array]]:
    """Builds a callable that evaluates the score function on trials.

    Args:
      score_fn: Score function that takes arrays as input.

    Returns:
      Score function that takes trials as input.
    """

    def acquisition(trials: Sequence[vz.Trial]) -> Mapping[str, jax.Array]:
      jax_acquisitions = eqx.filter_jit(score_fn)(
          self._converter.to_features(trials)
      )
      return {'acquisition': jax_acquisitions}

    return acquisition

  @profiler.record_runtime
  def _trials_to_data(self, trials: Sequence[vz.Trial]) -> types.ModelData:
    """Convert trials to scaled features and warped labels."""
    # TrialToArrayConverter returns floating arrays.
    data = self._converter.to_xy(trials)
    logging.info(
        'Transforming the labels of shape %s. Features has shape: %s',
        data.labels.shape,
        _get_features_shape(data.features),
    )
    warped_labels = output_warpers.create_default_warper().warp(
        np.array(data.labels.unpad())
    )
    labels = types.PaddedArray.from_array(
        warped_labels,
        data.labels.padded_array.shape,
        fill_value=data.labels.fill_value,
    )
    logging.info('Transformed the labels. Now has shape: %s', labels.shape)
    return types.ModelData(features=data.features, labels=labels)

  @profiler.record_runtime(
      name_prefix='VizierGPUCBPEBandit', name='get_predictive_all_features'
  )
  def _get_predictive_all_features(
      self,
      pending_features: types.ModelInput,
      data: types.ModelData,
      model: sp.StochasticProcessWithCoroutine,
      noise_is_high: bool,
  ) -> sp.UniformEnsemblePredictive:
    """Builds the predictive model conditioned on observed and pending features.

    Args:
      pending_features: Pending features.
      data: Features/labels for completed trials.
      model: The GP model.
      noise_is_high: Whether the noise is estimated to be high.

    Returns:
      Predictive model with cached Cholesky conditioned on observed and pending
      features.
    """
    # TODO: Use `PaddedArray.concatenate` when implemented.
    all_features_continuous = jnp.concatenate(
        [
            data.features.continuous.unpad(),
            pending_features.continuous.unpad(),
        ],
        axis=0,
    )
    all_features_categorical = jnp.concatenate(
        [
            data.features.categorical.unpad(),
            pending_features.categorical.unpad(),
        ],
        axis=0,
    )
    all_features = types.ModelInput(
        continuous=self._padding_schedule.pad_features(all_features_continuous),
        categorical=self._padding_schedule.pad_features(
            all_features_categorical
        ),
    )
    # Pending features are only used to predict standard deviation, so their
    # labels do not matter, and we simply set them to 0.
    dummy_labels = jnp.zeros(
        shape=(pending_features.continuous.unpad().shape[0], 1),
        dtype=data.labels.padded_array.dtype,
    )
    all_labels = jnp.concatenate([data.labels.unpad(), dummy_labels], axis=0)
    all_labels = self._padding_schedule.pad_labels(all_labels)
    all_data = types.ModelData(features=all_features, labels=all_labels)
    if noise_is_high:
      pe_params = dict(copy.deepcopy(model.params))
      pe_params['observation_noise_variance'] = jnp.array([1e-10])
      pe_model = sp.StochasticProcessWithCoroutine(model.coroutine, pe_params)
    else:
      pe_model = model
    return sp.UniformEnsemblePredictive(
        predictives=eqx.filter_jit(pe_model.precompute_predictive)(all_data)
    )

  def _suggest_one(
      self,
      active_trials: Sequence[vz.Trial],
      data: types.ModelData,
      model: sp.StochasticProcessWithCoroutine,
      predictive: sp.UniformEnsemblePredictive,
      tr: acquisitions.TrustRegion,
      acquisition_problem: vz.ProblemStatement,
  ) -> vz.TrialSuggestion:
    """Generates one suggestion."""
    start_time = datetime.datetime.now()
    self._rng, rng = jax.random.split(self._rng, 2)
    snr = model.params['signal_variance'] / jnp.maximum(
        model.params['observation_noise_variance'], 1e-12
    )
    noise_is_high = (snr < self._config.signal_to_noise_threshold).all()
    pe_overwrite_probability = (
        self._config.pe_overwrite_probability_in_high_noise
        if noise_is_high
        else self._config.pe_overwrite_probability
    )
    if _has_new_completed_trials(
        completed_trials=self._all_completed_trials,
        active_trials=active_trials,
    ):
      # When there are trials completed after all active trials were created,
      # we optimize the UCB acquisition function except with a small
      # probability the PE acquisition function to ensure exploration.
      use_ucb = not jax.random.bernoulli(key=rng, p=pe_overwrite_probability)
    else:
      has_completed_trials = len(self._all_completed_trials) > 0  # pylint:disable=g-explicit-length-test
      # When there are no trials completed after all active trials were
      # created, we optimize the PE acquisition function except with a small
      # probability the UCB acquisition function, in case the UCB acquisition
      # function is not well optimized.
      use_ucb = has_completed_trials and jax.random.bernoulli(
          key=rng, p=self._config.ucb_overwrite_probability
      )

    # TODO: Feed the eagle strategy with completed trials.
    # TODO: Change budget based on requested suggestion count.
    acquisition_optimizer = self._acquisition_optimizer_factory(self._converter)

    pending_features = self._converter.to_features(active_trials)
    predictive_all_features = self._get_predictive_all_features(
        pending_features, data, model, noise_is_high
    )

    # When `use_ucb` is true, the acquisition function computes the UCB
    # values. Otherwise, it computes the Pure-Exploration acquisition values.
    if use_ucb:
      scoring_fn = UCBScoreFunction(
          predictive,
          predictive_all_features,
          ucb_coefficient=self._config.ucb_coefficient,
          trust_region=tr if self._use_trust_region else None,
      )
    else:
      scoring_fn = PEScoreFunction(
          predictive,
          predictive_all_features,
          penalty_coefficient=self._config.cb_violation_penalty_coefficient,
          ucb_coefficient=self._config.ucb_coefficient,
          explore_ucb_coefficient=self._config.explore_region_ucb_coefficient,
          trust_region=tr if self._use_trust_region else None,
      )

    if isinstance(acquisition_optimizer, vb.VectorizedOptimizer):
      acq_rng, self._rng = jax.random.split(self._rng)
      with profiler.timeit('acquisition_optimizer', also_log=True):
        best_candidates = eqx.filter_jit(acquisition_optimizer)(
            scoring_fn.score,
            prior_features=vb.trials_to_sorted_array(
                self._all_completed_trials, self._converter
            ),
            count=1,
            seed=acq_rng,
            score_with_aux_fn=scoring_fn.score_with_aux,
        )
        jax.block_until_ready(best_candidates)
      with profiler.timeit('best_candidates_to_trials', also_log=True):
        best_candidate = vb.best_candidates_to_trials(
            best_candidates, self._converter
        )[0]
    elif isinstance(acquisition_optimizer, vza.GradientFreeOptimizer):
      # Seed the optimizer with previous trials.
      acquisition = self.get_score_fn_on_trials(scoring_fn.score)
      best_candidate = acquisition_optimizer.optimize(
          acquisition,
          acquisition_problem,
          count=1,
          seed_candidates=copy.deepcopy(self._all_completed_trials),
      )[0]
    else:
      raise ValueError(
          f'Unrecognized acquisition_optimizer: {type(acquisition_optimizer)}'
      )

    # Make predictions (in the warped space).
    logging.info('Converting the optimization result into suggestion...')
    optimal_features = self._converter.to_features([best_candidate])  # [1, D]
    aux = eqx.filter_jit(scoring_fn.aux)(optimal_features)
    predict_mean = aux['mean']  # [1,]
    predict_stddev = aux['stddev']  # [1,]
    predict_stddev_from_all = aux['stddev_from_all']  # [1,]
    acquisition = best_candidate.final_measurement_or_die.metrics.get_value(
        'acquisition', float('nan')
    )
    logging.info(
        'Created predictions for the best candidates which were converted to'
        f' an array of shape: {_get_features_shape(optimal_features)}. mean'
        f' has shape {predict_mean.shape}. stddev has shape'
        f' {predict_stddev.shape}.stddev_from_all has shape'
        f' {predict_stddev_from_all.shape}. acquisition value of'
        f' best_candidate: {acquisition}, use_ucb: {use_ucb}'
    )

    # Create a suggestion, injecting the predictions as metadata for
    # debugging needs.
    metadata = best_candidate.metadata.ns(self._metadata_ns)
    metadata.ns('prediction_in_warped_y_space').update({
        'mean': f'{predict_mean[0]}',
        'stddev': f'{predict_stddev[0]}',
        'stddev_from_all': f'{predict_stddev_from_all[0]}',
        'acquisition': f'{acquisition}',
        'use_ucb': f'{use_ucb}',
        'trust_radius': f'{tr.trust_radius}',
        'params': f'{model.params}',
    })
    metadata.ns('timing').update(
        {'time': f'{datetime.datetime.now() - start_time}'}
    )
    return vz.TrialSuggestion(
        best_candidate.parameters, metadata=best_candidate.metadata
    )

  def _suggest_batch_with_exploration(
      self,
      count: int,
      active_trials: Sequence[vz.Trial],
      data: types.ModelData,
      model: sp.StochasticProcessWithCoroutine,
      predictive: sp.UniformEnsemblePredictive,
      tr: acquisitions.TrustRegion,
  ):
    """Generates a batch of suggestions with exploration."""
    start_time = datetime.datetime.now()
    snr = model.params['signal_variance'] / jnp.maximum(
        model.params['observation_noise_variance'], 1e-12
    )
    pending_features = self._converter.to_features(active_trials)
    predictive_all_features = self._get_predictive_all_features(
        pending_features,
        data,
        model,
        noise_is_high=(snr < self._config.signal_to_noise_threshold),
    )
    scoring_fn = SetPEScoreFunction(
        predictive,
        predictive_all_features,
        penalty_coefficient=self._config.cb_violation_penalty_coefficient,
        ucb_coefficient=self._config.ucb_coefficient,
        explore_ucb_coefficient=self._config.explore_region_ucb_coefficient,
        trust_region=tr if self._use_trust_region else None,
    )

    acquisition_optimizer = self._acquisition_optimizer_factory(self._converter)

    acq_rng, self._rng = jax.random.split(self._rng)
    with profiler.timeit('acquisition_optimizer', also_log=True):
      best_candidates = eqx.filter_jit(acquisition_optimizer)(
          scoring_fn.score,
          prior_features=vb.trials_to_sorted_array(
              self._all_completed_trials, self._converter
          ),
          count=1,
          seed=acq_rng,
          score_with_aux_fn=scoring_fn.score_with_aux,
          n_parallel=count,
      )
      jax.block_until_ready(best_candidates)
    with profiler.timeit('best_candidates_to_trials', also_log=True):
      trials = vb.best_candidates_to_trials(best_candidates, self._converter)[
          :count
      ]

    optimal_features = self._converter.to_features(trials)  # [count, D]
    aux = eqx.filter_jit(scoring_fn.aux)(
        jax.tree_util.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), optimal_features
        )
    )
    predict_mean = aux['mean']  # [1, count]
    predict_stddev = aux['stddev']  # [1, count]
    predict_stddev_from_all = aux['stddev_from_all']  # [1, count]
    acquisition = trials[0].final_measurement_or_die.metrics.get_value(
        'acquisition', float('nan')
    )
    logging.info(
        'Created predictions for the best candidates which were converted to'
        f' an array of shape: {_get_features_shape(optimal_features)}. mean'
        f' has shape {predict_mean.shape}. stddev has shape'
        f' {predict_stddev.shape}.stddev_from_all has shape'
        f' {predict_stddev_from_all.shape}. acquisition value of'
        f' best_candidate: {acquisition}, use_ucb: False'
    )

    logging.info(
        'Converting the optimization result into %d suggestions...', count
    )
    suggestions = []
    end_time = datetime.datetime.now()
    for idx, best_candidate in enumerate(trials):
      # Make predictions (in the warped space).
      # Create suggestions, injecting the predictions as metadata for
      # debugging needs.
      metadata = best_candidate.metadata.ns(self._metadata_ns)
      metadata.ns('prediction_in_warped_y_space').update({
          'mean': f'{predict_mean[0, idx]}',
          'stddev': f'{predict_stddev[0, idx]}',
          'stddev_from_all': f'{predict_stddev_from_all[0, idx]}',
          'acquisition': f'{acquisition}',
          'use_ucb': 'False',
          'trust_radius': f'{tr.trust_radius}',
          'params': f'{model.params}',
      })
      metadata.ns('timing').update({'time': f'{end_time - start_time}'})
      suggestions.append(
          vz.TrialSuggestion(
              best_candidate.parameters, metadata=best_candidate.metadata
          )
      )
    return suggestions

  @profiler.record_runtime(name_prefix='VizierGPUCBPEBandit', name='suggest')
  def suggest(
      self, count: Optional[int] = None
  ) -> Sequence[vz.TrialSuggestion]:
    count = count or 1
    num_total = len(self._all_completed_trials) + len(self._all_active_trials)
    if num_total < self._num_seed_trials:
      return self._generate_seed_trials(count)

    if self._clear_jax_cache:
      jax.clear_caches()

    self._rng, rng = jax.random.split(self._rng, 2)
    data = self._trials_to_data(self._all_completed_trials)
    model = self._build_gp_model_and_optimize_parameters(data, rng)
    predictive = sp.UniformEnsemblePredictive(
        predictives=eqx.filter_jit(model.precompute_predictive)(data)
    )

    # Optimize acquisition.
    active_trial_features = self._converter.to_features(self._all_active_trials)

    tr_features = types.ModelInput(
        continuous=self._padding_schedule.pad_features(
            jnp.concatenate(
                [
                    data.features.continuous.unpad(),
                    active_trial_features.continuous.unpad(),
                ],
                axis=0,
            )
        ),
        categorical=self._padding_schedule.pad_features(
            jnp.concatenate(
                [
                    data.features.categorical.unpad(),
                    active_trial_features.categorical.unpad(),
                ],
                axis=0,
            ),
        ),
    )
    tr = acquisitions.TrustRegion(trusted=tr_features)

    acquisition_problem = copy.deepcopy(self._problem)
    acquisition_problem.metric_information = [
        vz.MetricInformation(
            name='acquisition', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    ]
    logging.info('Optimizing acquisition...')

    # TODO: Feed the eagle strategy with completed trials.
    # TODO: Change budget based on requested suggestion count.
    active_trials = list(self._all_active_trials)
    if count <= 1:
      return [
          self._suggest_one(
              active_trials, data, model, predictive, tr, acquisition_problem
          )
      ]

    suggestions = []
    if self._config.optimize_set_acquisition_for_exploration:
      if _has_new_completed_trials(
          completed_trials=self._all_completed_trials,
          active_trials=active_trials,
      ):
        suggestions.append(
            self._suggest_one(
                active_trials, data, model, predictive, tr, acquisition_problem
            )
        )
        active_trials.append(suggestions[-1].to_trial())
      return suggestions + self._suggest_batch_with_exploration(
          count - len(suggestions), active_trials, data, model, predictive, tr
      )
    else:
      for _ in range(count):
        suggestions.append(
            self._suggest_one(
                active_trials, data, model, predictive, tr, acquisition_problem
            )
        )
        active_trials.append(suggestions[-1].to_trial())
      return suggestions
