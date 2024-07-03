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

"""Acquisition functions and builders implementations."""

from typing import Callable, Mapping, Optional, Protocol

import chex
import equinox as eqx
from flax import struct
import jax
from jax import numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier._src.algorithms.designers import scalarization
from vizier._src.jax import types
from vizier._src.jax.models import continuous_only_kernel


tfd = tfp.distributions
tfp_bo = tfp.experimental.bayesopt
tfpke = tfp.experimental.psd_kernels


class AcquisitionFunction(Protocol):
  """Acquisition function protocol."""

  # TODO: Acquisition functions should take xs as additional input.
  def __call__(
      self,
      dist: tfd.Distribution,
      seed: Optional[jax.Array] = None,
  ) -> jax.Array:
    pass


class ScoreFunction(Protocol):
  """Protocol for scoring candidate points."""

  def score(self, xs: types.ModelInput, seed: jax.Array) -> jax.Array:
    pass

  def score_with_aux(
      self, xs: types.ModelInput, seed: jax.Array
  ) -> tuple[jax.Array, chex.ArrayTree]:
    pass


class Predictive(Protocol):
  """Protocol for predicting distributions given candidate points."""

  def predict_with_aux(
      self, features: types.ModelInput
  ) -> tuple[tfd.Distribution, chex.ArrayTree]:
    pass


class ScoringFunctionFactory(Protocol):

  def __call__(
      self,
      data: types.ModelData,
      predictive: Predictive,
      use_trust_region: bool = False,
  ) -> ScoreFunction:
    pass


def sample_from_predictive(
    predictive: Predictive,
    xs: types.ModelInput,
    num_samples: int,
    *,
    key: jax.Array
) -> jax.Array:
  return predictive.predict_with_aux(xs)[0].sample([num_samples], seed=key)


def get_best_labels(labels: types.PaddedArray) -> jax.Array:
  """Returns the maximum values of labels.

  A note on "labels" in TFP acquisition functions: TFP acquisition functions
  (EI, PI, qEI, qUCB) take the maximum of `"observations"` (labels) over the
  rightmost axis, which is assumed to correspond to the number of observations.
  `best_labels` has a (singleton) rightmost dimension corresponding to the
  number of metrics. The shapes therefore work out correctly, although the
  semantics are different.

  Args:
    labels: Observed labels with padded shape `(num_observations, num_metrics)`.

  Returns: Maximum label values for each metric.
  """
  if jnp.size(labels.padded_array) == 0:
    return -np.inf
  return jnp.max(labels.replace_fill_value(-np.inf).padded_array, axis=-2)


def get_worst_labels(labels: types.PaddedArray) -> jax.Array:
  """Returns the minimum values of labels.

  A note on "labels" in TFP acquisition functions: TFP acquisition functions
  (EI, PI, qEI, qUCB) take the minimum of `"observations"` (labels) over the
  rightmost axis, which is assumed to correspond to the number of observations.
  `best_labels` has a (singleton) rightmost dimension corresponding to the
  number of metrics. The shapes therefore work out correctly, although the
  semantics are different.

  Args:
    labels: Observed labels with padded shape `(num_observations, num_metrics)`.

  Returns: Minimum label values for each metric.
  """
  if jnp.size(labels.padded_array) == 0:
    return np.inf
  return jnp.min(labels.replace_fill_value(np.inf).padded_array, axis=-2)


def get_reference_point(
    labels: types.PaddedArray, scale: float = 0.1
) -> jax.Array:
  """Returns the reference point for hypervolume computations.

  Reference point is generally set to nadir - 0.1 * range. See [Ishibuchi2011]
  find 0.1 to be a robust multiplier for scaling the nadir point.

  Args:
    labels: Observed labels with padded shape `(num_observations, num_metrics)`.
    scale: The scaling factor for the range.

  Returns: Reference point.
  """
  best_labels = get_best_labels(labels)
  worst_labels = get_worst_labels(labels)
  labels_range = best_labels - worst_labels
  return worst_labels - scale * labels_range


def _apply_trust_region(
    region: 'TrustRegion',
    xs: types.ModelInput,
    acquisition: jax.Array,
    pred: tfd.Distribution,
    aux: chex.ArrayTree,
) -> tuple[jax.Array, chex.ArrayTree]:
  """Applies the trust region to acquisition values."""
  distance = region.min_linf_distance(xs)
  raw_acquisition = acquisition
  acquisition = jnp.where(
      ((distance <= region.trust_radius) | (region.trust_radius > 0.5)),
      acquisition,
      -1e12 - distance,
  )
  aux = aux | {
      'mean': pred.mean(),
      'stddev': pred.stddev(),
      'raw_acquisition': raw_acquisition,
      'linf_distance': distance,
      'radius': jnp.ones_like(distance) * region.trust_radius,
  }
  return acquisition, aux


class BayesianScoringFunction(eqx.Module):
  """Combines `Predictive` with acquisition function."""

  predictor: Predictive
  acquisition_fn: AcquisitionFunction

  # TODO: This should be moved out of here.
  # If set, uses trust region.
  trust_region: Optional['TrustRegion']

  def score(self, xs, seed: jax.Array) -> jax.Array:
    return self.score_with_aux(xs, seed)[0]

  def score_with_aux(
      self, xs, seed: jax.Array
  ) -> tuple[jax.Array, chex.ArrayTree]:
    jax.monitoring.record_event(
        '/vizier/jax/acquisitions/bayesian_scoring_function/score_with_aux/traced'
    )
    pred, aux = self.predictor.predict_with_aux(xs)

    acquisition = self.acquisition_fn(pred, seed=seed)
    if self.trust_region is not None:
      acquisition, aux = _apply_trust_region(
          self.trust_region,
          xs,
          acquisition,
          pred,
          aux,
      )
    return acquisition, aux


# Vizier library acquisition functions use `flax.struct`, instead of `attrs` and
# a hash function, so that acquisition functions can be passed as args to JIT-ed
# functions without triggering retracing when attribute values change.
@struct.dataclass
class UCB(AcquisitionFunction):
  """UCB AcquisitionFunction."""

  coefficient: float = 1.8

  def __call__(
      self,
      dist: tfd.Distribution,
      seed: Optional[jax.Array] = None,
  ) -> jax.Array:
    del seed
    return dist.mean() + self.coefficient * dist.stddev()


@struct.dataclass
class LCB(AcquisitionFunction):
  """LCB AcquisitionFunction."""

  coefficient: float = 1.8

  def __call__(
      self,
      dist: tfd.Distribution,
      seed: Optional[jax.Array] = None,
  ) -> jax.Array:
    del seed
    return dist.mean() - self.coefficient * dist.stddev()


@struct.dataclass
class HyperVolumeScalarization(AcquisitionFunction):
  """HyperVolume Scalarization acquisition function."""

  coefficient: float = 1.0

  def __call__(
      self,
      dist: tfd.Distribution,
      seed: Optional[jax.Array] = None,
  ) -> jax.Array:
    del seed
    # Uses scalarizations in https://arxiv.org/abs/2006.04655 for
    # non-convex biobjective optimization of mean vs stddev.
    return jnp.minimum(dist.mean(), self.coefficient * dist.stddev())


@struct.dataclass
class EI(AcquisitionFunction):
  """Expected Improvement acquisition function."""

  best_labels: jax.Array

  def __call__(
      self,
      dist: tfd.Distribution,
      seed: Optional[jax.Array] = None,
  ) -> jax.Array:
    del seed
    return tfp_bo.acquisition.GaussianProcessExpectedImprovement(
        dist, observations=self.best_labels
    )()


@struct.dataclass
class PI(AcquisitionFunction):
  """Probability of Improvement acquisition function."""

  best_labels: jax.Array

  def __call__(
      self,
      dist: tfd.Distribution,
      seed: Optional[jax.Array] = None,
  ) -> jax.Array:
    del seed
    return tfp_bo.acquisition.GaussianProcessProbabilityOfImprovement(
        dist, observations=self.best_labels
    )()


class MaxValueEntropySearch(eqx.Module):
  """MES score function. Implements the `ScoreFunction` protocol."""

  tfp_mes: tfp_bo.acquisition.GaussianProcessMaxValueEntropySearch
  trust_region: Optional['TrustRegion']

  @classmethod
  def scoring_fn_factory(
      cls,
      data: types.ModelData,
      predictive: Predictive,
      use_trust_region: bool = False,
  ) -> 'MaxValueEntropySearch':
    """Builds a MaxValueEntropySearch scoring function."""
    if data.features.categorical.shape[-1] > 0:
      raise ValueError('Categorical data is not supported.')
    pred_gp = predictive.predict_with_aux(data.features)[0]
    continuous_gp = pred_gp.copy(
        kernel=continuous_only_kernel.EmptyCategoricalKernel(pred_gp.kernel),
        observation_index_points=(pred_gp.observation_index_points.continuous),
        index_points=jnp.zeros_like(
            pred_gp.index_points.continuous,
            shape=(0,) + pred_gp.index_points.continuous.shape[1:],
        ),
        # pylint: disable=protected-access
        _conditional_kernel=continuous_only_kernel.EmptyCategoricalKernel(
            pred_gp._conditional_kernel
        ),
        _conditional_mean_fn=lambda x: pred_gp._conditional_mean_fn(  # pylint: disable=g-long-lambda
            continuous_only_kernel._continuous_to_cacv(x)
        ),
        # pylint: enable=protected-access
    )

    observations = pred_gp.observations
    # Allow the score function to build with no observations.
    if not observations.size:
      observations = jnp.asarray([-jnp.inf])
    tfp_mes = tfp_bo.acquisition.GaussianProcessMaxValueEntropySearch(
        predictive_distribution=continuous_gp,
        observations=observations,
        # TODO: Finalize API and plumb seed through.
        seed=jax.random.PRNGKey(123),
        num_max_value_samples=100,
    )
    trust_region = TrustRegion(data.features) if use_trust_region else None
    return cls(tfp_mes, trust_region=trust_region)

  def score(self, xs: types.ModelInput, seed: jax.Array) -> jax.Array:
    return self.score_with_aux(xs, seed)[0]

  def score_with_aux(
      self, xs: types.ModelInput, seed: jax.Array
  ) -> tuple[jax.Array, chex.ArrayTree]:
    del seed
    if xs.categorical.shape[-1] > 0:
      raise ValueError('Categorical data is not supported.')
    acquisition = self.tfp_mes(index_points=xs.continuous.padded_array)
    aux = {}
    if self.trust_region is not None:
      acquisition, aux = _apply_trust_region(
          self.trust_region,
          xs,
          acquisition,
          self.tfp_mes.predictive_distribution,
          aux,
      )
    return acquisition, aux


def bayesian_scoring_function_factory(
    acquisition_fn_factory: Callable[[types.ModelData], AcquisitionFunction],
) -> ScoringFunctionFactory:
  """Builds a ScoringFunctionFactory."""

  def f(
      data: types.ModelData,
      predictive: Predictive,
      use_trust_region: bool = False,
  ) -> ScoreFunction:
    acquisition_fn = acquisition_fn_factory(data)
    trust_region = TrustRegion(data.features) if use_trust_region else None
    return BayesianScoringFunction(predictive, acquisition_fn, trust_region)

  return f


@struct.dataclass
class AcquisitionTrustRegion(AcquisitionFunction):
  """Acquisition masked by a thresholding acquisition values.

  When optimizing the main acquisition, the goal is to consider idea is to
  disregard the region in the search space with
  thresholding_acquisition values lower than a threshold.
  Attributes:
    main_acquisition: The main acquisition function.
    thresholding_acquisition: The acquisition function used to detect promising
      (trust) regions.
    bad_acq_value: The lower bound for the main acquisition value.
    labels:
    threshold: The threshold for the thresholding_acquisition values to
      distinguish between the promising and unpromising regions.
    apply_tr_after: The minimum number of labels required to apply the trust
      region.
  """

  main_acquisition: AcquisitionFunction
  thresholding_acquisition: AcquisitionFunction
  bad_acq_value: float = struct.field(kw_only=True)
  labels: Optional[types.PaddedArray] = struct.field(kw_only=True)
  threshold: Optional[float] = struct.field(kw_only=True, default=None)
  apply_tr_after: Optional[int] = struct.field(kw_only=True, default=0)

  # TODO: Move factories to `vza_designer_factory.py`.
  @classmethod
  def default_ucb_pi(cls, data: types.ModelData) -> 'AcquisitionTrustRegion':
    best_labels = get_best_labels(data.labels)
    return cls(
        UCB(1.8),
        PI(best_labels),
        bad_acq_value=-1e12,
        labels=data.labels,
        threshold=0.3,
        apply_tr_after=0,
    )

  @classmethod
  def default_ucb_lcb(cls, data: types.ModelData) -> 'AcquisitionTrustRegion':
    return cls(
        UCB(1.8),
        LCB(1.8),
        labels=data.labels,
        bad_acq_value=-1e12,
        threshold=None,
        apply_tr_after=0,
    )

  @classmethod
  def default_ucb_lcb_wide(
      cls, data: types.ModelData
  ) -> 'AcquisitionTrustRegion':
    return cls(
        UCB(1.8),
        LCB(2.5),
        labels=data.labels,
        bad_acq_value=-1e12,
        threshold=None,
        apply_tr_after=0,
    )

  @classmethod
  def default_ucb_lcb_delay_tr(
      cls, data: types.ModelData
  ) -> 'AcquisitionTrustRegion':
    return cls(
        UCB(1.8),
        LCB(1.8),
        labels=data.labels,
        bad_acq_value=-1e12,
        threshold=None,
        apply_tr_after=5,
    )

  def __call__(
      self,
      dist: tfd.Distribution,
      seed: Optional[jax.Array] = None,
  ) -> jax.Array:
    del seed
    threshold_values = self.thresholding_acquisition(dist)
    acq_values = self.main_acquisition(dist)

    # TODO: Refactor so that the following is handled in the factory
    # instead.
    threshold = -jnp.inf
    apply_tr = False
    if self.labels is not None:
      labels_padded = self.labels.replace_fill_value(np.nan).padded_array
      threshold = jnp.minimum(
          jnp.nanmean(labels_padded), jnp.nanmedian(labels_padded)
      )
      apply_tr = self.labels._original_shape[0] <= self.apply_tr_after
    if self.threshold is not None:
      threshold = self.threshold
    cond = jnp.isnan(threshold) | (threshold_values >= threshold) | apply_tr
    return jnp.where(
        cond,
        acq_values,
        self.bad_acq_value - threshold_values,
    )


@struct.dataclass
class QEI(AcquisitionFunction):
  """Sampling-based batch expected improvement."""

  best_labels: jax.Array
  num_samples: int = 100

  def __call__(
      self,
      dist: tfd.Distribution,
      seed: Optional[jax.Array] = None,
  ) -> jax.Array:
    if seed is None:
      raise ValueError('QEI requires a value for `seed`.')
    return tfp_bo.acquisition.ParallelExpectedImprovement(
        dist,
        observations=self.best_labels,
        seed=seed,
        num_samples=self.num_samples,
    )()


@struct.dataclass
class QPI(AcquisitionFunction):
  """Sampling-based batch probability of improvement."""

  best_labels: jax.Array
  num_samples: int = 100

  def __call__(
      self,
      dist: tfd.Distribution,
      seed: Optional[jax.Array] = None,
  ) -> jax.Array:
    if seed is None:
      raise ValueError('QPI requires a value for `seed`.')
    return tfp_bo.acquisition.ParallelProbabilityOfImprovement(
        dist,
        observations=self.best_labels,
        seed=seed,
        num_samples=self.num_samples,
    )()


@struct.dataclass
class QUCB(AcquisitionFunction):
  """Sampling-based batch upper confidence bound.

  Attributes:
    coefficient: UCB coefficient. For a Gaussian distribution, note that
      `UCB(coefficient=c)` is equivalent to `QUCB(coefficient=c * sqrt(pi / 2))`
      if QUCB batch size is 1. See the TensorFlow Probability docs for more
      details:
      https://www.tensorflow.org/probability/api_docs/python/tfp/experimental/bayesopt/acquisition/ParallelUpperConfidenceBound
    num_samples: Number of distribution samples used to compute qUCB.
  """

  coefficient: float = 1.8
  num_samples: int = 100

  def __call__(
      self,
      dist: tfd.Distribution,
      seed: Optional[jax.Array] = None,
  ) -> jax.Array:
    if seed is None:
      raise ValueError('QEI requires a value for `seed`.')
    return tfp_bo.acquisition.ParallelUpperConfidenceBound(
        dist,
        seed=seed,
        exploration=self.coefficient,
        num_samples=self.num_samples,
        observations=None,
    )()


# TODO: What do we end up jitting? If we end up directly jitting this call
# then we should make it `eqx.Module` and set
# `reduction_fn=eqx.field(static=True)` instead.
@struct.dataclass
class ScalarizedAcquisition(AcquisitionFunction):
  """Wrapper that scalarizes multiple objective before acquisition eval."""

  acquisition_fn: AcquisitionFunction
  scalarizer: scalarization.Scalarization
  reduction_fn: Callable[[jax.Array], jax.Array] = struct.field(
      pytree_node=False, default=lambda x: x
  )

  def __call__(
      self,
      dist: tfd.Distribution,
      seed: Optional[jax.Array] = None,
  ) -> jax.Array:
    scalarized = self.scalarizer(self.acquisition_fn(dist, seed).squeeze())
    return self.reduction_fn(scalarized)


@struct.dataclass
class MultiAcquisitionFunction(AcquisitionFunction):
  """Wrapper that calls multiple acquisition functions."""

  acquisition_fns: Mapping[str, AcquisitionFunction]

  def __call__(
      self,
      dist: tfd.Distribution,
      seed: Optional[jax.Array] = None,
  ) -> jax.Array:
    acquisitions = []
    if seed is None:
      seeds = [None] * len(self.acquisition_fns)
    else:
      seeds = jax.random.split(seed, num=len(self.acquisition_fns))
    for i, acquisition_fn in enumerate(self.acquisition_fns.values()):
      acquisitions.append(acquisition_fn(dist, seeds[i]))
    # TODO: Change the return type to a dict with the same
    # structure as `acquisition_fns` to clarify the meaning of the return
    # values.
    return jnp.stack(acquisitions, axis=0)


# TODO: Support discretes and categoricals.
# TODO: Support custom distances.
class TrustRegion(eqx.Module):
  """L-inf norm based TrustRegion.

  Limits the suggestion within the union of small L-inf norm balls around each
  of the trusted points, which are in most cases observed points. The radius
  of the L-inf norm ball grows in the number of observed points.

  Assumes that all points are in the unit hypercube.

  The trust region can be used e.g. during acquisition optimization:
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    features, labels = converter.to_xy(trials)
    tr = TrustRegion(features, converter.output_specs)
    # xs is a point in the search space.
    distance = tr.min_linf_distance(xs)
    if distance <= tr.trust_radius:
      print('xs in trust region')
  """

  trusted: types.ModelInput

  @property
  def trust_radius(self) -> jax.Array:
    # TODO: Make hyperparameters configurable.
    min_radius = 0.2  # Hyperparameter
    dimension_factor = 5.0  # Hyperparameter

    # pylint: disable=protected-access
    dof = (
        self.trusted.continuous._original_shape[-1]
        + self.trusted.categorical._original_shape[-1]
    )
    # pylint: enable=protected-access
    num_obs = jnp.sum(~self.trusted.continuous.is_missing[0])
    # TODO: Discount the infeasible points. The 0.1 and 0.9 split
    # is associated with weights to feasible and infeasbile trials.
    trust_level = (0.1 * num_obs + 0.9 * num_obs) / (
        dimension_factor * (dof + 1)
    )
    return jnp.where(
        num_obs == 0, 1.0, min_radius + (0.5 - min_radius) * trust_level
    )

  def min_linf_distance(self, xs: types.ModelInput) -> jax.Array:
    """l-inf norm distance to the closest trusted point.

    Caps distances between one-hot encoded features to the trust-region radius,
    so that the trust region cutoff does not discourage exploration of these
    features.

    Args:
      xs: (M_0, M_1, ..., D) array where each element is in [0, 1]. The leading
        axes are the batch dimensions, and the last axis is the feature
        dimension.

    Returns:
      (M_0, M_1, ...) array of floating numbers that has one fewer axis than
      `xs`: L-infinity distances to the nearest trusted point.
    """
    trusted = self.trusted.continuous.replace_fill_value(
        0.0
    ).padded_array  # (N, D)
    xs = xs.continuous.replace_fill_value(0.0).padded_array
    distances = jnp.abs(
        trusted - xs[..., jnp.newaxis, :]
    )  # (M_0, M_1, ..,, N, D)
    # Mask out padded features. We set these distances to infinite since
    # they should never be considered.
    distances = jnp.where(
        self.trusted.continuous.is_missing[0][..., jnp.newaxis],
        np.inf,
        distances,
    )
    if distances.size == 0:
      return -np.inf * jnp.ones_like(xs, shape=xs.shape[:-1])
    linf_distance = jnp.max(distances, axis=-1)  # (M_0, M_1, ..., N)
    return jnp.min(linf_distance, axis=-1)  # (M_0, M_1, ...)
