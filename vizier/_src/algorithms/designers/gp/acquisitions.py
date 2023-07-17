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

"""Acquisition functions and builders implementations."""

from typing import Mapping, Optional, Protocol

import chex
import equinox as eqx
from flax import struct
import jax
from jax import numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import types


tfd = tfp.distributions
tfp_bo = tfp.experimental.bayesopt
tfpke = tfp.experimental.psd_kernels


class AcquisitionFunction(Protocol):
  """Acquisition function protocol."""

  # TODO: Acquisition functions should take
  # xs as additional input. All the data terms should go into the factory
  # or constructor.
  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.ModelInput] = None,
      labels: Optional[types.PaddedArray] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    pass


class Predictive(Protocol):
  """Protocol for predicting distributions given candidate points."""

  def predict_with_aux(
      self, features: types.ModelInput
  ) -> tuple[tfd.Distribution, chex.ArrayTree]:
    pass


class ScoreFunction(Protocol):
  """Protocol for scoring candidate points."""

  def score(
      self, xs: types.ModelInput, seed: Optional[jax.random.KeyArray] = None
  ) -> jax.Array:
    pass

  def score_with_aux(
      self, xs: types.ModelInput, seed: Optional[jax.random.KeyArray] = None
  ) -> tuple[jax.Array, chex.ArrayTree]:
    pass


def sample_from_predictive(
    predictive: Predictive,
    xs: types.ModelInput,
    num_samples: int,
    *,
    key: jax.random.KeyArray
) -> jax.Array:
  return predictive.predict_with_aux(xs)[0].sample([num_samples], seed=key)


class BayesianScoringFunction(eqx.Module):
  """Combines `Predictive` with acquisition function."""

  predictor: Predictive
  data: types.ModelData
  acquisition_fn: AcquisitionFunction

  # TODO: This should be moved out of here.
  # If set, uses trust region.
  trust_region: Optional['TrustRegion']

  def score(self, xs, seed: Optional[jax.random.KeyArray] = None) -> jax.Array:
    return self.score_with_aux(xs, seed)[0]

  def score_with_aux(
      self, xs, seed: Optional[jax.random.KeyArray] = None
  ) -> tuple[jax.Array, chex.ArrayTree]:
    pred, aux = self.predictor.predict_with_aux(xs)

    acquisition = self.acquisition_fn(
        pred, self.data.features, self.data.labels, seed
    )
    if self.trust_region is not None:
      region: TrustRegion = self.trust_region  #  type: ignore
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
      features: Optional[types.ModelInput] = None,
      labels: Optional[types.PaddedArray] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    del features, labels, seed
    return dist.mean() + self.coefficient * dist.stddev()


@struct.dataclass
class LCB(AcquisitionFunction):
  """LCB AcquisitionFunction."""

  coefficient: float = 1.8

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.ModelInput] = None,
      labels: Optional[types.PaddedArray] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    del features, labels, seed
    return dist.mean() - self.coefficient * dist.stddev()


@struct.dataclass
class HyperVolumeScalarization(AcquisitionFunction):
  """HyperVolume Scalarization acquisition function."""

  coefficient: float = 1.0

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.ModelInput] = None,
      labels: Optional[types.PaddedArray] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    del features, labels, seed
    # Uses scalarizations in https://arxiv.org/abs/2006.04655 for
    # non-convex biobjective optimization of mean vs stddev.
    return jnp.minimum(dist.mean(), self.coefficient * dist.stddev())


@struct.dataclass
class EI(AcquisitionFunction):
  """Expected Improvement acquisition function."""

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.ModelInput] = None,
      labels: Optional[types.PaddedArray] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    del features, seed
    if labels is not None:
      labels = labels.replace_fill_value(-np.inf).padded_array
    return tfp_bo.acquisition.GaussianProcessExpectedImprovement(dist, labels)()


@struct.dataclass
class PI(AcquisitionFunction):
  """Probability of Improvement acquisition function."""

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.ModelInput] = None,
      labels: Optional[types.PaddedArray] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    del features, seed
    if labels is not None:
      labels = labels.replace_fill_value(-np.inf).padded_array
    return tfp_bo.acquisition.GaussianProcessProbabilityOfImprovement(
        dist, labels
    )()


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
    threshold: The threshold for the thresholding_acquisition values to
      distinguish between the promising and unpromising regions.
    bad_acq_value: The lower bound for the main acquisition value.
    apply_tr_after: The minimum number of labels required to apply the trust
      region.
  """

  main_acquisition: AcquisitionFunction
  thresholding_acquisition: AcquisitionFunction
  bad_acq_value: float = struct.field(kw_only=True)
  threshold: Optional[float] = struct.field(kw_only=True, default=None)
  apply_tr_after: Optional[int] = struct.field(kw_only=True, default=0)

  @classmethod
  def default_ucb_pi(cls) -> 'AcquisitionTrustRegion':
    return cls(
        UCB(1.8), PI(), bad_acq_value=-1e12, threshold=0.3, apply_tr_after=0
    )

  @classmethod
  def default_ucb_lcb(cls) -> 'AcquisitionTrustRegion':
    return cls(
        UCB(1.8),
        LCB(1.8),
        bad_acq_value=-1e12,
        threshold=None,
        apply_tr_after=0,
    )

  @classmethod
  def default_ucb_lcb_wide(cls) -> 'AcquisitionTrustRegion':
    return cls(
        UCB(1.8),
        LCB(2.5),
        bad_acq_value=-1e12,
        threshold=None,
        apply_tr_after=0,
    )

  @classmethod
  def default_ucb_lcb_delay_tr(cls) -> 'AcquisitionTrustRegion':
    return cls(
        UCB(1.8),
        LCB(1.8),
        bad_acq_value=-1e12,
        threshold=None,
        apply_tr_after=5,
    )

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.ModelInput] = None,
      labels: Optional[types.PaddedArray] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    del features, seed
    threshold_values = self.thresholding_acquisition(dist, labels=labels)
    acq_values = self.main_acquisition(dist)
    threshold = -jnp.inf
    apply_tr = False
    if labels is not None:
      labels_padded = labels.replace_fill_value(np.nan).padded_array
      threshold = jnp.minimum(
          jnp.nanmean(labels_padded), jnp.nanmedian(labels_padded)
      )
      apply_tr = labels._original_shape[0] <= self.apply_tr_after
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

  num_samples: int = 100

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.ModelInput] = None,
      labels: Optional[types.PaddedArray] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    del features
    if seed is None:
      raise ValueError('QEI requires a value for `seed`.')
    return tfp_bo.acquisition.ParallelExpectedImprovement(
        dist, labels, seed=seed, num_samples=self.num_samples
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
    seed: Random seed for sampling.
  """

  coefficient: float = 1.8
  num_samples: int = 100

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.ModelInput] = None,
      labels: Optional[types.PaddedArray] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    del features
    if seed is None:
      raise ValueError('QEI requires a value for `seed`.')
    return tfp_bo.acquisition.ParallelUpperConfidenceBound(
        dist,
        labels,
        seed=seed,
        exploration=self.coefficient,
        num_samples=self.num_samples,
    )()


@struct.dataclass
class MultiAcquisitionFunction(AcquisitionFunction):
  """Wrapper that calls multiple acquisition functions."""

  acquisition_fns: Mapping[str, AcquisitionFunction]

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.ModelInput] = None,
      labels: Optional[types.PaddedArray] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    acquisitions = []
    for acquisition_fn in self.acquisition_fns.values():
      acquisitions.append(acquisition_fn(dist, features, labels))
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
      xs: (M, D) array where each element is in [0, 1].

    Returns:
      (M,) array of floating numbers, L-infinity distances to the nearest
      trusted point.
    """
    trusted = self.trusted.continuous.replace_fill_value(0.0).padded_array
    xs = xs.continuous.replace_fill_value(0.0).padded_array
    distances = jnp.abs(trusted - xs[..., jnp.newaxis, :])  # (M, N, D)
    # Mask out padded features. We set these distances to infinite since
    # they should never be considered.
    distances = jnp.where(
        self.trusted.continuous.is_missing[0][..., jnp.newaxis],
        np.inf,
        distances,
    )
    if distances.size == 0:
      return -np.inf * jnp.ones_like(xs, shape=xs.shape[:1])
    linf_distance = jnp.max(distances, axis=-1)  # (M, N)
    return jnp.min(linf_distance, axis=-1)  # (M,)
