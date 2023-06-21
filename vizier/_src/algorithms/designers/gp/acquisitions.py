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

from typing import Mapping, Optional, Protocol, Sequence

import chex
import equinox as eqx
from flax import struct
import jax
from jax import numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import types
from vizier.pyvizier import converters


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
      features: Optional[types.Features] = None,
      labels: Optional[types.Array] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    pass


class Predictive(Protocol):
  """Protocol for predicting distributions given candidate points."""

  def predict_with_aux(
      self, features: types.Features
  ) -> tuple[tfd.Distribution, chex.ArrayTree]:
    pass


class ScoreFunction(Protocol):
  """Protocol for scoring candidate points."""

  def score(
      self, xs: types.Features, seed: Optional[jax.random.KeyArray] = None
  ) -> jax.Array:
    pass

  def score_with_aux(
      self, xs: types.Features, seed: Optional[jax.random.KeyArray] = None
  ) -> tuple[jax.Array, chex.ArrayTree]:
    pass


def sample_from_predictive(
    predictive: Predictive,
    xs: chex.ArrayTree,
    num_samples: int,
    *,
    key: jax.random.KeyArray
) -> chex.ArrayTree:
  return predictive.predict_with_aux(xs)[0].sample([num_samples], seed=key)


class BayesianScoringFunction(eqx.Module):
  """Combines `Predictive` with acquisition function."""

  predictor: Predictive
  data: types.StochasticProcessModelData
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
      features: Optional[types.Features] = None,
      labels: Optional[types.Array] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    del features, labels, seed
    return dist.mean() + self.coefficient * dist.stddev()


@struct.dataclass
class HyperVolumeScalarization(AcquisitionFunction):
  """HyperVolume Scalarization acquisition function."""

  coefficient: float = 1.0

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.Features] = None,
      labels: Optional[types.Array] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    del features, labels, seed
    # Uses scalarizations in https://arxiv.org/abs/2006.04655 for
    # non-convex biobjective optimization of mean vs stddev.
    return jnp.minimum(dist.mean(), self.coefficient * dist.stddev())


@struct.dataclass
class EI(AcquisitionFunction):

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.Features] = None,
      labels: Optional[types.Array] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    del features, seed
    return tfp_bo.acquisition.GaussianProcessExpectedImprovement(dist, labels)()


@struct.dataclass
class PI(AcquisitionFunction):

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.Features] = None,
      labels: Optional[types.Array] = None,
      seed: Optional[jax.random.KeyArray] = None,
  ) -> jax.Array:
    del features, seed
    return tfp_bo.acquisition.GaussianProcessProbabilityOfImprovement(
        dist, labels
    )()


@struct.dataclass
class QEI(AcquisitionFunction):
  """Sampling-based batch expected improvement."""

  num_samples: int = 100

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[types.Features] = None,
      labels: Optional[types.Array] = None,
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
      features: Optional[types.Features] = None,
      labels: Optional[types.Array] = None,
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
      features: Optional[types.Features] = None,
      labels: Optional[types.Array] = None,
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
  max_distances: jax.Array = eqx.field(
      converter=lambda x: jnp.asarray(x, dtype=jnp.float64)
  )
  data: types.StochasticProcessModelData
  trust_radius: jax.Array = eqx.field(
      converter=lambda x: jnp.asarray(x, dtype=jnp.float64)
  )

  @classmethod
  def build(
      cls,
      specs: Optional[Sequence[converters.NumpyArraySpec]],
      data: types.StochasticProcessModelData,
  ) -> 'TrustRegion':
    """Init.

    Args:
      specs: List of output specs of the `TrialToArrayConverter`.
      data: Structure of observed features and labels, with optional boolean
        masks.

    Returns:
      trust_region
    """

    # TODO: Make hyperparameters configurable.
    min_radius = 0.2  # Hyperparameter
    dimension_factor = 5.0  # Hyperparameter

    if isinstance(data.features, types.ContinuousAndCategoricalArray):
      dof = sum(x.shape[-1] for x in jax.tree_util.tree_leaves(data.features))
    else:
      dof = len(specs)
    if data.label_is_missing is None:
      num_obs = data.labels.shape[0]
    else:
      num_obs = jnp.sum(~data.label_is_missing)
    # TODO: Discount the infeasible points. The 0.1 and 0.9 split
    # is associated with weights to feasible and infeasbile trials.
    trust_level = (0.1 * num_obs + 0.9 * num_obs) / (
        dimension_factor * (dof + 1)
    )
    trust_radius = min_radius + (0.5 - min_radius) * trust_level

    if num_obs == 0:
      trust_radius = 1.0

    if isinstance(data.features, types.ContinuousAndCategoricalArray):
      max_distance = [np.inf] * data.features.continuous.shape[-1]
    else:
      max_distance = []
      for spec in specs:
        # Cap distances between one-hot encoded features so that they fall
        # within the trust region radius.
        if spec.type is converters.NumpyArraySpecType.ONEHOT_EMBEDDING:
          max_distance.extend([trust_radius] * spec.num_dimensions)
        else:
          max_distance.append(np.inf)
      if data.dimension_is_missing is not None:
        # These extra dimensions should be ignored.
        max_distance.extend(
            [0.0] * (data.dimension_is_missing.shape[-1] - len(max_distance))  # pytype: disable=attribute-error
        )

    max_distances = np.array(max_distance)
    return TrustRegion(max_distances, data, trust_radius)

  def min_linf_distance(self, xs: types.Features) -> jax.Array:
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
    trusted = self.data.features
    if isinstance(trusted, types.ContinuousAndCategoricalArray):
      trusted = trusted.continuous
    if isinstance(xs, types.ContinuousAndCategoricalArray):
      xs = xs.continuous
    if self.data.dimension_is_missing is not None:
      # Mask out padded dimensions
      trusted = jnp.where(
          self.data.dimension_is_missing, 0.0, self.data.features
      )
      xs = jnp.where(self.data.dimension_is_missing, jnp.zeros_like(xs), xs)
    distances = jnp.abs(trusted - xs[..., jnp.newaxis, :])  # (M, N, D)
    if self.data.label_is_missing is not None:
      # Mask out padded features. We set these distances to infinite since
      # they should never be considered.
      distances = jnp.where(
          self.data.label_is_missing[..., jnp.newaxis], np.inf, distances
      )
    distances_bounded = jnp.minimum(distances, self.max_distances)
    linf_distance = jnp.max(distances_bounded, axis=-1)  # (M, N)
    return jnp.min(linf_distance, axis=-1)  # (M,)
