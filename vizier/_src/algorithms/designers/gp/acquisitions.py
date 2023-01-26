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

"""Acquisition functions implementations."""
from typing import Sequence

import chex
from jax import numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier.pyvizier import converters

tfd = tfp.distributions


# TODO: Support discretes and categoricals.
# TODO: Support custom distances.
class TrustRegion:
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

  def __init__(
      self, trusted: chex.Array, specs: Sequence[converters.NumpyArraySpec]
  ):
    """Init.

    Args:
      trusted: Array of shape (N, D) where each element is in [0, 1]. Each row
        is the D-dimensional vector representing a trusted point.
      specs: List of output specs of the `TrialToArrayConverter`.
    """
    self._trusted = trusted
    self._dof = len(specs)
    self._trust_radius = self._compute_trust_radius(self._trusted)

    max_distance = []
    for spec in specs:
      # Cap distances between one-hot encoded features so that they fall within
      # the trust region radius.
      if spec.type is converters.NumpyArraySpecType.ONEHOT_EMBEDDING:
        max_distance.extend([self._trust_radius] * spec.num_dimensions)
      else:
        max_distance.append(np.inf)
    self._max_distances = np.array(max_distance)

  def _compute_trust_radius(self, trusted: chex.Array) -> chex.Scalar:
    """Computes the trust region radius."""
    # TODO: Make hyperparameters configurable.
    min_radius = 0.2  # Hyperparameter
    dimension_factor = 5.0  # Hyperparameter

    # TODO: Discount the infeasible points.

    trust_level = (0.1 * trusted.shape[0] + 0.9 * trusted.shape[0]) / (
        dimension_factor * (self._dof + 1)
    )
    trust_region_radius = min_radius + (0.5 - min_radius) * trust_level
    return trust_region_radius

  @property
  def trust_radius(self) -> float:
    return self._trust_radius

  def min_linf_distance(self, xs: chex.Array) -> chex.Array:
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
    distances = jnp.abs(self._trusted - xs[..., jnp.newaxis, :])  # (M, N, D)
    distances_bounded = jnp.minimum(distances, self._max_distances)
    linf_distance = jnp.max(distances_bounded, axis=-1)  # (M, N)
    return jnp.min(linf_distance, axis=-1)  # (M,)
