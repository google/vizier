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

"""Hypervolume calculation (stochastic approximation) functions."""
import math
from typing import Optional, Callable


import numpy as np


def _cum_hypervolume_origin(points: np.ndarray,
                            vectors: np.ndarray) -> np.ndarray:
  """Returns a randomized approximation of the cumulative dominated hypervolume.

  See Section 3, Lemma 5 of https://arxiv.org/pdf/2006.04655.pdf for a fuller
  explanation of the technique. This assumes the reference point is the
  origin.

  NOTE: This returns an unnormalized hypervolume.

  Args:
    points: Any set of points with shape (num_points, dimension).
    vectors: Set of vectors with shape (num_vectors, dimension).

  Returns:
    Approximated cumulative dominated hypervolume of points[:i].

  Raises:
    ValueError: Points and vectors do not match in dimension.
  """
  if vectors.shape[1] != points.shape[1]:
    raise ValueError(f'Vectors shape {vectors.shape} do not match dimension of'
                     f' (second value in tuple of points shape {points.shape}')
  num_points, dimension = points.shape
  num_vectors = vectors.shape[0]
  temp_points = np.broadcast_to(points[np.newaxis, :, :],
                                [num_vectors, num_points, dimension])
  vectors = vectors.reshape([num_vectors, 1, dimension])
  # Here, ratios[i][j][k] is the kth coordinate of the jth point / ith vector.
  # Since points is (num_vectors, num_points, dimension) and vectors is
  # (num_vectors, 1, dimension), note that ratios is
  # (num_vectors, num_points, dimension).
  ratios = temp_points / vectors

  # These calculations are from Lemma 5 in above cited paper (dimension axis).
  coordinate_min_ratio = np.min(ratios, axis=2)

  # Maximizing across all points (num_points axis).
  point_max_ratio = np.maximum.accumulate(coordinate_min_ratio, axis=1)
  # Averaging across the vector axis.
  return np.mean(point_max_ratio**dimension, axis=0)


class ParetoFrontier:
  """Calculate hypervolume approximations for a Pareto frontier/set."""

  def __init__(
      self,
      points: np.ndarray,
      origin: np.ndarray,
      num_vectors: int = 10000,
      cum_hypervolume_base: Callable[[np.ndarray, np.ndarray],
                                     np.ndarray] = _cum_hypervolume_origin):
    """Takes a set of points and initializes approximating vectors.

    To use with XLA:
      from vizier.pyvizier.multimetric import xla_pareto

      front = ParetoFrontier(points, origin,
        cum_hypervolume_base = xla_pareto.jax_cum_hypervolume_origin)
      front.hypervolume(is_cumulative=True)

    Args:
      points: Any set of points with shape (num_points, dimension).
      origin: An point (1D or 2D array) with length = dimension from which
        hypervolume is computed.
      num_vectors: Number of random vectors used to approximate hypervolume.
      cum_hypervolume_base: The base algorithm used to calculate hypervolume
      from the origin. Parameters are [points, vectors].

    Raises:
      ValueError: When dimensions are mismatched between points and origin.
    """
    self._points = points
    if points.shape[1] != len(origin):
      raise ValueError(
          f'Dimension mismatch frontier points {self._points.shape}'
          f' and origin with length {len(origin)}')
    self._origin = origin
    self._cum_hypervolume_base = cum_hypervolume_base
    # Generating random vectors in the positive orthant for approximation.
    self._vectors = abs(np.random.normal(size=(num_vectors, len(origin))))
    self._vectors /= np.linalg.norm(self._vectors, axis=1)[..., np.newaxis]

  def hypervolume(self,
                  additional_points: Optional[np.ndarray] = None,
                  is_cumulative: bool = False,
                  num_shards: int = 10) -> np.ndarray:
    """Returns a randomized approximation of the dominated hypervolume.

    See Section 3, Lemma 5 of https://arxiv.org/pdf/2006.04655.pdf for a fuller
    explanation of the technique.

    Args:
      additional_points: Additional 2D array to add to hypervolume computation.
      is_cumulative: If true, a cumulative hypervolume is returned. A cumulative
      hypervolume is the a vector whose i-th entry is the running hypervolume of
      the first i-th points of points + additional_points.
      num_shards: Number of shards for breaking up self._vectors for reduced
      memory. A larger number means lower memory.

    Returns:
      Approximated dominated hypervolume of frontier and additional points.

    Raises:
      ValueError: Frontier and additional points do not match in dimension.
    """
    if additional_points is None:
      points = self._points
    else:
      if self._points.shape[1] != additional_points.shape[1]:
        raise ValueError(
            f'Dimension mismatch frontier points {self._points.shape}'
            f' and additional points {additional_points.shape}')
      points = np.concatenate((self._points, additional_points), axis=0)

    # Shift points by the origin and remove all dominated/negative points.
    points = points - self._origin
    non_positive_points = np.invert(np.all(points > 0, axis=1))
    points[non_positive_points] = 0
    if not points.size:
      return np.asarray([0.0])

    # We shard the vectors for better reduced memory usage.
    idx = np.linspace(0, len(self._vectors), num_shards + 1).astype(np.int32)
    idx = list(reversed(idx))
    # We average each vector's hypervolume estimate for a final estimate. More
    # vectors mean a more accurate estimate. (vector axis average)
    approx_hypervolume = 0
    _, dimension = points.shape
    unit_hypersphere_volume = math.pi**(
        dimension / 2) / math.gamma(dimension / 2 + 1) / 2**dimension
    for begin, end in zip(idx[1:], idx[:-1]):
      vectors = self._vectors[begin:end, :]
      # Apply maximization because Jax -> NP conversion can be imprecise.
      approx_hypervolume += np.maximum.accumulate(
          self._cum_hypervolume_base(points, vectors) * unit_hypersphere_volume
      )

    if is_cumulative:
      return np.array(approx_hypervolume / num_shards)
    else:
      return np.array(np.max(approx_hypervolume / num_shards))
