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

"""Fast pareto frontier computation using Jax."""
import functools

import jax
from jax import numpy as jnp
import jaxtyping as jt
import numpy as np
from vizier._src.pyvizier.multimetric import pareto_optimal


def _is_dominated(
    y1: jt.Float[jt.Array, "M"],
    y2: jt.Float[jt.Array, "M"],
    strict: bool = True,
) -> jt.Bool[jt.Array, ""]:
  """True if y2 > y1 (or y2 >= y1 if strict is False) every coordinate."""
  dominated_or_equal = jnp.all(y1 <= y2)
  if strict:
    return dominated_or_equal & jnp.any(y2 > y1)
  else:
    return dominated_or_equal


@functools.partial(jax.jit, static_argnames="strict")
def _is_pareto_optimal_against(
    yy: jt.Float[jt.Array, "B1 M"],
    baseline: jt.Float[jt.Array, "B2 M"],
    *,
    strict: bool,
) -> jt.Bool[jt.Array, "B1"]:
  """Computes if nothing in `baseline` dominates `yy`.

  Args:
    yy: array of shape [B1, M] where M is number of metrics.
    baseline: array of shape [B2, M] where M is number of metrics.
    strict: If true, strict dominance is used.

  Returns:
    Boolean array of shape [B1]
  """
  jax_dominated_mv = jax.vmap(
      functools.partial(_is_dominated, strict=strict), (None, 0), 0
  )  #  ([b,a], [a]) -> [b]
  jax_dominated_mm = jax.vmap(jax_dominated_mv, (0, None),
                              0)  #  ([b,a], [c,a]) -> [b,c]
  return jnp.logical_not(jnp.any(jax_dominated_mm(yy, baseline),
                                 axis=-1))  # N_y, N_b -> N_y


def is_frontier(
    ys: jt.Float[jt.ArrayLike, "B M"],
    *,
    num_shards: int = 10,
    verbose: bool = False,
) -> jt.Bool[jt.ArrayLike, "B"]:
  """Efficiently compute `_is_pareto_optimal_against(ys, ys, strict=True)`.

  Divide `ys` into shards and gradually trim down the candidates.

  Args:
    ys: Array of shape [B, M] where M is number of metrics.
    num_shards: Each sharding results in filtering, i.e. indexing the array with
      boolean vector. This operation can be very expensive and dominate the cost
      of computation. Use a moderate number sublinear in B, e.g. log(B).
    verbose:

  Returns:
    Boolean numpy Array of shape [B].
  """
  idx = np.linspace(0, ys.shape[0], num_shards).astype(np.int32)
  idx = list(reversed(idx))
  # Initialize candidates with all points
  frontier = np.ones(ys.shape[0], dtype=np.bool_)

  for begin, end in zip(idx[1:], idx[:-1]):
    candidates = ys[frontier]
    # Filter candidates by comparing against the slice.
    if verbose:
      print(f"Compare {len(candidates)} against {begin}:{end}.")
    tt = _is_pareto_optimal_against(candidates, ys[begin:end], strict=True)
    frontier[frontier] = tt
  return frontier


class JaxParetoOptimalAlgorithm(pareto_optimal.BaseParetoOptimalAlgorithm):
  """Jax-based functions for calculating pareto frontiers."""

  def is_pareto_optimal(self, points: np.ndarray) -> np.ndarray:
    """Jax-enabled Pareto frontier algorithm. See base class."""
    return np.array(is_frontier(points), dtype=bool)

  def is_pareto_optimal_against(self, points: np.ndarray,
                                dominating_points: np.ndarray,
                                strict: bool) -> np.ndarray:
    """Jax-enabled optimality/domination algorithm. See base class."""
    return np.array(
        _is_pareto_optimal_against(points, dominating_points, strict=strict),
        dtype=bool,
    )


def get_frontier(
    ys: jt.Float[jt.ArrayLike, "B M"],
    *,
    num_shards: int = 10,
    verbose: bool = True,
) -> jnp.ndarray:
  """Efficiently compute `ys[_is_pareto_optimal_against(ys, ys, strict=True)]` using iterative filtering.

  Divide `ys` into shards and gradually trim down the candidates.
  `get_frontier` doesn't call `is_frontier`, because `get_frontier` runs faster
  by not slicing the full `ys` every iteration.

  Args:
    ys: Array of shape [B, M] where M is number of metrics.
    num_shards: Each sharding results in filtering, i.e. indexing the array with
      boolean vector. This operation can be very expensive and dominate the cost
      of computation. Use a moderate number sublinear in B, e.g. log(B).
    verbose:

  Returns:
    Array of shape [B, M].
  """
  idx = np.linspace(0, ys.shape[0], num_shards).astype(np.int32)
  idx = list(reversed(idx))
  # Initialize candidates with all points
  candidates = jnp.asarray(ys)
  for begin, end in zip(idx[1:], idx[:-1]):
    # Filter candidates by comparing against the slice.
    if verbose:
      # Use print. This method won't run in production anyways.
      print(f"Compare {len(candidates)} against {begin}:{end}.")
    tt = _is_pareto_optimal_against(candidates, ys[begin:end], strict=True)
    candidates = candidates[tt]
  return candidates


@jax.jit
def pareto_rank(ys: jt.Float[jt.ArrayLike, "B M"]) -> jt.Int[jt.ArrayLike, "B"]:
  """Returns the pareto rank."""
  jax_dominated_mv = jax.vmap(
      functools.partial(_is_dominated, strict=True), (None, 0), 0
  )  #  ([b,a], [a]) -> [b]
  jax_dominated_mm = jax.vmap(
      jax_dominated_mv, (0, None), 0
  )  #  ([b,a], [c,a]) -> [b,c]
  domination_matrix = jax_dominated_mm(ys, ys)
  return jnp.sum(domination_matrix, axis=1)


def _cum_hypervolume_origin(
    points: jt.Float[jt.ArrayLike, "B M"], vector: jt.Float[jt.Array, "... M"]
) -> jt.Float[jt.Array, "B"]:
  """Returns a randomized approximation of the cumulative dominated hypervolume.

  See Section 3, Lemma 5 of https://arxiv.org/pdf/2006.04655.pdf for a fuller
  explanation of the technique. This assumes the reference point is the
  origin.

  NOTE: This returns an unnormalized hypervolume.

  Args:
    points: Any set of points with shape (num_points, dimension).
    vector: A vector of length dimension.

  Returns:
    Approximated cumulative dominated hypervolume of points[:i]. Length is
    num_points.
  """
  ratios = points / vector
  coordinate_min_ratio = jnp.min(ratios, axis=1)
  return jax.lax.cummax(coordinate_min_ratio, axis=0)**len(vector)


@jax.jit
def jax_cum_hypervolume_origin(
    points: jt.Float[jt.ArrayLike, "B M"], vectors: jt.Float[jt.Array, "B2 M"]
) -> jt.Float[jt.Array, "B B2"]:
  #  ([B,M], [B2,M]) -> [B,B2]
  cum_hypervolume_mm = jax.vmap(_cum_hypervolume_origin, (None, 0), 0)
  return jnp.mean(cum_hypervolume_mm(points, vectors), axis=0)
