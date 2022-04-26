"""Fast pareto frontier computation using Jax."""

import functools
from typing import Any

import jax
from jax import numpy as jnp
import numpy as np

from vizier._src.pyvizier.multimetric import pareto_optimal

# Type aliases for readability only.
# We don't over-specify for compatibility between jax and np.
Matrix = Any
Vector = Any
Scalar = Any


# y2 dominates y1 if y2 >= y1 coordinate wise. This is strict if we also have
# y2 > y1 in some coordinate.
def _jax_dominated(y1: Vector,
                   y2: Vector,
                   strict: bool = True) -> Scalar:  # ([a], [a]) -> []
  dominated_or_equal = jnp.all(y1 <= y2)
  if strict:
    return dominated_or_equal & jnp.any(y2 > y1)
  else:
    return dominated_or_equal


def _is_pareto_optimal_against(yy: Matrix, baseline: Matrix, *,
                               strict: bool) -> Vector:
  """Computes if nothing in `baseline` dominates `yy`.

  Args:
    yy: array of shape [N_y, M] where M is number of metrics.
    baseline: array of shape [N_b, M] where M is number of metrics.
    strict: If true, strict dominance is used.

  Returns:
    Boolean array of shape [N_y]
  """
  jax_dominated_mv = jax.vmap(
      functools.partial(_jax_dominated, strict=strict), (None, 0),
      0)  #  ([b,a], [a]) -> [b]
  jax_dominated_mm = jax.vmap(jax_dominated_mv, (0, None),
                              0)  #  ([b,a], [c,a]) -> [b,c]
  return jnp.logical_not(jnp.any(jax_dominated_mm(yy, baseline),
                                 axis=-1))  # N_y, N_b -> N_y


_jax_is_pareto_optimal_against = jax.jit(
    _is_pareto_optimal_against, static_argnames='strict')


def is_frontier(ys: Matrix,
                *,
                num_shards: int = 10,
                verbose: bool = False) -> jnp.ndarray:
  """Efficiently compute `_jax_is_pareto_optimal_against(ys, ys, strict=True)`.

  Divide `ys` into shards and gradually trim down the candidates.

  Args:
    ys: Array of shape [N_y, M] where M is number of metrics.
    num_shards: Each sharding results in filtering, i.e. indexing the array with
      boolean vector. This operation can be very expensive and dominate the cost
      of computation. Use a moderate number sublinear in N_y, e.g. log(N).
    verbose:

  Returns:
    Boolean Array of shape [N_y].
  """
  idx = np.linspace(0, ys.shape[0], num_shards).astype(np.int32)
  idx = list(reversed(idx))
  # Initialize candidates with all points
  frontier = np.ones(ys.shape[0], dtype=np.bool_)

  for begin, end in zip(idx[1:], idx[:-1]):
    candidates = ys[frontier]
    # Filter candidates by comparing against the slice.
    if verbose:
      print(f'Compare {len(candidates)} against {begin}:{end}.')
    tt = _jax_is_pareto_optimal_against(candidates, ys[begin:end], strict=True)
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
        _jax_is_pareto_optimal_against(points, dominating_points, strict),
        dtype=bool)


def get_frontier(ys: Matrix,
                 *,
                 num_shards: int = 10,
                 verbose: bool = True) -> jnp.ndarray:
  """Efficiently compute `ys[_jax_is_pareto_optimal_against(ys, ys, strict=True)]` using iterative filtering.

  Divide `ys` into shards and gradually trim down the candidates.
  `get_frontier` doesn't call `is_frontier`, because `get_frontier` runs faster
  by not slicing the full `ys` every iteration.

  Args:
    ys: Array of shape [N_y, M] where M is number of metrics.
    num_shards: Each sharding results in filtering, i.e. indexing the array with
      boolean vector. This operation can be very expensive and dominate the cost
      of computation. Use a moderate number sublinear in N_y, e.g. log(N).
    verbose:

  Returns:
    Array of shape [N_y, M].
  """
  idx = np.linspace(0, ys.shape[0], num_shards).astype(np.int32)
  idx = list(reversed(idx))
  # Initialize candidates with all points
  candidates = jnp.asarray(ys)
  for begin, end in zip(idx[1:], idx[:-1]):
    # Filter candidates by comparing against the slice.
    if verbose:
      # Use print. This method won't run in production anyways.
      print(f'Compare {len(candidates)} against {begin}:{end}.')
    tt = _jax_is_pareto_optimal_against(candidates, ys[begin:end], strict=True)
    candidates = candidates[tt]
  return candidates


@jax.jit
def pareto_rank(ys: Matrix) -> jnp.ndarray:
  jax_dominated_mv = jax.vmap(
      functools.partial(_jax_dominated, strict=True), (None, 0),
      0)  #  ([b,a], [a]) -> [b]
  jax_dominated_mm = jax.vmap(jax_dominated_mv, (0, None),
                              0)  #  ([b,a], [c,a]) -> [b,c]
  domination_matrix = jax_dominated_mm(ys, ys)
  return jnp.sum(domination_matrix, axis=1)


def _cum_hypervolume_origin(points: Matrix, vector: Vector) -> Vector:
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


_cum_hypervolume_mm = jax.vmap(_cum_hypervolume_origin, (None, 0),
                               0)  #  ([b,a], [c,a]) -> [b,c]


@jax.jit
def jax_cum_hypervolume_origin(points, vectors):
  return jnp.mean(_cum_hypervolume_mm(points, vectors), axis=0)
