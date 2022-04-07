"""Fast divide-and-conquer (DC) algorithms for computing Pareto frontier."""

from typing import Callable

import numpy as np


def is_pareto_optimal_against_naive(points: np.ndarray,
                                    dominating_points: np.ndarray,
                                    strict: bool) -> np.ndarray:
  """Naive iterative algorithm to check if the points are optimal or dominated.

  NOTE: p dominates q if p_i >= q_i and strictly if p_i != q_i.

  Dominated: A point is strictly dominated by some point in dominated_points.
  Equal: A non-strictly dominated point equal to some point in dominated_points.
  Optimal: A point is not dominated.

  Args:
    points: M-by-D 2D array of points to be checked.  M = number of points, D =
      dimension.
    dominating_points: N-by-D 2D array of dominating points. N = number of
      dominating points, D = dimension.
    strict: If true, non-strictly dominated (i.e. equal) points are considered
      optimal.

  Returns:
    A length N boolean array corresponding to each point in points. True means
    that the point is not dominated by some point in dominating_points.

  Raises:
    ValueError: If dominating_points and points are the same last dim.
  """
  if dominating_points.shape[1] != points.shape[1]:
    raise ValueError(f'Shape for dominating_points {dominating_points.shape}'
                     f'does not match in last dim for points {points.shape}')
  is_optimal = np.zeros(len(points), dtype=bool)
  for i, point in enumerate(points):
    strict_dominating = np.any(point > dominating_points, axis=1)
    if np.all(strict_dominating):
      # Optimal point since no points dominate it.
      is_optimal[i] = True
    # Mark equal as optimal if strict is True.
    elif strict and np.all(strict_dominating
                           | np.all(point == dominating_points, axis=1)):
      is_optimal[i] = True

  return is_optimal


def is_pareto_optimal_naive(points: np.ndarray) -> np.ndarray:
  """Naive iterative algorithm to find the Pareto frontier.

  Args:
    points: m-by-d 2D array of points. m = number of points, d = dimension.

  Returns:
    A boolean array of length m corresponding to each point in points. True
    if pareto optimal.
  """
  is_optimal = np.ones(len(points), dtype=bool)
  for i, point in enumerate(points):
    if is_optimal[i]:
      # At each step, revise all potentially optimal points.
      is_optimal[is_optimal] = (
          np.any(points[is_optimal] > point, axis=1)
          | np.all(points[is_optimal] == point, axis=1))
  return is_optimal


class FastParetoOptimalAlgorithm:
  """Fast DC algorithm deciding if points are optimal or dominated.

  NOTE: p dominates q if p_i >= q_i and strictly if p_i != q_i.

  Citation on first publication:
  https://static.aminer.org/pdf/PDF/000/211/201/on_the_computational_complexity_of_finding_the_maxima_of_a.pdf
  """

  def __init__(
      self, is_pareto_optimal_base: Callable[[np.ndarray], np.ndarray],
      is_pareto_optimal_against_base: Callable[[np.ndarray, np.ndarray, bool],
                                               np.ndarray]):
    """Init.

    To use with XLA:
      algo = FastParetoOptimalAlgorithm(xla_pareto.is_frontier,
        xla_pareto.jax_is_pareto_optimal_against)
      algo.is_pareto_optimal(points)

    Args:
      is_pareto_optimal_base: A base case pareto optimality algorithm.
      is_pareto_optimal_against_base: An base non-domination check algorithm.
        The function signature should be [points, dominating_points, strict] and
        if strict = True, then we use strict dominance.
    """
    self._is_pareto_optimal_base = is_pareto_optimal_base
    self._is_pareto_optimal_against_base = is_pareto_optimal_against_base

  def is_pareto_optimal_against(self,
                                points: np.ndarray,
                                dominating_points: np.ndarray,
                                strict: bool = True,
                                recursive_threshold: int = 10000) -> np.ndarray:
    """Fast DC algorithm deciding if points are optimal or dominated.

    NOTE: Domination of points is determined compared against dominating_points.

    Args:
      points: M-by-D 2D array of points to be checked.  M = number of points, D
        = dimension.
      dominating_points: N-by-D 2D array of dominating points. N = number of
        dominating points, D = dimension.
      strict: If true, non-strictly dominated (i.e. equal) points are considered
        optimal.
      recursive_threshold: If points are fewer than this threshold, use base
        case naive algorithm.

    Returns:
      A length N boolean array corresponding to each point in points. True means
      that the point is not dominated by some point in dominating_points.

    Raises:
      ValueError: If dominating_points and points are the same last dim.
    """
    # Edge case: When either points or dominating_points are empty.
    if len(points) <= 0:
      return np.array([], dtype=bool)
    if len(dominating_points) <= 0:
      return np.array(np.ones(len(points), dtype=bool))

    # Base case.
    if len(dominating_points) < recursive_threshold or len(
        points) < recursive_threshold:
      if strict:
        return np.array(
            self._is_pareto_optimal_against_base(points, dominating_points,
                                                 True),
            dtype=bool)
      else:
        return np.array(
            self._is_pareto_optimal_against_base(points, dominating_points,
                                                 False),
            dtype=bool)

    # Base case: If 1D, this is a simple linear time algorithm.
    if dominating_points.shape[1] <= 1:
      max_value = np.max(dominating_points)
      if strict:
        return (points >= max_value).squeeze()
      else:
        return (points > max_value).squeeze()

    # Sort points from lowest to highest for first dimension and find split.
    ascending_indices = (points[:, 0]).argsort()
    sorted_points = points[ascending_indices]
    split_index = round(len(points) / 2)
    split_value = sorted_points[split_index][0]
    while sorted_points[split_index][0] == split_value:
      split_index += 1
      # Hard to find a clean split. Resort to simple algorithm.
      if split_index == len(sorted_points):
        if strict:
          return np.array(
              self._is_pareto_optimal_against_base(points, dominating_points,
                                                   True),
              dtype=bool)
        else:
          return np.array(
              self._is_pareto_optimal_against_base(points, dominating_points,
                                                   False),
              dtype=bool)

    # Sort dominating points from lowest to highest for first dim and split.
    sorted_dominating = dominating_points[(dominating_points[:, 0]).argsort()]
    dominating_split_index = np.searchsorted(
        sorted_dominating[:, 0], split_value, side='right')

    # Divide points, dominated_points and then recurse.
    lower_points = sorted_points[:split_index]
    upper_points = sorted_points[split_index:]
    lower_dominating = sorted_dominating[:dominating_split_index]
    upper_dominating = sorted_dominating[dominating_split_index:]

    upper_optimal = self.is_pareto_optimal_against(
        points=upper_points,
        dominating_points=upper_dominating,
        strict=strict,
        recursive_threshold=recursive_threshold)
    lower_optimal = self.is_pareto_optimal_against(
        points=lower_points,
        dominating_points=lower_dominating,
        strict=strict,
        recursive_threshold=recursive_threshold)
    # Can remove the first dimension due to sorting criteria. Also any equal
    # point is dominated because of the sorting of the first dimension.
    cross_optimal = self.is_pareto_optimal_against(
        points=lower_points[:, 1:],
        dominating_points=upper_dominating[:, 1:],
        strict=False,
        recursive_threshold=recursive_threshold)

    # Combine subresults and return.
    lower_optimal = lower_optimal & cross_optimal

    is_optimal = np.array(np.zeros(len(points), dtype=bool))
    is_optimal[ascending_indices[:split_index]] = lower_optimal
    is_optimal[ascending_indices[split_index:]] = upper_optimal

    return is_optimal

  def is_pareto_optimal(self,
                        points: np.ndarray,
                        recursive_threshold: int = 10000) -> np.ndarray:
    """Fast DC algorithm to find the Pareto frontier.

    Args:
      points: m-by-d 2D array of points. m = number of points, d = dimension.
      recursive_threshold: If points are fewer than this threshold, use base
        case naive algorithm.

    Returns:
      A boolean array of length m corresponding to each point in points. True
      if pareto optimal.
    """

    # Base case naive algorthm for find Pareto optimality.
    if len(points) <= recursive_threshold:
      return np.array(self._is_pareto_optimal_base(points), dtype=bool)

    # Sort from lowest to highest on first dimension and split.
    ascending_indices = (points[:, 0]).argsort()
    sorted_points = points[ascending_indices]
    split_index = round(len(points) / 2)

    # Recurse on both subarrays and check for cross domination.
    lower_array = sorted_points[:split_index]
    higher_array = sorted_points[split_index:]
    higher_pareto = self.is_pareto_optimal(higher_array, recursive_threshold)
    lower_pareto = self.is_pareto_optimal(lower_array, recursive_threshold)
    cross_check = self.is_pareto_optimal_against(
        lower_array, higher_array, strict=True)
    lower_pareto = lower_pareto & cross_check

    is_optimal = np.zeros(len(points), dtype=bool)
    is_optimal[ascending_indices[:split_index]] = lower_pareto
    is_optimal[ascending_indices[split_index:]] = higher_pareto
    return is_optimal
