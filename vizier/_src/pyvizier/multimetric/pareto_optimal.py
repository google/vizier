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

"""Fast divide-and-conquer (DC) algorithms for computing Pareto frontier."""

import abc

import numpy as np


class BaseParetoOptimalAlgorithm(metaclass=abc.ABCMeta):
  """Base metaclass for calculating pareto frontiers."""

  @abc.abstractmethod
  def is_pareto_optimal(self, points: np.ndarray) -> np.ndarray:
    """An algorithm to find the Pareto frontier.

    Args:
      points: M-by-D 2D array of points. M = number of points, D = dimension.

    Returns:
      A boolean array of length m corresponding to each point in points. True
      if pareto optimal.
    """
    pass

  @abc.abstractmethod
  def is_pareto_optimal_against(self, points: np.ndarray, against: np.ndarray,
                                *, strict: bool) -> np.ndarray:
    """An algorithm to check if the points are pareto optimal or dominated.

    NOTE: p dominates q if p_i >= q_i and strictly if p_i != q_i.

    Dominated: A point is strictly dominated by some point in dominated_points.
    Equal: A non-strictly dominated point equal to some point in
    dominated_points.
    Optimal: A point is not dominated.

    Args:
      points: M-by-D 2D array of points to be checked.  M = number of points, D
        = dimension.
      against: N-by-D 2D array of dominating points. N = number of dominating
        points, D = dimension.
      strict: If true, non-strictly dominated (i.e. equal) points are considered
        optimal.

    Returns:
      A length N boolean array corresponding to each point in points. True means
      that the point is not dominated by any point in against.
    """
    pass

  def update_pareto_optimal(
      self, current_pareto_optimal: np.ndarray, incremental_points: np.ndarray
  ) -> np.ndarray:
    """An algorithm to update the Pareto frontier.

    Args:
      current_pareto_optimal: M-by-D 2D array of current pareto optimal points.
        M = number of points, D = dimension.
      incremental_points: N-by-D 2D array of incremental points. N = number of
        points, D = dimension.

    Returns:
      List of point indices of which are in the pareto frontier.
      Example: returns [2, 4] when only ckpt_2 and ckpt_4 are on pareto.
    """
    is_optimal = self.is_pareto_optimal(
        np.append(current_pareto_optimal, incremental_points, axis=0)
    )
    return np.asarray(is_optimal).nonzero()[0]


class NaiveParetoOptimalAlgorithm(BaseParetoOptimalAlgorithm):
  """Naive implementation of Pareto frontier calculation."""

  def is_pareto_optimal_against(self, points: np.ndarray, against: np.ndarray,
                                *, strict: bool) -> np.ndarray:
    """Naive quadratic-time implementation. See base class."""
    if against.shape[1] != points.shape[1]:
      raise ValueError(f'Shape for against {against.shape}'
                       f'does not match in last dim for points {points.shape}')
    is_optimal = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
      strict_dominating = np.any(point > against, axis=1)
      if np.all(strict_dominating):
        # Optimal point since no points dominate it.
        is_optimal[i] = True
      # Mark equal as optimal if strict is True.
      elif strict and np.all(strict_dominating
                             | np.all(point == against, axis=1)):
        is_optimal[i] = True

    return is_optimal

  def is_pareto_optimal(self, points: np.ndarray) -> np.ndarray:
    """Naive quadratic-time implementation. See base class."""
    is_optimal = np.ones(len(points), dtype=bool)
    for i, point in enumerate(points):
      if is_optimal[i]:
        # At each step, revise all potentially optimal points.
        is_optimal[is_optimal] = (
            np.any(points[is_optimal] > point, axis=1)
            | np.all(points[is_optimal] == point, axis=1))
    return is_optimal


class FastParetoOptimalAlgorithm(BaseParetoOptimalAlgorithm):
  """Fast DC/recursive algorithm deciding if points are optimal or dominated.

  NOTE: p dominates q if p_i >= q_i and strictly if p_i != q_i.

  Citation on first publication:
  https://static.aminer.org/pdf/PDF/000/211/201/on_the_computational_complexity_of_finding_the_maxima_of_a.pdf
  """

  def __init__(
      self,
      base_algorithm: BaseParetoOptimalAlgorithm = NaiveParetoOptimalAlgorithm(
      ),
      *,
      recursive_threshold: int = 10000):
    """Init.

    To use with XLA:
      from vizier.pyvizier.multimetric import xla_pareto

      algo =
      FastParetoOptimalAlgorithm(xla_pareto.JaxParetoOptimalAlgorithm())
      algo.is_pareto_optimal(points)

    Args:
      base_algorithm: Base pareto computation algorithm used as base case.
      recursive_threshold: If points are fewer than this threshold, use base
        case naive algorithm.
    """
    self._base_algorithm = base_algorithm
    self._recursive_threshold = recursive_threshold

  def is_pareto_optimal_against(self, points: np.ndarray, against: np.ndarray,
                                *, strict: bool) -> np.ndarray:
    """Fast DC algorithm/recursive algorithm. See base class."""
    # Edge case: When either points or against are empty.
    if len(points) <= 0:
      return np.array([], dtype=bool)
    if len(against) <= 0:
      return np.array(np.ones(len(points), dtype=bool))

    # Base case.
    if len(against) < self._recursive_threshold or len(
        points) < self._recursive_threshold:
      if strict:
        return np.array(
            self._base_algorithm.is_pareto_optimal_against(
                points, against, strict=True),
            dtype=bool)
      else:
        return np.array(
            self._base_algorithm.is_pareto_optimal_against(
                points, against, strict=False),
            dtype=bool)

    # Base case: If 1D, this is a simple linear time algorithm.
    if against.shape[1] <= 1:
      max_value = np.max(against)
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
              self._base_algorithm.is_pareto_optimal_against(
                  points, against, strict=True),
              dtype=bool)
        else:
          return np.array(
              self._base_algorithm.is_pareto_optimal_against(
                  points, against, strict=False),
              dtype=bool)

    # Sort dominating points from lowest to highest for first dim and split.
    sorted_dominating = against[(against[:, 0]).argsort()]
    dominating_split_index = np.searchsorted(
        sorted_dominating[:, 0], split_value, side='right')

    # Divide points, dominated_points and then recurse.
    lower_points = sorted_points[:split_index]
    upper_points = sorted_points[split_index:]
    lower_dominating = sorted_dominating[:dominating_split_index]
    upper_dominating = sorted_dominating[dominating_split_index:]

    upper_optimal = self.is_pareto_optimal_against(
        points=upper_points, against=upper_dominating, strict=strict)
    lower_optimal = self.is_pareto_optimal_against(
        points=lower_points, against=lower_dominating, strict=strict)
    # Can remove the first dimension due to sorting criteria. Also any equal
    # point is dominated because of the sorting of the first dimension.
    cross_optimal = self.is_pareto_optimal_against(
        points=lower_points[:, 1:],
        against=upper_dominating[:, 1:],
        strict=False)

    # Combine subresults and return.
    lower_optimal = lower_optimal & cross_optimal

    is_optimal = np.array(np.zeros(len(points), dtype=bool))
    is_optimal[ascending_indices[:split_index]] = lower_optimal
    is_optimal[ascending_indices[split_index:]] = upper_optimal

    return is_optimal

  def is_pareto_optimal(self, points: np.ndarray) -> np.ndarray:
    """Fast DC algorithm to find the Pareto frontier. See base class."""

    # Base case naive algorithm for find Pareto optimality.
    if len(points) <= self._recursive_threshold:
      return np.array(
          self._base_algorithm.is_pareto_optimal(points), dtype=bool)

    # Sort from lowest to highest on first dimension and split.
    ascending_indices = (points[:, 0]).argsort()
    sorted_points = points[ascending_indices]
    split_index = round(len(points) / 2)

    # Recurse on both subarrays and check for cross domination.
    lower_array = sorted_points[:split_index]
    higher_array = sorted_points[split_index:]
    higher_pareto = self.is_pareto_optimal(higher_array)
    lower_pareto = self.is_pareto_optimal(lower_array)
    cross_check = self.is_pareto_optimal_against(
        lower_array, higher_array, strict=True)
    lower_pareto = lower_pareto & cross_check

    is_optimal = np.zeros(len(points), dtype=bool)
    is_optimal[ascending_indices[:split_index]] = lower_pareto
    is_optimal[ascending_indices[split_index:]] = higher_pareto
    return is_optimal
