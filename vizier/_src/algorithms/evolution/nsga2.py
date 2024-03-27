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

"""NSGA-II algorithm: https://ieeexplore.ieee.org/document/996017."""

from typing import Callable, Optional, Tuple

import attr
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.evolution import numpy_populations
from vizier._src.algorithms.evolution import templates

Population = numpy_populations.Population
Offspring = numpy_populations.Offspring
Mutation = templates.Mutation


def _pareto_rank(ys: np.ndarray) -> np.ndarray:
  """Pareto rank, which is the number of points dominating it.

  Args:
    ys: (number of population) x (number of metrics) array.

  Returns:
    (number of population) integer array.
  """
  if ys.shape[0] == 0:
    return np.zeros([0])
  dominated = [np.all(ys <= r, axis=-1) & np.any(r > ys, axis=-1) for r in ys]
  return np.sum(np.stack(dominated), axis=0)


def _crowding_distance(ys: np.ndarray) -> np.ndarray:
  """Crowding distance.

  Args:
    ys: (number of population) x (number of metrics) array.

  Returns:
    (number of population) float32 array. Higher numbers mean less crowding
    and more desirable.
  """
  scores = np.zeros([ys.shape[0]], dtype=np.float32)
  for m in range(ys.shape[1]):
    # Sort by the m-th metric.
    yy = ys[:, m]  # Shape: (num_population,)
    sid = np.argsort(yy)

    # Boundary are assigned infinity.
    scores[sid[0]] += np.inf
    scores[sid[-1]] += np.inf

    # Compute the crowding distance.
    yrange = yy[sid[-1]] - yy[sid[0]] + np.finfo(np.float32).eps
    scores[sid[1:-1]] += (yy[sid[2:]] - yy[sid[:-2]]) / yrange
  return scores


def _constraint_violation(ys: np.ndarray) -> np.ndarray:
  """Counts the constraints violated.

  Args:
    ys: (number of population) x (number of metrics) array.

  Returns:
    (number of population) array of integers.
  """
  return np.sum(ys < 0, axis=1)


def _select_by(ys: np.ndarray, target: int) -> Tuple[np.ndarray, np.ndarray]:
  """Returns a boolean index array for the top `target` elements of ys.

  This method is tough to parse. Please improve the API if you see a better
  design!

  Args:
    ys: Array of shape [M]. Entries are expected to have a small set of unique
      values.
    target: Count to return.

  Returns:
    A tuple of two bolean index arrays `top` and `border`.
     * `ys[top]` has length less than or equal to `target`. They are within
       top `target`.
     * `ys[top | border]` has length greater than or equal to `target`.
     * `ys[border]` have all-identical entries. Callers should break ties
       among them.
     * `top & border` is all False.
  """
  if ys.shape[0] <= target:
    return (
        np.ones(ys.shape[:1], dtype=np.bool_),
        np.zeros(ys.shape[:1], dtype=np.bool_),
    )
  unique, counts = np.unique(ys, return_counts=True)
  cutoffidx = np.argmax(np.cumsum(counts) > target)
  cutoffnumber = unique[cutoffidx]
  return ys < cutoffnumber, ys == cutoffnumber


class NSGA2Survival(templates.Survival):
  """NSGA2 Survival step.

  Reference: https://ieeexplore.ieee.org/document/996017
  """

  def __init__(
      self,
      target_size: int,
      *,
      ranking_fn: Callable[[np.ndarray], np.ndarray] = _pareto_rank,
      eviction_limit: Optional[int] = None
  ):
    """Init.

    Args:
      target_size: select() method reduces the population to this size, unless
        the population is already at or below this size in which case select()
        is a no-op.
      ranking_fn: Takes (number of population) x (number of metrics) array of
        floating numbers and returns (number of population) array of integers,
        representing the pareto rank aka number of points it is dominated by.
        For performance-sensitive applications, plug in an XLA-compiled
        function.
      eviction_limit: Evict genes that were alive for this many generations.
    """
    self._target_size = target_size
    self._ranking_fn = ranking_fn
    self._eviction_limit = eviction_limit or float('inf')

  def select(self, population: Population) -> Population:
    """Applies survival process.

    Sorted all points by 3-tuple
    1. Descending order of safety constraint violation score. Zero
      means no violations.
    2. Ascending order of how many points it's dominated by.
    3. Descending order of crowding distance.

    Args:
      population:

    Returns:
      Population of size self._population_size.
    """
    population = population[population.ages < self._eviction_limit]
    if not population:
      # return empty.
      return population

    selected = population.empty_like()
    # Sort by the safety constraint.
    if selected.cs.shape[1]:
      top, border = _select_by(
          _constraint_violation(population.cs), target=self._target_size
      )
      selected += population[top]
      population = population[border]

    # Sort by the pareto rank.
    pareto_ranks = self._ranking_fn(population.ys)
    top, border = _select_by(
        pareto_ranks, target=self._target_size - len(selected)
    )
    selected += population[top]
    population = population[border]

    # Sort by the distance. Include the points that are already selected for
    # the computation.
    # Flip the sign so it works with ascending sort.
    distance = -_crowding_distance((selected + population).ys)
    sids = np.argsort(distance)
    # Selected points have fewer constraint violations or better pareto rank.
    # Regardless of the distance, they remain selected. Rank the remainder only.
    sids = sids[sids >= len(selected)] - len(selected)
    selected += population[sids[: self._target_size - len(selected)]]

    return attr.evolve(selected, ages=selected.ages + 1)


class NSGA2Designer(
    templates.CanonicalEvolutionDesigner[Population, Offspring]
):
  """NSGA2 Designer.

  Reference: https://ieeexplore.ieee.org/document/996017
  """

  def __init__(
      self,
      problem: vz.ProblemStatement,
      population_size: int = 50,
      first_survival_after: Optional[int] = None,
      *,
      ranking_fn: Callable[[np.ndarray], np.ndarray] = _pareto_rank,
      eviction_limit: Optional[int] = None,
      adaptation: Optional[Mutation[Population, Offspring]] = None,
      adaptation_callable: Optional[
          Callable[[int], Mutation[Population, Offspring]]
      ] = None,
      metadata_namespace: str = 'nsga2',
      seed: Optional[int] = None
  ):
    """Creates NSGA2 Designer.

    Args:
      problem:
      population_size: Survival steps reduce the population to this size.
      first_survival_after: Apply the survival step after observing this many
        trials. Leave it unset to use the default behavior.
      ranking_fn: Takes (number of population) x (number of metrics) array of
        floating numbers and returns (number of population) array of integers,
        representing the pareto rank aka number of points it is dominated by.
        The default implementation is reasonably fast for hundreds of trials,
        but if you want to improve performance, your own implementation can be
        injected.
      eviction_limit: Evict a gene that has been alive for this many
        generations.
      adaptation: Fixed mutation used to evolve population.
      adaptation_callable: If specified, adaptation callable is a function of
        num_trials that returns the trial-dependent adaptation.
      metadata_namespace: Metadata namespace to use.
      seed: Random seed.

    Returns:
      NSGA2 Designer.
    """
    super().__init__(
        numpy_populations.PopulationConverter(
            problem.search_space,
            problem.metric_information,
            metadata_ns=metadata_namespace,
        ),
        numpy_populations.UniformRandomSampler(problem.search_space, seed=seed),
        NSGA2Survival(
            population_size,
            ranking_fn=ranking_fn,
            eviction_limit=eviction_limit,
        ),
        adaptation=adaptation
        or numpy_populations.LinfMutation(seed=seed, norm=0.001),
        first_survival_after=first_survival_after,
        adaptation_callable=adaptation_callable,
        population_size=population_size,
    )
