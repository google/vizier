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

from typing import Callable, Optional, Sequence, Tuple

from absl import logging
import attr
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.evolution import numpy_populations
from vizier._src.algorithms.evolution import templates

Population = numpy_populations.Population
Offspring = numpy_populations.Offspring
Mutation = templates.Mutation


def pareto_rank(ys: np.ndarray) -> np.ndarray:
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


def crowding_distance(
    ys: np.ndarray, *, extra_tiebreakers: Sequence[np.ndarray] = tuple()
) -> np.ndarray:
  """Crowding distance.

  Reference:
  https://medium.com/@rossleecooloh/optimization-algorithm-nsga-ii-and-python-package-deap-fca0be6b2ffc
  except that the lower boundary does not get infinity crowding score.

  Args:
    ys: (number of population) x (number of metrics) array.
    extra_tiebreakers: A sequence of (number of population) array of floating
      numbers. If specified, they are used to break ties when sorting the
      population. By default, a random number is always used to break ties
      consistently across all metrics.

  Returns:
    (number of population) float32 array. Higher numbers mean less crowding
    and more desirable.
  """
  scores = np.zeros([ys.shape[0]], dtype=np.float32)

  if ys.shape[0] <= 1:
    return scores

  rng = np.random.default_rng()
  # Use a random number to break ties. But the same random number is used for
  # all metrics.
  random_tiebreaker = rng.random(ys.shape[0])
  tiebreakers = list(extra_tiebreakers) + [random_tiebreaker]

  for m in range(ys.shape[1]):
    # Sort by the m-th metric.
    sid = sorted(
        np.arange(ys.shape[0]),
        key=lambda i, m=m: (ys[i, m],)
        + tuple(tiebreaker[i] for tiebreaker in tiebreakers),
    )

    # Compute the range of the m-th metric.
    yy = ys[:, m]  # Shape: (num_population,)
    yrange = yy[sid[-1]] - yy[sid[0]] + np.finfo(np.float32).eps

    # Lower boundary is assigned a one-sided score and does not automatically
    # get infinity. This is different from the paper. The lower boundary means
    # it's dominated by all other points in one dimension. There's no reason to
    # favor it over other points.
    scores[sid[0]] += (yy[sid[1]] - yy[sid[0]]) / yrange
    # Upper boundary is assigned infinity. This point will survive anyways
    # because it's pareto-optimal. But in case there are ties, it's useful to
    # make only one of them stand out.
    scores[sid[-1]] += np.inf

    scores[sid[1:-1]] += (yy[sid[2:]] - yy[sid[:-2]]) / yrange
  # Normalize the score to [0, 1].
  return scores / ys.shape[1]


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
      ranking_fn: Callable[[np.ndarray], np.ndarray] = pareto_rank,
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

    logging.info('Selecting %s from %s', self._target_size, len(population))
    selected = population.empty_like()
    # Sort by the safety constraint.
    if selected.cs.shape[1]:
      top, border = _select_by(
          _constraint_violation(population.cs), target=self._target_size
      )
      selected += population[top]
      population = population[border]
      logging.info(
          'Selected %s by safety constraints, and will break ties among: %s',
          len(selected),
          len(population),
      )
    # Sort by the pareto rank.
    pareto_ranks = self._ranking_fn(population.ys)
    top, border = _select_by(
        pareto_ranks, target=self._target_size - len(selected)
    )
    considered_pareto_ranks = np.concatenate(
        [-np.ones(len(selected)), pareto_ranks[top], pareto_ranks[border]]
    )
    selected += population[top]
    population = population[border]
    logging.info(
        'Selected %s by pareto rank, and will break ties among: %s',
        len(selected),
        len(population),
    )
    # Sort by the distance. Include the points that are already selected for
    # the computation.
    # Flip the sign so it works with ascending sort.
    distance = -crowding_distance(
        (selected + population).ys,
        extra_tiebreakers=[considered_pareto_ranks],
    )
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
      ranking_fn: Callable[[np.ndarray], np.ndarray] = pareto_rank,
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
