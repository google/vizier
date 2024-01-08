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

"""Base class for acquisition optimizers."""

import abc
from typing import Any, Callable, Optional, Protocol, Sequence

import attr
from vizier import pyvizier as vz

Array = Any


def _is_positive(instance, attribute, value):
  del instance, attribute
  if value < 0:
    raise ValueError(f'value must be positive! given: {value}')


class BatchTrialScoreFunction(Protocol):
  """Protocol (https://peps.python.org/pep-0544/) for scoring trials."""

  # TODO: Decide what to do with NaNs.
  def __call__(self, trials: Sequence[vz.Trial]) -> dict[str, Array]:
    """Evaluates the trials.

    Args:
      trials: A sequence of N trials

    Returns:
      A dict of shape (N, 1) arrays.
    """


@attr.s(frozen=False, init=True, slots=True)
class BranchSelection:
  """Branch Selection.

  Decision of a branch selector that contains the flat subspace and number
  of suggestions to be generated in that subspace.

  Instead of creating N suggestions on a conditional space S, we create
  N_1, ..., N_k suggestions on flat spaces S_1, .., S_k such that
  N_1 + ... + N_k = N and S_1, ..., S_k are contained in S. Each BranchSelection
  object represents (N_i, S_i) tuple.

  Attributres:
    search_space: Search space that does not contain any conditional parameters.
    num_suggestions: Number of suggestions to be made in the search space.
  """
  search_space: vz.SearchSpace = attr.ib(
      validator=attr.validators.instance_of(vz.SearchSpace),
      on_setattr=attr.setters.validate)
  num_suggestions: int = attr.ib(
      validator=[attr.validators.instance_of(int), _is_positive],
      on_setattr=attr.setters.validate)


class BranchSelector(abc.ABC):

  @abc.abstractmethod
  def select_branches(self, num_suggestions: int) -> list[BranchSelection]:
    pass


class GradientFreeOptimizer(abc.ABC):
  """Optimizes a function on Vizier search space.

  Typically used for optimizing acquisition functions.
  """

  @abc.abstractmethod
  def optimize(
      self,
      score_fn: BatchTrialScoreFunction,
      problem: vz.ProblemStatement,
      *,
      count: int = 1,
      budget_factor: float = 1.0,
      seed_candidates: Sequence[vz.TrialSuggestion] = tuple()
  ) -> list[vz.Trial]:
    """Optimizes a function.

    Args:
      score_fn: Should return a dict whose keys contain the metric names of
        "problem.metric_information".
      problem:
      count: Optimizer tries to return this many trials.
      budget_factor:  For optimizers with a notion of a budget, use this much
        fraction of the standard budget for the call.
      seed_candidates: Seed suggestions to be used as initial batch for
        optimization.

    Returns:
      Trials, of length less than or equal to max_num_suggestions.
      Trials are COMPLETED with score_fn results.
    """
    pass


@attr.frozen
class BranchThenOptimizer(GradientFreeOptimizer):
  """Optimizes a function by first choosing a branch and then apply Optimizer.

  Attributes:
    _branch_selector: Selects all conditional parent values
    _optimizer_factory: Creates an optimizer in the flat (non-conditional) after
      branch selector fixing all conditional parent values.
    max_num_suggestions_per_branch: Limits the number of suggestions per branch.
      This is useful when acquisition function isn't well-suited for batch
      suggestions.
  """
  _branch_selector: BranchSelector
  _optimizer_factory: Callable[[], GradientFreeOptimizer]
  max_num_suggestions_per_branch: Optional[int] = None

  def _num_suggestions_for_branch(self, branch: BranchSelection) -> int:
    if self.max_num_suggestions_per_branch is None:
      return branch.num_suggestions
    else:
      return min(self.max_num_suggestions_per_branch, branch.num_suggestions)

  def optimize(
      self,
      score_fn: BatchTrialScoreFunction,
      problem: vz.ProblemStatement,
      *,
      count: int = 1,
      budget_factor: float = 1.0,
      seed_candidates: Sequence[vz.TrialSuggestion] = tuple(),
  ) -> list[vz.Trial]:
    # If there are conditional branches, use Vizier's default branch
    # selection mechanism.
    branches = self._branch_selector.select_branches(count)
    suggestions = []
    optimizer = self._optimizer_factory()
    for branch in branches:
      subproblem = attr.evolve(problem, search_space=branch.search_space)
      suggestions.extend(
          optimizer.optimize(
              score_fn,
              subproblem,
              count=self._num_suggestions_for_branch(branch),
              budget_factor=budget_factor * (branch.num_suggestions / count)))
    return suggestions
