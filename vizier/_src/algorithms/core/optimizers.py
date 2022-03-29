"""Base class for acquisition optimizers."""

import abc
import dataclasses
from typing import Any, Callable, List, Optional

import attr
from vizier import pyvizier as vz
from vizier.pyvizier import converters

Array = Any


def _is_positive(instance, attribute, value):
  del instance, attribute
  if value < 0:
    raise ValueError(f'value must be positive! given: {value}')


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
  def select_branches(self, num_suggestions: int) -> List[BranchSelection]:
    pass


class GradientFreeMaximizer(abc.ABC):
  """Optimizes a function on Vizier search space.

  Typically used for optimizing acquisition functions.
  """

  @abc.abstractmethod
  def maximize(self,
               score_fn: Callable[[Array], Array],
               converter: converters.DefaultTrialConverter,
               search_space: vz.SearchSpace,
               *,
               count: int = 1,
               budget_factor: float = 1.0,
               **kwargs) -> List[vz.Trial]:
    """Maximize a function.

    Args:
      score_fn: A function that takes as input `converter.to_features()` and
        outputs a numpy array-like object of shape (B,).
      converter: Responsible for converting between `Trial`s (which is the
        return type) and `score_fn`'s desired input type.
      search_space: Returned Trials must be contained in this search space.
      count: Optimizer tries to return this many trials.
      budget_factor: Every optimizer has a rough notion of "standard" budget.
        Use this much fraction of the standard budget for the call.
      **kwargs: For experimental keyword arguments.

    Returns:
      Trials, of length less than or equal to max_num_suggestions.
    """
    pass


@dataclasses.dataclass(frozen=True)
class BranchThenMaximizer(GradientFreeMaximizer):
  """Maximizes a function by first choosing a branch and then apply maximizer.

  Attributes:
    branch_selector: Selects all conditional parent values
    maximizer_factory: Creates an optimizer in the flat (non-conditional) after
      branch selector fixing all conditional parent values.
    max_num_suggestions_per_branch: Limits the number of suggestions per branch.
      This is useful when acquisition function isn't well-suited for batch
      suggestions.
  """
  branch_selector: BranchSelector
  maximizer_factory: Callable[[], GradientFreeMaximizer]
  max_num_suggestions_per_branch: Optional[int] = None

  def _num_suggestions_for_branch(self, branch: BranchSelection) -> int:
    if self.max_num_suggestions_per_branch is None:
      return branch.num_suggestions
    else:
      return min(self.max_num_suggestions_per_branch, branch.num_suggestions)

  def maximize(self,
               score_fn: Callable[[Array], Array],
               converter: converters.DefaultTrialConverter,
               search_space: vz.SearchSpace,
               *,
               count: int = 1,
               budget_factor: float = 1.0,
               **kwargs) -> List[vz.Trial]:
    # If there are conditional branches, use Vizier's default branch
    # selection mechanism.
    branches = self.branch_selector.select_branches(count)
    suggestions = []
    maximizer = self.maximizer_factory()
    for branch in branches:
      suggestions.extend(
          maximizer.maximize(
              score_fn,
              converter,
              branch.search_space,
              count=self._num_suggestions_for_branch(branch),
              budget_factor=budget_factor * (branch.num_suggestions / count)))
    return suggestions
