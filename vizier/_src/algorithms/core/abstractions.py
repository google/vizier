"""Abstractions."""

import abc
from typing import Optional, Sequence, TypeVar

import attr
from vizier import pyvizier as vz
from vizier.interfaces import serializable

_T = TypeVar('_T')


@attr.define(frozen=True)
class CompletedTrials:
  """A group of completed trials.

  Attributes:
    completed: Completed Trials.
  """

  def __attrs_post_init__(self):
    for trial in self.completed:
      if trial.status != vz.TrialStatus.COMPLETED:
        raise ValueError(f'All trials must be completed. Bad trial:\n{trial}')

  completed: Sequence[vz.Trial] = attr.field(
      converter=tuple,
      validator=attr.validators.deep_iterable(
          attr.validators.instance_of(vz.Trial)))


class _SuggestionAlgorithm(abc.ABC):
  """Suggestion algorithm."""

  @abc.abstractmethod
  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    """Make new suggestions.

    Args:
      count: Makes best effort to generate this many suggestions. If None,
        suggests as many as the algorithm wants.

    Returns:
      New suggestions.
    """
    pass


class Designer(_SuggestionAlgorithm):
  """Suggestion algorithm for sequential usage.

  `Designer` is the recommended interface for implementing commonly used
  algorithms such as GP-UCB and evolutionary algorithms. A `Designer` can be
  wrapped into a pythia `Policy` via `DesignerPolicy`. When run inside a service
  binary, `Designer` instance is not guaranteed to persist during
  the lifetime of a `Study`, and should be assumed to receive all trials from
  the beginning of the study in `update()` calls.

  If your `Designer` can benefit from a persistent state, implement a
  either `SerializableDesigner` or `PartiallySerializableDesigner`, and use
  `SerializableDesignerPolicy` or `PartiallySerializableDesignerPolicy`,
  respectively.
  """

  @abc.abstractmethod
  def update(self, delta: CompletedTrials) -> None:
    """Reflect the delta in the designer's state."""
    pass


class PartiallySerializableDesigner(Designer,
                                    serializable.PartiallySerializable):
  pass


class SerializableDesigner(Designer, serializable.Serializable):
  pass
