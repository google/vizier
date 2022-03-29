"""Abstractions."""

import abc
from typing import Generic, Optional, Sequence, TypeVar
from vizier import pyvizier as vz
from vizier._src.algorithms.core import deltas
from vizier.interfaces import serializable

_T = TypeVar('_T')


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


class _StatefulSuggestionAlgorithm(Generic[_T], _SuggestionAlgorithm):
  """Stateful Suggestion algorithm.

  The generic type _T represents the delta that the algorithm understands.
  """

  @abc.abstractmethod
  def update(self, delta: _T) -> None:
    pass


class _PartiallySerializableAlgorithm(_StatefulSuggestionAlgorithm[_T],
                                      serializable.PartiallySerializable):
  pass


class _SerializableAlgorithm(_StatefulSuggestionAlgorithm[_T],
                             serializable.Serializable):
  pass


# Samplers only have suggest().
Sampler = _SuggestionAlgorithm

# Designers are applicable to settings where trials are
# sequentially completed and never removed.
Designer = _StatefulSuggestionAlgorithm[deltas.CompletedTrials]

PartiallySerializableDesigner = _PartiallySerializableAlgorithm[
    deltas.CompletedTrials]

SerializableDesigner = _SerializableAlgorithm[deltas.CompletedTrials]
