# Copyright 2023 Google LLC.
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

  Designer is the recommended interface for implementing commonly used
  algorithms such as GP-UCB and evolutionary algorithms. A Designer can be
  wrapped into a pythia `Policy` via `DesignerPolicy`. When run inside a service
  binary, a Designer instance does not persist during the lifetime of a `Study`.
  It receives all trials from the beginning of the study in `update()` calls.
  This can be inefficient.

  If your Designer can benefit from a persistent state, implement
  `(Partially)SerializableDesigner` interface and use
  `(Partially)SerializableDesignerPolicy` to wrap it.
  Vizier service will serialize the Designer's state in DB, restore it for
  the next usage, and update it with the newly completed trials since the last
  suggestion.

  > IMPORTANT: If your Designer changes its state inside `suggest()` (e.g. to
  > incorporate its own suggestions before completion), then use
  > (Partially)SerializableDesigner interface instead.
  """

  @abc.abstractmethod
  def update(self, delta: CompletedTrials) -> None:
    """Incorporates newly completed trials into the designer's state."""
    pass


class PartiallySerializableDesigner(Designer,
                                    serializable.PartiallySerializable):
  pass


class SerializableDesigner(Designer, serializable.Serializable):
  pass
