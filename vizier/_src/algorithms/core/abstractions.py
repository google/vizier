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
from typing import Optional, Sequence, TypeVar, Protocol

import attr
from vizier import pyvizier as vz
from vizier.interfaces import serializable

_T = TypeVar('_T')


@attr.define(frozen=True)
class CompletedTrials:
  """A group of completed trials.

  Attributes:
    trials: Completed Trials.
  """

  def __attrs_post_init__(self):
    for trial in self.trials:
      if trial.status != vz.TrialStatus.COMPLETED:
        raise ValueError(f'All trials must be completed. Bad trial:\n{trial}')

  trials: Sequence[vz.Trial] = attr.field(
      converter=tuple,
      validator=attr.validators.deep_iterable(
          attr.validators.instance_of(vz.Trial)
      ),
  )


@attr.define(frozen=True)
class ActiveTrials:
  """A group of active (a.k.a pending) trials.

  Attributes:
    trials: Active Trials.
  """

  def __attrs_post_init__(self):
    for trial in self.trials:
      if trial.status != vz.TrialStatus.ACTIVE:
        raise ValueError(f'All trials must be active. Bad trial:\n{trial}')

  trials: Sequence[vz.Trial] = attr.field(
      converter=tuple,
      default=attr.Factory(list),
      validator=attr.validators.deep_iterable(
          attr.validators.instance_of(vz.Trial)
      ),
  )


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
  wrapped into a pythia `Policy` via `DesignerPolicy` (stateless) or
  '(Partially)SerializableDesignerPolicy' (stateful).

  If your Designer is stateless it should match with 'DesignerPolicy' which
  is responsible for calling the 'update' method and pass all completed trials
  since the beginning of the trials as well as all currently active
  trials.

  If your Designer can benefit from a persistent state (stateful), implement
  `(Partially)SerializableDesigner` interface and use
  `(Partially)SerializableDesignerPolicy` to wrap it.
  Vizier service will serialize the Designer's state in DB, restore it for
  the next usage, and update it with the newly completed and active trials
  since the last suggestion.

  IMPORTANT: If your Designer changes its state inside `suggest()` (e.g. to
  incorporate its own suggestions before completion), then use
  (Partially)SerializableDesigner interface instead or take advantage of the
  'all_active' trials argument.

  Note that when run inside a service binary, a Designer instance does not
  persist during the lifetime of a `Study`.
  """

  @abc.abstractmethod
  def update(
      self, completed: CompletedTrials, all_active: ActiveTrials
  ) -> None:
    """Incorporates completed and active trials into the designer's state.

    In production, 'completed' refer to all the completed trials in the study.
    In benchmarking, 'completed' refer to newly created trials that the designer
      hasn't seen yet ("delta").

    In both production and benchmarking, 'all_active' refers to ALL the current
    ACTIVE trials.

    Arguments:
      completed: COMPLETED trials.
      all_active: ACTIVE (aka PENDING) trials.
    """
    pass


class DesignerFactory(Protocol[_T]):
  """Protocol (PEP-544) for a designer factory."""

  def __call__(self, problem: vz.ProblemStatement, **kwargs) -> _T:
    pass


class PartiallySerializableDesigner(
    Designer, serializable.PartiallySerializable
):
  pass


class SerializableDesigner(Designer, serializable.Serializable):
  pass
