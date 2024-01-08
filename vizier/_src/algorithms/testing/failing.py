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

"""Failing designers used for testing."""

from typing import Optional, Sequence
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random


class FailedSuggestError(Exception):
  """Exception to raise during failing suggest call."""


class FailingDesigner(vza.Designer):
  """Failing Designer.

  The designer raises exception at every suggest call.
  """

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    pass

  def suggest(
      self, count: Optional[int] = None
  ) -> Sequence[vz.TrialSuggestion]:
    raise FailedSuggestError()


class AlternateFailingDesigner(vza.Designer):
  """Alternate Failing Designer.

  The designer raises exception at every second suggest call.

  Note: the designer doesn't persist a state across different instantiations and
  therefore should only be used during in memory run.
  """

  def __init__(self, search_space: vz.SearchSpace):
    self._suggest_count = 0
    self._random_designer = random.RandomDesigner(search_space)

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    pass

  def suggest(
      self, count: Optional[int] = None
  ) -> Sequence[vz.TrialSuggestion]:
    count = count or 1
    self._suggest_count += count
    if self._suggest_count % 2 == 0 or count > 1:
      raise FailedSuggestError()
    else:
      return self._random_designer.suggest(1)
