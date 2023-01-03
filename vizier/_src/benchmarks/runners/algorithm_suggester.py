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

"""Stateful algorithm suggesters to be used inside a runner.

Definitions:
------------
- A 'suggester' generates suggestions and update its state based on the provided
suggestion results.

- A 'runner' uses a suggester to simulate a study. It calls the suggester
multiple times, evaluate the suggestions and report back the results.
"""

import abc
from typing import Collection, Optional

import attr
from vizier import algorithms as vza
from vizier import pythia
from vizier import pyvizier as vz


class AlgorithmSuggester(abc.ABC):
  """Abstraction for core suggestion algorithm.

  All runner methods will be extended from the following template:

  def run(state: BenchmarkState):
    for _ in range(MAX_NUM_TRIALS):
      trials = state.alg.suggest(...)
      evaluate_and_complete_trials(trials)
      state.alg.post_completion_callback(trials)
  """

  @abc.abstractmethod
  def suggest(self, batch_size: Optional[int]) -> Collection[vz.Trial]:
    """Method to generate suggestions and assign ids to them."""

  @abc.abstractmethod
  def post_completion_callback(self, completed: vza.CompletedTrials) -> None:
    """Callback after the suggestions are completed, if needed."""

  @property
  @abc.abstractmethod
  def supporter(self) -> pythia.InRamPolicySupporter:
    """Returns the InRamPolicySupporter, which acts as a local client."""


@attr.define
class DesignerSuggester(AlgorithmSuggester):
  """Wraps a Designer into a AlgorithmSuggester.

  Designers return Suggestions which don't have ids assigned to them. We use
  InRamPolicySupporter to manage Trial ids.
  """
  _designer: vza.Designer = attr.field()
  _local_supporter: pythia.InRamPolicySupporter = attr.field()

  def suggest(self, batch_size: Optional[int]) -> Collection[vz.Trial]:
    suggestions: Collection[vz.TrialSuggestion] = self._designer.suggest(
        batch_size)
    # Assign ids.
    trials: Collection[vz.Trial] = self._local_supporter.AddSuggestions(
        suggestions)
    return trials

  def post_completion_callback(self, completed: vza.CompletedTrials) -> None:
    return self._designer.update(completed)

  @property
  def supporter(self) -> pythia.InRamPolicySupporter:
    return self._local_supporter


@attr.define
class PolicySuggester(AlgorithmSuggester):
  """Wraps a Policy into a AlgorithmSuggester."""
  _policy: pythia.Policy = attr.field()
  _local_supporter: pythia.InRamPolicySupporter = attr.field()

  def suggest(self, batch_size: Optional[int]) -> Collection[vz.Trial]:
    return self._local_supporter.SuggestTrials(
        self._policy, count=batch_size or 1)

  def post_completion_callback(self, completed: vza.CompletedTrials) -> None:
    # Nothing to do.
    pass

  @property
  def supporter(self) -> pythia.InRamPolicySupporter:
    return self._local_supporter
