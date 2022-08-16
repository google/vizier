"""Runner protocols encapsulate a stateful algorithm to be used in a runner."""

from typing import Collection, Optional, Protocol

import attr
from vizier import algorithms as vza
from vizier import pythia
from vizier import pyvizier as vz


class AlgorithmRunnerProtocol(Protocol):
  """Abstraction for core suggestion algorithm routines.

  All runner methods will be extended from the following template:

  def run(alg: AlgorithmRunnerProtocol):
    for _ in range(MAX_NUM_TRIALS):
      trials = alg.suggest(...)
      evaluate_and_complete_trials(trials)
      alg.post_completion_callback(trials)
  """

  def suggest(self, batch_size: Optional[int]) -> Collection[vz.Trial]:
    """Method to generate suggestions and assign ids to them."""

  def post_completion_callback(self, completed: vza.CompletedTrials) -> None:
    """Callback after the suggestions are completed, if needed."""

  @property
  def supporter(self) -> pythia.InRamPolicySupporter:
    """Returns the InRamPolicySupporter, which acts as a local client."""


@attr.define
class DesignerRunnerProtocol(AlgorithmRunnerProtocol):
  """Wraps a Designer into the RunnerProtocol.

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
class PolicyRunnerProtocol(AlgorithmRunnerProtocol):
  """Wraps a Policy into the RunnerProtocol."""
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
