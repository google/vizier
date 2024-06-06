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

"""Cross-platform Vizier client interfaces.

Code written using these interfaces are compatible with OSS and Cloud Vertex
Vizier. Note importantly that subclasses may have more methods than what is
required by interfaces, and such methods are not cross compatible. Our
recommendation is to use type annotatations of `StudyInterface` or
`TrialInterface` wherever applicable.

Keywords:

#Materialize: The method returns a deep copy of the underlying pyvizier object.
Modifying the returned object does not update the Vizier service.
"""

import abc
from typing import Any, Iterator, List, Mapping, Optional, Type, TypeVar

from vizier import pyvizier as vz

_T = TypeVar('_T')


# TODO: Add more errors, and unit tests to ensure that
# all vizier clients have the same error handling logic.
class ResourceNotFoundError(LookupError):
  """Error raised by Vizier clients when resource is not found."""

  pass


class TrialInterface(abc.ABC):
  """Responsible for trial-level operations."""

  @property
  @abc.abstractmethod
  def id(self) -> int:
    """Identifier of the trial within the study.

    Ids are sorted in increasing order of creation time.
    """

  @property
  @abc.abstractmethod
  def parameters(self) -> Mapping[str, Any]:
    """#Materializes the parameters of the trial.

    The parameters are parsed to the external types. The values in the returned
    dict can be a list of values, if the search space is configured with
    indices.

    As a result, TrialInterface.parameters can be totally different
    from trial.materialize().parameters.
    """

  @abc.abstractmethod
  def delete(self) -> None:
    """Delete the Trial in Vizier service.

    There is currently no promise on how this object behaves after `delete()`.
    If you are sharing a Trial object in parallel processes, proceed with
    caution.
    """

  @abc.abstractmethod
  def update_metadata(self, delta: vz.Metadata) -> None:
    """Updates Trial metadata.

    New keys will be appended, while old keys will be
    updated with new values.

    Args:
      delta: Change in Metadata from the original.
    """

  @abc.abstractmethod
  def complete(
      self,
      measurement: Optional[vz.Measurement] = None,
      *,
      infeasible_reason: Optional[str] = None,
  ) -> Optional[vz.Measurement]:
    """Completes the trial and #materializes the measurement.

    * If `measurement` is provided, then Vizier writes it as the trial's final
    measurement and returns it.
    * If `infeasible_reason` is provided, `measurement` is not required but
    can still be specified.
    * If neither is provided, then Vizier selects an existing (intermediate)
    measurement to be the final measurement and returns it.

    Args:
      measurement: Final measurement.
      infeasible_reason: Infeasible reason for missing final measurement.

    Returns:
      The final measurement of the trial, or None if the trial is marked
      infeasible.

    Raises:
      ValueError: If neither `measurement` nor `infeasible_reason` is provided
        but the trial does not contain any intermediate measurements.
    """

  @abc.abstractmethod
  def check_early_stopping(self) -> bool:
    """Decide if the trial should stop.

    If the Trial is in ACTIVE state and an early stopping algorithm is
    configured fior the Study, the stopping algorithm makes a stopping decision.
    If the algorithm decides that the trial is not worth continuing, Vizier
    service moves it into `STOPPING` state.

    Then, this method returns True if the Trial is in `STOPPING` state.

    Returns:
      True if trial is already in STOPPING state or entered STOPPING as a
      result of this method invocation.
    """
    pass

  @abc.abstractmethod
  def stop(self) -> None:
    """Asks to change the trial status to STOPPING.

    Should be called only if the trial status is ACTIVE. If the trial is
    STOPPING or COMPLETED, this is a no-op.
    """
    pass

  @abc.abstractmethod
  def add_measurement(self, measurement: vz.Measurement) -> None:
    """Adds an intermediate measurement."""

  @abc.abstractmethod
  def materialize(self, *, include_all_measurements: bool = True) -> vz.Trial:
    """#Materializes the Trial.

    Args:
      include_all_measurements: If True, returned Trial includes all
        intermediate measurements. The `final_measurement` is always included if
        one exists.

    Returns:
      Trial object.
    """

  @property
  @abc.abstractmethod
  def study(self) -> 'StudyInterface':
    """Returns the Study that this Trial belongs to."""


class TrialIterable(abc.ABC):
  """Allows iterating through both `TrialInterface` and `vz.Trial`.

  TrialIterable satisfies Iterable[TrialInterface] Protocol which is not
  explicitly inherited (See https://peps.python.org/pep-0544/). In addition,
  it guarantees that `list(StudyInterface.trials().get())` is at least as fast
  as `[t.materialize() for t in StudyInterface.trials()]`.

  StudyInterface returns this object when trials are already materialized
  while processing a request. A typical implementation of TrialIterable.get()
  uses a generator of the materialized trials.
  """

  @abc.abstractmethod
  def __iter__(self) -> Iterator[TrialInterface]:
    """Returns an iterator of TrialInterfaces, which are clients."""

  @abc.abstractmethod
  def get(self) -> Iterator[vz.Trial]:
    """Returns Trial objects, which are local objects."""


class StudyInterface(abc.ABC):
  """Responsible for study-level operations."""

  @property
  @abc.abstractmethod
  def resource_name(self) -> str:
    """Globally unique identifier of the study in Vizier service."""

  @abc.abstractmethod
  def suggest(
      self, *, count: Optional[int] = None, client_id: str = 'default_client_id'
  ) -> List[TrialInterface]:
    """Returns Trials to be evaluated by client_id.

    Args:
      count: Number of suggestions.
      client_id: When new Trials are generated, their `client_id` field is
        populated with this client_id. The Vizier service first looks for
        existing ACTIVE Trials that are assigned to `client_id`, before
        generating new ones.

    Returns:
      Trials.
    """

  # TODO: Request does not play well with boolean or discrete
  # integer parameters.
  @abc.abstractmethod
  def request(self, suggestion: vz.TrialSuggestion) -> TrialInterface:
    """Request a trial to be suggested in the future.

    Requested trials are "queued" in the vizier database. Next time a client
    requests for a new suggestion, it receives requested trials before any
    algorithm-generated suggestions.

    Args:
      suggestion: Suggestion to be requested.
    """

  @abc.abstractmethod
  def delete(self) -> None:
    """Delete the Study in Vizier service.

    There is currently no promise on how this object behaves after `delete()`.
    If you are sharing a Study object in parallel processes, proceed with
    caution.
    """

  @abc.abstractmethod
  def update_metadata(self, delta: vz.Metadata) -> None:
    """Updates StudyConfig metadata.

    New keys will be appended, while old keys will be
    updated with new values.

    Args:
      delta: Change in Metadata from the original.
    """

  # TODO: Make this method public.
  @abc.abstractmethod
  def _add_trial(self, trial: vz.Trial) -> TrialInterface:
    """Adds a trial to the Study. For testing only."""

  @abc.abstractmethod
  def trials(
      self, trial_filter: Optional[vz.TrialFilter] = None
  ) -> TrialIterable:
    """Fetches a collection of trials. Default uses vz.TrialFilter()."""

  @abc.abstractmethod
  def get_trial(self, uid: int) -> TrialInterface:
    """Fetches a single trial.

    Args:
      uid: Unique identifier of the trial within study.

    Returns:
      Trial.

    Raises:
      ResourceNotFoundError: If trial does not exist.
    """

  @abc.abstractmethod
  def optimal_trials(self, *, count: Optional[int] = None) -> TrialIterable:
    """Returns (pareto) optimal trial(s). Can be multiple Trial(s).

    Args:
      count: If provided, returns that many best Trials, breaking ties by
        earlier Trial id.
    """

  @abc.abstractmethod
  def materialize_problem_statement(self) -> vz.ProblemStatement:
    """#Materializes the problem statement."""

  @classmethod
  @abc.abstractmethod
  # TODO: Make the function have position only parameter
  # once cloud vizier support python 3.8+ version.
  def from_resource_name(cls: Type[_T], name: str) -> _T:
    """Fetches an existing study from the Vizier service.

    Args:
      name: Globally unique identifier of the study.

    Returns:
      Study.

    Raises:
      ResourceNotFoundError: If study does not exist.
    """

  @abc.abstractmethod
  def set_state(self, state: vz.StudyState) -> None:
    """Sets the state of the study.

    Args:
      state: New state of the study.
    """

  # TODO: Make this method purely abstract.
  def materialize_state(self) -> vz.StudyState:
    """#Materializes the study state."""
    raise NotImplementedError(
        f'materialize_state is not implemented in {type(self)}!'
    )
