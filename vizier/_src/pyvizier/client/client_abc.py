"""Cross-platform Vizier client interfaces.

Aside from "materialize_" methods, code written using these interfaces are
compatible with OSS and Cloud Vertex Vizier. Note importantly that subclasses
may have more methods than what is required by interfaces, and such methods
are not cross compatible. Our recommendation is to explicitly type your objects
to be `StudyInterface` or `TrialInterface` when you want to guarantee that
a code block is cross-platform.

Keywords:

#Materialize: The method returns a deep copy of the underlying pyvizier object.
Modifying the returned object does not update the Vizier service.
"""

# TODO: Add a dedicated .md file with more code examples.

import abc

from typing import Iterable, Optional, Collection, Type, TypeVar, Mapping, Any

from vizier import pyvizier as vz

_T = TypeVar('_T')


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
    """#Materializes the parameters of the trial."""

  @abc.abstractmethod
  def delete(self) -> None:
    """Delete the Trial in Vizier service.

    There is currently no promise on how this object behaves after `delete()`.
    If you are sharing a Trial object in parallel processes, proceed with
    caution.
    """

  @abc.abstractmethod
  def complete(
      self,
      measurement: Optional[vz.Measurement] = None,
      *,
      infeasible_reason: Optional[str] = None) -> Optional[vz.Measurement]:
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


class StudyInterface(abc.ABC):
  """Responsible for study-level operations."""

  @property
  @abc.abstractmethod
  def resource_name(self) -> str:
    """Globally unique identifier of the study in Vizier service."""

  @abc.abstractmethod
  def suggest(
      self,
      *,
      count: Optional[int] = None,
      client_id: str = 'default_client_id') -> Collection[TrialInterface]:
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

  @abc.abstractmethod
  def delete(self) -> None:
    """Delete the Study in Vizier service.

    There is currently no promise on how this object behaves after `delete()`.
    If you are sharing a Study object in parallel processes, proceed with
    caution.
    """

  # TODO: Make this method public.
  @abc.abstractmethod
  def _add_trial(self, trial: vz.Trial) -> TrialInterface:
    """Adds a trial to the Study. For testing only."""

  @abc.abstractmethod
  def trials(
      self,
      trial_filter: Optional[vz.TrialFilter] = None
  ) -> Iterable[TrialInterface]:
    """Fetches a collection of trials."""

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
  def optimal_trials(self) -> Iterable[TrialInterface]:
    """Returns optimal trial(s)."""

  @abc.abstractmethod
  def materialize_problem_statement(self) -> vz.ProblemStatement:
    """#Materializes the problem statement."""

  @classmethod
  @abc.abstractmethod
  def from_resource_name(cls: Type[_T], name: str, /) -> _T:
    """Fetches an existing study from the Vizier service.

    Args:
      name: Globally unique identifier of the study.

    Returns:
      Study.

    Raises:
      ResourceNotFoundError: If study does not exist.
    """
