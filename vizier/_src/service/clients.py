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

"""OSS Vizier client."""

# TODO: Raise vizier-specific exceptions.

from typing import Any, Callable, Iterable, Iterator, List, Mapping, Optional, Type
import attr

from vizier._src.service import constants
from vizier._src.service import resources
from vizier._src.service import vizier_client
from vizier.client import client_abc
from vizier.service import pyvizier as vz

# Redeclared so users do not have to also import client_abc and clients.py.
ResourceNotFoundError = client_abc.ResourceNotFoundError

# Redeclared since clients.py is the user-facing client API.
environment_variables = vizier_client.environment_variables


@attr.define
class Trial(client_abc.TrialInterface):
  """Trial class.

  This class owns a Vizier client of the Study that contains the Trial that
  it is associated with.
  """

  _client: vizier_client.VizierClient = attr.field()
  _id: int = attr.field(validator=attr.validators.instance_of(int))

  @property
  def id(self) -> int:
    return self._id

  @property
  def parameters(self) -> Mapping[str, Any]:
    trial = self.materialize(include_all_measurements=False)
    study_config = self._client.get_study_config()
    return study_config.trial_parameters(vz.TrialConverter.to_proto(trial))

  def delete(self) -> None:
    self._client.delete_trial(self._id)

  def update_metadata(self, delta: vz.Metadata) -> None:
    actual_delta = vz.MetadataDelta(on_trials={self._id: delta})
    self._client.update_metadata(actual_delta)

  def complete(
      self,
      measurement: Optional[vz.Measurement] = None,
      *,
      infeasible_reason: Optional[str] = None,
  ) -> Optional[vz.Measurement]:
    self._trial = self._client.complete_trial(
        self._id, measurement, infeasible_reason
    )
    return self._trial.final_measurement

  def check_early_stopping(self) -> bool:
    return self._client.should_trial_stop(self._id)

  def stop(self) -> None:
    return self._client.stop_trial(self._id)

  def add_measurement(self, measurement: vz.Measurement) -> None:
    self._client.report_intermediate_objective_value(
        int(measurement.steps),
        measurement.elapsed_secs,
        [{k: v.value for k, v in measurement.metrics.items()}],
        trial_id=self._id,
    )

  def materialize(
      self,
      *,
      include_all_measurements: bool = True,
  ) -> vz.Trial:
    trial = self._client.get_trial(self._id)
    if not include_all_measurements:
      trial.measurements.clear()
    return trial

  @property
  def study(self) -> 'Study':
    return Study(self._client)


@attr.define
class TrialIterable(client_abc.TrialIterable):
  """Holds a collection of materialized Trials.

  See the parent class for full pydocs.
  """

  _iterable_factory: Callable[[], Iterable[vz.Trial]] = attr.field()
  _client: vizier_client.VizierClient = attr.field()

  def __iter__(self) -> Iterator[Trial]:
    for trial in self._iterable_factory():
      yield Trial(self._client, trial.id)

  def get(self) -> Iterator[vz.Trial]:
    for trial in self._iterable_factory():
      yield trial


@attr.define
class Study(client_abc.StudyInterface):
  """Responsible for study-level operations."""

  _client: vizier_client.VizierClient = attr.field()

  @property
  def resource_name(self) -> str:
    return self._client.study_resource_name

  def _trial_client(self, trial: vz.Trial) -> Trial:
    """Returns the client for the vz.Trial object."""
    return Trial(self._client, trial.id)

  def suggest(
      self, *, count: Optional[int] = None, client_id: str = 'default_client_id'
  ) -> List[Trial]:
    return [
        self._trial_client(t)
        for t in self._client.get_suggestions(
            count, client_id_override=client_id
        )
    ]

  def delete(self) -> None:
    self._client.delete_study()

  def update_metadata(self, delta: vz.Metadata) -> None:
    actual_delta = vz.MetadataDelta(on_study=delta)
    self._client.update_metadata(actual_delta)

  def _add_trial(self, trial: vz.Trial) -> Trial:
    return self._trial_client(self._client.add_trial(trial))

  def request(self, suggestion: vz.TrialSuggestion) -> Trial:
    trial = suggestion.to_trial()
    trial.is_requested = True
    return self._trial_client(self._client.add_trial(trial))

  def trials(
      self, trial_filter: Optional[vz.TrialFilter] = None
  ) -> TrialIterable:
    all_trials = self._client.list_trials()
    trial_filter = trial_filter or vz.TrialFilter()

    def iterable_factory():
      for t in filter(trial_filter, all_trials):
        yield t

    return TrialIterable(iterable_factory, self._client)

  def get_trial(self, trial_id: int) -> Trial:
    try:
      # Check if the trial actually exists.
      trial = self._client.get_trial(trial_id)
      return self._trial_client(trial)
    except KeyError as err:
      raise ResourceNotFoundError(
          f'Study f{self.resource_name} does not have Trial {trial_id}.'
      ) from err

  def optimal_trials(self, count: Optional[int] = None) -> TrialIterable:
    if count is None:
      trials = self._client.list_optimal_trials()
    else:
      raise ValueError('Count not supported.')
    return TrialIterable(lambda: trials, self._client)

  def materialize_problem_statement(self) -> vz.ProblemStatement:
    """Returns a cross-platform compatible minimal StudyConfig."""
    return self._client.get_study_config().to_problem()

  def materialize_study_config(self) -> vz.StudyConfig:
    """Returns a fully specific StudyConfig."""
    return self._client.get_study_config(self.resource_name)

  def set_state(self, state: vz.StudyState) -> None:
    self._client.set_study_state(state)

  def materialize_state(self) -> vz.StudyState:
    return self._client.get_study_state()

  @classmethod
  def from_resource_name(cls: Type['Study'], name: str) -> 'Study':
    client = vizier_client.VizierClient(
        name,
        constants.UNUSED_CLIENT_ID,
    )
    try:
      _ = client.get_study_config()  # Make sure study exists.
    except Exception as err:
      raise KeyError(f'Study {name} does not exist.') from err
    return Study(client)

  @classmethod
  def from_owner_and_id(
      cls: Type['Study'], owner: str, study_id: str
  ) -> 'Study':
    """Create study from `owner` and `study_id`.

    Args:
      owner: Owner of the study.
      study_id: Unique identifier within the same owner.

    Returns:
      Study.

    Raises:
      ResourceNotFoundError.
    """
    study_resource_name = resources.StudyResource(
        owner_id=owner, study_id=study_id
    ).name
    return cls.from_resource_name(study_resource_name)

  @classmethod
  def from_study_config(
      cls, config: vz.StudyConfig, *, owner: str, study_id: str
  ) -> 'Study':
    """Create study from StudyConfig.

    Args:
      config: OSS Study configuration. It is platform-specific. It is ignored if
        `owner` already has Study with `study_id`. # TODO: Instead
        of ignoring it, compare with the existing # study and return error if
        there's a change.
      owner: Owner of the study.
      study_id: Unique identifier within the same owner.

    Returns:
      Study.
    """
    # Use a dummy client_id, since for multi-worker workflows, each worker is
    # expected to provide a client_id in the suggest() call.
    return Study(
        vizier_client.create_or_load_study(
            owner_id=owner,
            client_id=constants.UNUSED_CLIENT_ID,
            study_id=study_id,
            study_config=config,
        )
    )
