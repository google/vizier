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

"""Makes client calls via gRPC to an existing Vizier Service Server.

This client can be used interchangeably with the Cloud Vizier client.
"""

import datetime
import functools
import time
from typing import Any, Dict, List, Mapping, Optional, Union

from absl import logging
import attr
import grpc

from vizier._src.service import constants
from vizier._src.service import resources
from vizier._src.service import stubs_util
from vizier._src.service import study_pb2
from vizier._src.service import types
from vizier._src.service import vizier_service_pb2
from vizier._src.service import vizier_service_pb2_grpc
from vizier.service import pyvizier
from vizier.utils import attrs_utils

from google.longrunning import operations_pb2
from google.protobuf import duration_pb2
from google.protobuf import json_format


@attr.define
class _EnvironmentVariables:
  """Global environment variables.

  Attributes:
    server_endpoint: Endpoint to the Vizier server.
    servicer_kwargs:
    new_suggestion_polling_secs: The period to wait between polling for the
      status of long-running SuggestOperations. Vizier may increase this period
      if multiple polls are needed. (You may use zero for interactive demos, but
      it is only appropriate for very small Studies.)
  """

  server_endpoint: str = attr.field(
      default=constants.NO_ENDPOINT, validator=attr.validators.instance_of(str)
  )
  servicer_kwargs: Dict[str, Any] = attr.field(factory=dict)

  # TODO: Add an e2e test for this.
  new_suggestion_polling_secs: float = attr.field(default=1.0)

  def servicer_use_sql_ram(self) -> None:
    """Should be used in tests to avoid filepath issues."""
    self.servicer_kwargs['database_url'] = constants.SQL_MEMORY_URL


environment_variables = _EnvironmentVariables()


@functools.lru_cache(maxsize=None)
def _create_local_vizier_servicer() -> (
    vizier_service_pb2_grpc.VizierServiceServicer
):
  from vizier._src.service import vizier_service  # pylint:disable=g-import-not-at-top

  return vizier_service.VizierServicer(**environment_variables.servicer_kwargs)


def create_vizier_servicer_or_stub() -> types.VizierService:
  endpoint = environment_variables.server_endpoint
  if endpoint == constants.NO_ENDPOINT:
    logging.info('No endpoint given; using cached local VizierServicer.')
    logging.warning('Python 3.8+ is required in this case.')
    return _create_local_vizier_servicer()
  return stubs_util.create_vizier_server_stub(endpoint)


@attr.frozen(init=True)
class VizierClient:
  """Client for communicating with the Vizier Service via GRPC.

  It can be initialized directly with a VizierService, or
  created from endpoint. See also `create_server_stub`.
  """

  _study_resource_name: str = attr.field(
      validator=attr.validators.instance_of(str)
  )
  _client_id: str = attr.field(
      validator=[attr.validators.instance_of(str), attrs_utils.assert_not_empty]
  )
  _service: types.VizierService = attr.field(
      repr=False, factory=create_vizier_servicer_or_stub
  )

  @property
  def _study_resource(self) -> resources.StudyResource:
    return resources.StudyResource.from_name(self._study_resource_name)

  @property
  def _owner_id(self) -> str:
    return self._study_resource.owner_id

  @property
  def _study_id(self) -> str:
    return self._study_resource.study_id

  @property
  def study_resource_name(self) -> str:
    return self._study_resource_name

  def get_suggestions(
      self, suggestion_count: int, *, client_id_override: Optional[str] = None
  ) -> List[pyvizier.Trial]:
    """Gets a list of suggested Trials.

    Args:
        suggestion_count: The number of suggestions to request.
        client_id_override: If set, overrides self._client_id for this call.

    Returns:
      A list of PyVizier Trials. This may be an empty list if:
      1. A finite search space has been exhausted.
      2. If max_num_trials has been reached.
      3. Or if there are no longer any trials that match a supplied Context.

    Raises:
        RuntimeError: Indicates that a suggestion was requested
            from an inactive study. Note that this is NOT raised when a
            finite Study runs out of suggestions. In such a case, an empty
            list is returned.
    """
    if client_id_override is not None:
      client_id = client_id_override
    else:
      client_id = self._client_id
    request = vizier_service_pb2.SuggestTrialsRequest(
        parent=resources.StudyResource(self._owner_id, self._study_id).name,
        suggestion_count=suggestion_count,
        client_id=client_id,
    )
    try:
      operation = self._service.SuggestTrials(request)
    except grpc.RpcError as rpc_error:
      # If ImmutableStudyError occurs, we simply return empty suggestion list.
      # Otherwise, halt the client and raise error.
      if rpc_error.code() == grpc.StatusCode.FAILED_PRECONDITION:  # pytype:disable=attribute-error
        return []
      raise rpc_error

    num_attempts = 0
    while not operation.done:
      sleep_time = PollingDelay(
          num_attempts, environment_variables.new_suggestion_polling_secs
      )
      num_attempts += 1
      logging.info(
          'Waiting for operation with name %s to be done', operation.name
      )
      time.sleep(sleep_time.total_seconds())

      operation = self._service.GetOperation(
          operations_pb2.GetOperationRequest(name=operation.name)
      )

    if operation.HasField('error'):
      error_message = 'SuggestOperation {} failed with message: {}'.format(
          operation.name, operation.error
      )
      logging.error(error_message)
      raise RuntimeError(error_message)
    # TODO: Replace with any.Unpack().
    trials = vizier_service_pb2.SuggestTrialsResponse.FromString(
        operation.response.value
    ).trials
    return pyvizier.TrialConverter.from_protos(trials)

  def report_intermediate_objective_value(
      self,
      step: int,
      elapsed_secs: float,
      metric_list: List[Mapping[str, Union[int, float]]],
      trial_id: int,
  ) -> pyvizier.Trial:
    """Sends intermediate objective value for the trial identified by trial_id."""
    new_metric_list = []
    for metric in metric_list:
      for metric_name in metric:
        metric_pb2 = study_pb2.Measurement.Metric(
            metric_id=metric_name, value=metric[metric_name]
        )
        new_metric_list.append(metric_pb2)

    integer_seconds = int(elapsed_secs)
    nano_seconds = int((elapsed_secs - integer_seconds) * 1e9)
    measurement = study_pb2.Measurement(
        elapsed_duration=duration_pb2.Duration(
            seconds=integer_seconds, nanos=nano_seconds
        ),
        step_count=step,
        metrics=new_metric_list,
    )
    request = vizier_service_pb2.AddTrialMeasurementRequest(
        trial_name=resources.TrialResource(
            self._owner_id, self._study_id, trial_id
        ).name,
        measurement=measurement,
    )
    trial = self._service.AddTrialMeasurement(request)
    return pyvizier.TrialConverter.from_proto(trial)

  def should_trial_stop(self, trial_id: int) -> bool:
    request = vizier_service_pb2.CheckTrialEarlyStoppingStateRequest(
        trial_name=resources.TrialResource(
            self._owner_id, self._study_id, trial_id
        ).name
    )
    early_stopping_response = self._service.CheckTrialEarlyStoppingState(
        request
    )
    return early_stopping_response.should_stop

  def stop_trial(self, trial_id: int) -> None:
    request = vizier_service_pb2.StopTrialRequest(
        name=resources.TrialResource(
            self._owner_id, self._study_id, trial_id
        ).name
    )
    self._service.StopTrial(request)
    logging.info('Trial with id %s stopped.', trial_id)

  def complete_trial(
      self,
      trial_id: int,
      final_measurement: Optional[pyvizier.Measurement] = None,
      infeasibility_reason: Optional[str] = None,
  ) -> pyvizier.Trial:
    """Completes the trial, which is infeasible if given a infeasibility_reason."""
    request = vizier_service_pb2.CompleteTrialRequest(
        name=resources.TrialResource(
            self._owner_id, self._study_id, trial_id
        ).name,
        trial_infeasible=infeasibility_reason is not None,
        infeasible_reason=infeasibility_reason,
    )

    if final_measurement is not None:
      # Final measurement can still be included even for infeasible trials, for
      # other metrics, or a subset of objective + safety metrics.
      request.final_measurement.CopyFrom(
          pyvizier.MeasurementConverter.to_proto(final_measurement)
      )

    trial = self._service.CompleteTrial(request)
    return pyvizier.TrialConverter.from_proto(trial)

  def get_trial(self, trial_id: int) -> pyvizier.Trial:
    """Return the Optimizer trial for the given trial_id."""
    request = vizier_service_pb2.GetTrialRequest(
        name=resources.TrialResource(
            self._owner_id, self._study_id, trial_id
        ).name
    )
    trial = self._service.GetTrial(request)
    return pyvizier.TrialConverter.from_proto(trial)

  def list_trials(self) -> List[pyvizier.Trial]:
    """List all trials."""
    parent = resources.StudyResource(self._owner_id, self._study_id).name
    request = vizier_service_pb2.ListTrialsRequest(parent=parent)
    response = self._service.ListTrials(request)
    return pyvizier.TrialConverter.from_protos(response.trials)

  def list_optimal_trials(self) -> List[pyvizier.Trial]:
    """List only the optimal completed trials."""
    parent = resources.StudyResource(self._owner_id, self._study_id).name
    request = vizier_service_pb2.ListOptimalTrialsRequest(parent=parent)
    response = self._service.ListOptimalTrials(request)
    return pyvizier.TrialConverter.from_protos(response.optimal_trials)

  def list_studies(self) -> List[Dict[str, Any]]:
    """List all studies for the given owner."""
    request = vizier_service_pb2.ListStudiesRequest(
        parent=resources.OwnerResource(self._owner_id).name
    )
    list_studies_response = self._service.ListStudies(request)
    # TODO: Use PyVizier StudyDescriptor instead.
    return [
        json_format.MessageToJson(study)
        for study in list_studies_response.studies
    ]

  def add_trial(self, trial: pyvizier.Trial) -> pyvizier.Trial:
    """Adds a trial.

    Args:
      trial:

    Returns:
      A new trial object from the database. It will have different timestamps,
      newly assigned id, and newly assigned state (either REQUESTED
      or COMPLETED).
    """
    request = vizier_service_pb2.CreateTrialRequest(
        parent=resources.StudyResource(self._owner_id, self._study_id).name,
        trial=pyvizier.TrialConverter.to_proto(trial),
    )
    trial_proto = self._service.CreateTrial(request)
    return pyvizier.TrialConverter.from_proto(trial_proto)

  def delete_trial(self, trial_id: int) -> None:
    """Deletes trial from datastore."""
    request_trial_name = resources.TrialResource(
        self._owner_id, self._study_id, trial_id
    ).name
    request = vizier_service_pb2.DeleteTrialRequest(name=request_trial_name)
    self._service.DeleteTrial(request)
    logging.info('Trial deleted: %s', trial_id)

  def delete_study(self, study_resource_name: Optional[str] = None) -> None:
    """Deletes study from datastore."""
    study_resource_name = study_resource_name or (
        resources.StudyResource(self._owner_id, self._study_id).name
    )
    request = vizier_service_pb2.DeleteStudyRequest(name=study_resource_name)
    self._service.DeleteStudy(request)
    logging.info('Study deleted: %s', study_resource_name)

  def get_study_config(
      self, study_resource_name: Optional[str] = None
  ) -> pyvizier.StudyConfig:
    """Returns the study config."""
    study_resource_name = study_resource_name or (
        resources.StudyResource(self._owner_id, self._study_id).name
    )
    request = vizier_service_pb2.GetStudyRequest(name=study_resource_name)
    study = self._service.GetStudy(request)
    return pyvizier.StudyConfig.from_proto(study.study_spec)

  def get_study_state(
      self, study_resource_name: Optional[str] = None
  ) -> pyvizier.StudyState:
    study_resource_name = study_resource_name or (
        resources.StudyResource(self._owner_id, self._study_id).name
    )
    request = vizier_service_pb2.GetStudyRequest(name=study_resource_name)
    study = self._service.GetStudy(request)
    return pyvizier.StudyStateConverter.from_proto(study.state)

  def set_study_state(
      self,
      state: pyvizier.StudyState,
      study_resource_name: Optional[str] = None,
  ) -> None:
    """Sets the study to a given state."""
    study_resource_name = study_resource_name or (
        resources.StudyResource(self._owner_id, self._study_id).name
    )
    proto_state = pyvizier.StudyStateConverter.to_proto(state)
    request = vizier_service_pb2.SetStudyStateRequest(
        parent=study_resource_name, state=proto_state
    )
    self._service.SetStudyState(request)
    logging.info(
        'Study with resource name %s set to %s state.',
        study_resource_name,
        state,
    )

  def update_metadata(
      self,
      delta: pyvizier.MetadataDelta,
      study_resource_name: Optional[str] = None,
  ) -> None:
    """Updates metadata.

    Args:
      delta: Batch updates to Metadata, consisting of numerous (namespace, key,
        value) objects.
      study_resource_name: An identifier of the study. The full study name will
        be `owners/{owner_id}/studies/{study_id}`.

    Returns:
      None.

    Raises:
      RuntimeError: If service reported an error or if a value could not be
      pickled.
    """
    study_resource_name = study_resource_name or (
        resources.StudyResource(self._owner_id, self._study_id).name
    )
    request = pyvizier.metadata_util.to_request_proto(
        study_resource_name, delta
    )
    response = self._service.UpdateMetadata(request)

    if response.error_details:
      raise RuntimeError(response.error_details)


def create_or_load_study(
    owner_id: str,
    client_id: str,
    study_id: str,
    study_config: pyvizier.StudyConfig,
) -> VizierClient:
  """Factory method for creating or loading a VizierClient.

  This will either create or load the specified study, given
  (owner_id, study_id, study_config). It will create it if it doesn't
  already exist, and load it if someone has already created it.

  Note that once a study is created, you CANNOT modify it with this function.

  This function is designed for use in a distributed system, where many jobs
  initially call create_or_load_study() nearly simultaneously with the same
  `study_config`. In that situation, all clients will end up pointing nicely
  to the same study.

  Args:
      owner_id: An owner id.
      client_id: ID for the VizierClient. See class for notes.
      study_id: Each study is uniquely identified by the tuple (owner_id,
        study_id).
      study_config: Study configuration for Vizier service. If not supplied, it
        will be assumed that the study with the given study_id already exists,
        and will try to retrieve that study.

  Returns:
      A VizierClient object with the specified study created or loaded.

  Raises:
      RuntimeError: Indicates that study_config is supplied but CreateStudy
          failed and GetStudy did not succeed after
          constants.MAX_NUM_TRIES_FOR_STUDIES tries.
      ValueError: Indicates that study_config is not supplied and the study
          with the given study_id does not exist.
  """
  vizier_stub = create_vizier_servicer_or_stub()
  study = study_pb2.Study(
      display_name=study_id, study_spec=study_config.to_proto()
  )
  request = vizier_service_pb2.CreateStudyRequest(
      parent=resources.OwnerResource(owner_id).name, study=study
  )
  # The response study contains a service assigned `name`, and may have been
  # created by this RPC or a previous RPC from another client.
  study = vizier_stub.CreateStudy(request)
  return VizierClient(study.name, client_id, vizier_stub)


def PollingDelay(num_attempts: int, time_scale: float) -> datetime.timedelta:  # pylint:disable=invalid-name
  """Computes a delay to the next attempt to poll the Vizier service.

  This does bounded exponential backoff, starting with $time_scale.
  If $time_scale == 0, it starts with a small time interval, less than
  1 second.

  Args:
    num_attempts: The number of times have we polled and found that the desired
      result was not yet available.
    time_scale: The shortest polling interval, in seconds, or zero. Zero is
      treated as a small interval, less than 1 second.

  Returns:
    A recommended delay interval, in seconds.
  """
  small_interval = 0.3  # Seconds
  interval = max(time_scale, small_interval) * 1.41 ** min(num_attempts, 9)
  return datetime.timedelta(seconds=interval)
