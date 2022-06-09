"""Makes client calls via gRPC to an existing Vizier Service Server.

This client can be used interchangeably with the Cloud Vizier client.
"""

import datetime
import functools
import time
from typing import Any, Dict, List, Mapping, Optional, Union

from absl import flags
from absl import logging
import attr
import grpc
from vizier.service import pyvizier
from vizier.service import resources
from vizier.service import study_pb2
from vizier.service import vizier_service_pb2
from vizier.service import vizier_service_pb2_grpc
from vizier.utils import attrs_utils

from google.longrunning import operations_pb2
from google.protobuf import duration_pb2
from google.protobuf import json_format

flags.DEFINE_integer(
    'vizier_new_suggestion_polling_secs', 1,
    'The period to wait between polling for the status of long-running '
    'SuggestOperations. Vizier may increase this period if multiple polls '
    'are needed. (You may use zero for interactive demos, but it is only '
    'appropriate for very small Studies.)')
FLAGS = flags.FLAGS


@functools.lru_cache
def create_server_stub(
    service_endpoint: str) -> vizier_service_pb2_grpc.VizierServiceStub:
  """Creates the GRPC stub.

  This method uses LRU cache so we create a single stub per endpoint (which is
  effectively one per binary). Stub and channel are both thread-safe and can
  take a while to create. The LRU cache makes binaries run faster, especially
  for unit tests.

  Args:
    service_endpoint:

  Returns:
    Vizier service stub at service_endpoint.
  """
  logging.info('Securing channel to %s.', service_endpoint)
  channel = grpc.secure_channel(service_endpoint,
                                grpc.local_channel_credentials())
  grpc.channel_ready_future(channel).result()
  logging.info('Secured channel to %s.', service_endpoint)
  return vizier_service_pb2_grpc.VizierServiceStub(channel)


@attr.frozen(init=True)
class VizierClient:
  """Client for communicating with the Vizer Service via GRPC.

  It can be initialized directly with a Vizier service stub, or created
  from endpoint. See also `create_server_stub`.
  """

  _server_stub: vizier_service_pb2_grpc.VizierServiceStub = attr.field()
  _study_resource_name: str = attr.field(
      validator=attr.validators.instance_of(str))
  _client_id: str = attr.field(validator=[
      attr.validators.instance_of(str), attrs_utils.assert_not_empty
  ])

  @property
  def _study_resource(self) -> resources.StudyResource:
    return resources.StudyResource.from_name(self._study_resource_name)

  @classmethod
  def from_endpoint(cls, service_endpoint: str, study_resource_name: str,
                    client_id: str) -> 'VizierClient':
    """Create a VizierClient object.

    Use this constructor when you know the study_resource_name, and when the
    Study already exists. Otherwise, you'll probably want to use
    create_or_load_study() instead of constructing the
    VizierClient class directly.

    Args:
      service_endpoint: Address of VizierService for creation of gRPC stub, e.g.
        'localhost:8998'.
      study_resource_name: An identifier of the study. The full study name will
        be `owners/{owner_id}/studies/{study_id}`.
      client_id: An ID that identifies the worker requesting a `Trial`. Workers
        that should run the same trial (for instance, when running a
        multi-worker model) should have the same ID. If multiple
        suggestTrialsRequests have the same client_id, the service will return
        the identical suggested trial if the trial is PENDING, and provide a new
        trial if the last suggest trial was completed.

    Returns:
      Vizier client.
    """
    return cls(
        create_server_stub(service_endpoint), study_resource_name, client_id)

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
      self,
      suggestion_count: int,
      *,
      client_id_override: Optional[str] = None) -> List[pyvizier.Trial]:
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
        client_id=client_id)
    future = self._server_stub.SuggestTrials.future(request)
    operation = future.result()

    num_attempts = 0
    while not operation.done:
      sleep_time = PollingDelay(num_attempts,
                                FLAGS.vizier_new_suggestion_polling_secs)
      num_attempts += 1
      logging.info('Waiting for operation with name %s to be done',
                   operation.name)
      time.sleep(sleep_time.total_seconds())

      operation = self._server_stub.GetOperation(
          operations_pb2.GetOperationRequest(name=operation.name))

    if operation.HasField('error'):
      error_message = 'SuggestOperation {} failed with message: {}'.format(
          operation.name, operation.error)
      logging.error(error_message)
      raise RuntimeError(error_message)
    # TODO: Replace with any.Unpack().
    trials = vizier_service_pb2.SuggestTrialsResponse.FromString(
        operation.response.value).trials
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
            metric_id=metric_name, value=metric[metric_name])
        new_metric_list.append(metric_pb2)

    integer_seconds = int(elapsed_secs)
    nano_seconds = int((elapsed_secs - integer_seconds) * 1e9)
    measurement = study_pb2.Measurement(
        elapsed_duration=duration_pb2.Duration(
            seconds=integer_seconds, nanos=nano_seconds),
        step_count=step,
        metrics=new_metric_list)
    request = vizier_service_pb2.AddTrialMeasurementRequest(
        trial_name=resources.TrialResource(self._owner_id, self._study_id,
                                           trial_id).name,
        measurement=measurement)
    future = self._server_stub.AddTrialMeasurement.future(request)
    trial = future.result()
    return pyvizier.TrialConverter.from_proto(trial)

  def should_trial_stop(self, trial_id: int) -> bool:
    request = vizier_service_pb2.CheckTrialEarlyStoppingStateRequest(
        trial_name=resources.TrialResource(self._owner_id, self._study_id,
                                           trial_id).name)
    future = self._server_stub.CheckTrialEarlyStoppingState.future(request)
    early_stopping_response = future.result()
    return early_stopping_response.should_stop

  def stop_trial(self, trial_id: int) -> None:
    request = vizier_service_pb2.StopTrialRequest(
        name=resources.TrialResource(self._owner_id, self._study_id,
                                     trial_id).name)
    self._server_stub.StopTrial(request)
    logging.info('Trial with id %s stopped.', trial_id)

  def complete_trial(
      self,
      trial_id: int,
      final_measurement: Optional[pyvizier.Measurement] = None,
      infeasibility_reason: Optional[str] = None) -> pyvizier.Trial:
    """Completes the trial, which is infeasible if given a infeasibility_reason."""
    request = vizier_service_pb2.CompleteTrialRequest(
        name=resources.TrialResource(self._owner_id, self._study_id,
                                     trial_id).name,
        trial_infeasible=infeasibility_reason is not None,
        infeasible_reason=infeasibility_reason)

    if final_measurement is not None:
      # Final measurement can still be included even for infeasible trials, for
      # other metrics, or a subset of objective + safety metrics.
      request.final_measurement.CopyFrom(
          pyvizier.MeasurementConverter.to_proto(final_measurement))

    future = self._server_stub.CompleteTrial.future(request)
    trial = future.result()
    return pyvizier.TrialConverter.from_proto(trial)

  def get_trial(self, trial_id: int) -> pyvizier.Trial:
    """Return the Optimizer trial for the given trial_id."""
    request = vizier_service_pb2.GetTrialRequest(
        name=resources.TrialResource(self._owner_id, self._study_id,
                                     trial_id).name)
    future = self._server_stub.GetTrial.future(request)
    trial = future.result()
    return pyvizier.TrialConverter.from_proto(trial)

  def list_trials(self) -> List[pyvizier.Trial]:
    """List all trials."""
    parent = resources.StudyResource(self._owner_id, self._study_id).name
    request = vizier_service_pb2.ListTrialsRequest(parent=parent)
    future = self._server_stub.ListTrials.future(request)

    response = future.result()
    return pyvizier.TrialConverter.from_protos(response.trials)

  def list_optimal_trials(self) -> List[pyvizier.Trial]:
    """List only the optimal completed trials."""
    parent = resources.StudyResource(self._owner_id, self._study_id).name
    request = vizier_service_pb2.ListOptimalTrialsRequest(parent=parent)
    future = self._server_stub.ListOptimalTrials.future(request)
    response = future.result()
    return pyvizier.TrialConverter.from_protos(response.optimal_trials)

  def list_studies(self) -> List[Dict[str, Any]]:
    """List all studies for the given owner."""
    request = vizier_service_pb2.ListStudiesRequest(
        parent=resources.OwnerResource(self._owner_id).name)
    future = self._server_stub.ListStudies.future(request)
    list_studies_response = future.result()
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
        trial=pyvizier.TrialConverter.to_proto(trial))
    future = self._server_stub.CreateTrial.future(request)
    trial_proto = future.result()
    return pyvizier.TrialConverter.from_proto(trial_proto)

  def delete_trial(self, trial_id: int) -> None:
    """Deletes trial from datastore."""
    request_trial_name = resources.TrialResource(self._owner_id, self._study_id,
                                                 trial_id).name
    request = vizier_service_pb2.DeleteTrialRequest(name=request_trial_name)
    future = self._server_stub.DeleteTrial.future(request)
    _ = future.result()
    logging.info('Trial deleted: %s', trial_id)

  def delete_study(self, study_resource_name: Optional[str] = None, /) -> None:
    """Deletes study from datastore."""
    study_resource_name = study_resource_name or (resources.StudyResource(
        self._owner_id, self._study_id).name)
    request = vizier_service_pb2.DeleteStudyRequest(name=study_resource_name)
    future = self._server_stub.DeleteStudy.future(request)
    _ = future.result()
    logging.info('Study deleted: %s', study_resource_name)

  def get_study_config(
      self, study_name: Optional[str] = None) -> pyvizier.StudyConfig:
    """Returns the study config."""
    if study_name:
      study_resource_name = study_name
    else:
      study_resource_name = resources.StudyResource(self._owner_id,
                                                    self._study_id).name
    request = vizier_service_pb2.GetStudyRequest(name=study_resource_name)
    future = self._server_stub.GetStudy.future(request)
    response = future.result()

    return pyvizier.StudyConfig.from_proto(response.study_spec)


def create_or_load_study(
    service_endpoint: str,
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
      service_endpoint: Address of VizierService for creation of gRPC stub, e.g.
        'localhost:8998'.
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
  vizier_stub = create_server_stub(service_endpoint)
  study = study_pb2.Study(
      display_name=study_id, study_spec=study_config.to_proto())
  request = vizier_service_pb2.CreateStudyRequest(
      parent=resources.OwnerResource(owner_id).name, study=study)
  future = vizier_stub.CreateStudy.future(request)

  # The response study contains a service assigned `name`, and may have been
  # created by this RPC or a previous RPC from another client.
  study = future.result()
  return VizierClient(vizier_stub, study.name, client_id)


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
  interval = max(time_scale, small_interval) * 1.41**min(num_attempts, 9)
  return datetime.timedelta(seconds=interval)
