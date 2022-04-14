"""Makes client calls via gRPC to an existing Vizier Service Server.

This client can be used interchangeably with the Cloud Vizier client.
"""

import datetime
import time
from typing import Any, Dict, List, Mapping, Optional, Text, Union

from absl import flags
from absl import logging
import grpc

from vizier.service import pyvizier
from vizier.service import resources
from vizier.service import study_pb2
from vizier.service import vizier_service_pb2
from vizier.service import vizier_service_pb2_grpc

from google.longrunning import operations_pb2
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import json_format

flags.DEFINE_integer(
    'vizier_new_suggestion_polling_secs', 1,
    'The period to wait between polling for the status of long-running '
    'SuggestOperations. Vizier may increase this period if multiple polls '
    'are needed. (You may use zero for interactive demos, but it is only '
    'appropriate for very small Studies.)')
FLAGS = flags.FLAGS


def _create_server_stub(
    service_endpoint: Text) -> vizier_service_pb2_grpc.VizierServiceStub:
  channel = grpc.secure_channel(service_endpoint,
                                grpc.local_channel_credentials())
  grpc.channel_ready_future(channel).result()
  return vizier_service_pb2_grpc.VizierServiceStub(channel)


class VizierClient:
  """Client for communicating with the Vizer Service via GRPC."""

  def __init__(self, service_endpoint: Text, study_name: Text, client_id: Text):
    """Create an VizierClient object.

    Use this constructor when you know the study_id, and when the Study
    already exists. Otherwise, you'll probably want to use
    create_or_load_study() instead of constructing the
    VizierClient class directly.

    Args:
        service_endpoint: Address of VizierService for creation of gRPC stub,
          e.g. 'localhost:8998'.
        study_name: An identifier of the study. The full study name will be
          `owners/{owner_id}/studies/{study_id}`.
        client_id: An ID that identifies the worker requesting a `Trial`.
          Workers that should run the same trial (for instance, when running a
          multi-worker model) should have the same ID. If multiple
          suggestTrialsRequests have the same client_id, the service will return
          the identical suggested trial if the trial is PENDING, and provide a
          new trial if the last suggest trial was completed.
    """
    self._server_stub = _create_server_stub(service_endpoint)
    self._study_name = study_name
    self._client_id = client_id
    study_resource = resources.StudyResource.from_name(self._study_name)
    self._owner_id = study_resource.owner_id
    self._study_id = study_resource.study_id

  @property
  def study_name(self) -> Text:
    return self._study_name

  def get_suggestions(self, suggestion_count: int) -> List[pyvizier.Trial]:
    """Gets a list of suggested Trials.

    Args:
        suggestion_count: The number of suggestions to request.

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
    request = vizier_service_pb2.SuggestTrialsRequest(
        parent=resources.StudyResource(self._owner_id, self._study_id).name,
        suggestion_count=suggestion_count,
        client_id=self._client_id)
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
      metric_list: List[Mapping[Text, Union[int, float]]],
      trial_id: int,
  ) -> None:
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
    logging.info('Trial modified: %s', trial)

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
      infeasibility_reason: Optional[Text] = None) -> pyvizier.Trial:
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

  def list_studies(self) -> List[Dict[Text, Any]]:
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

  def delete_study(self, study_name: Optional[Text] = None) -> None:
    """Deletes study from datastore."""
    if study_name:
      request_study_name = study_name
    else:
      request_study_name = resources.StudyResource(self._owner_id,
                                                   self._study_id).name
    request = vizier_service_pb2.DeleteStudyRequest(name=request_study_name)
    future = self._server_stub.DeleteStudy.future(request)
    empty_proto = future.result()

    # Confirms successful execution.
    assert isinstance(empty_proto, empty_pb2.Empty)
    logging.info('Study deleted: %s', study_name)


def create_or_load_study(
    service_endpoint: Text,
    owner_id: Text,
    client_id: Text,
    study_display_name: Text,
    study_config: pyvizier.StudyConfig,
) -> VizierClient:
  """Factory method for creating or loading a VizierClient.

  This will either create or load the specified study, given
  (owner_id, study_display_name, study_config). It will create it if it doesn't
  already exist, and load it if someone has already created it.

  Note that once a study is created, you CANNOT modify it with this function.

  This function is designed for use in a distributed system, where many jobs
  initially call create_or_load_study() nearly simultaneously with the same
  `study_config`. In that situation, all clients will end up pointing nicely
  to the same study.

  Args:
      service_endpoint: Address of VizierService for creation of gRPC stub,
        e.g. 'localhost:8998'.
      owner_id: An owner id.
      client_id: ID for the VizierClient. See class for notes.
      study_display_name: Each study is uniquely identified by (owner_id,
        study_display_name). The service will assign the study a unique
        `study_id`.
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
  vizier_stub = _create_server_stub(service_endpoint)
  study = study_pb2.Study(
      display_name=study_display_name, study_spec=study_config.to_proto())
  request = vizier_service_pb2.CreateStudyRequest(
      parent=resources.OwnerResource(owner_id).name, study=study)
  future = vizier_stub.CreateStudy.future(request)

  # The response study contains a service assigned `name`, and may have been
  # created by this RPC or a previous RPC from another client.
  study = future.result()
  return VizierClient(
      service_endpoint=service_endpoint,
      study_name=study.name,
      client_id=client_id)


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
