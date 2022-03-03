"""RPC functions implemented from vizier_service.proto."""
import collections
import random
import threading
from typing import Optional
from absl import logging
import grpc
import numpy as np

from vizier.pythia import base
from vizier.pythia.policies import random_policy
from vizier.pyvizier import oss as pyvizier
from vizier.pyvizier import pythia
from vizier.service import datastore
from vizier.service import resources
from vizier.service import service_policy_supporter
from vizier.service import study_pb2
from vizier.service import vizier_oss_pb2
from vizier.service import vizier_service_pb2
from vizier.service import vizier_service_pb2_grpc

from google.longrunning import operations_pb2
from google.protobuf import empty_pb2
from google.protobuf import timestamp_pb2
from google.rpc import code_pb2
from google.rpc import status_pb2


def policy_creator(
    algorithm: study_pb2.StudySpec.Algorithm,
    policy_supporter: service_policy_supporter.ServicePolicySupporter
) -> base.Policy:
  if algorithm == study_pb2.StudySpec.Algorithm.RANDOM_SEARCH:
    del policy_supporter  # Random Policy is stateless.
    return random_policy.RandomPolicy()
  else:
    raise ValueError(f'{algorithm} is not registered.')


def _get_current_time() -> timestamp_pb2.Timestamp:
  now = timestamp_pb2.Timestamp()
  now.GetCurrentTime()
  return now


MAX_STUDY_ID = 2147483647  # Max int32 value.


class VizierService(vizier_service_pb2_grpc.VizierServiceServicer):
  """Implements the GRPC functions outlined in vizier_service.proto."""

  def __init__(self):
    self.datastore = datastore.NestedDictRAMDataStore()

    # For database edits using owner names.
    self._owner_name_to_lock = collections.defaultdict(threading.Lock)
    # For database edits using study names.
    self._study_name_to_lock = collections.defaultdict(threading.Lock)

  def CreateStudy(
      self,
      request: vizier_service_pb2.CreateStudyRequest,
      context: Optional[grpc.ServicerContext] = None) -> study_pb2.Study:
    """Creates a study or loads an existing one.

    The initial request contains the study without the study name, only
    a user/client specified display_name for locating a potential pre-existing
    study. The returned study will then have the service-provided resource name.

    Args:
      request: A CreateStudyRequest.
      context: Optional GRPC ServicerContext.

    Returns:
      study: The Study created with generated service resource name.
    """
    study = request.study
    owner_id = resources.OwnerResource.from_name(request.parent).owner_id
    if request.study.name:
      raise ValueError(
          'Study should not have a resource name. Study names can only be assigned by the Vizier service.'
      )
    if not request.study.display_name:
      raise ValueError('Study display_name must be specified.')

    lock = self._owner_name_to_lock[request.parent]
    lock.acquire()
    # Database creates a new active study or loads existing study using the
    # display name.
    possible_candidate_studies = self.datastore.list_studies(request.parent)

    # Check if all possible study_id's have been taken.
    if len(possible_candidate_studies) >= MAX_STUDY_ID:
      raise ValueError(
          'Maximum number of studies reached for owner {}.'.format(owner_id))

    for candidate_study in possible_candidate_studies:
      if candidate_study.display_name == request.study.display_name:
        lock.release()
        return candidate_study
    # No study in the database matches the resource name. Making a new one.
    # study_id must be unique among existing studies.
    previous_study_ids = [
        resources.StudyResource.from_name(candidate_study.name).study_id
        for candidate_study in possible_candidate_studies
    ]
    study_id = str(random.randint(1, MAX_STUDY_ID))
    while study_id in previous_study_ids:
      study_id = str(random.randint(1, MAX_STUDY_ID))

    # Finally create study in database and return it.
    study.name = resources.StudyResource(owner_id, study_id).name
    self.datastore.create_study(study)

    lock.release()
    return study

  def GetStudy(
      self,
      request: vizier_service_pb2.GetStudyRequest,
      context: Optional[grpc.ServicerContext] = None) -> study_pb2.Study:
    """Gets a Study by name. If the study does not exist, return error."""
    return self.datastore.load_study(request.name)

  def ListStudies(
      self,
      request: vizier_service_pb2.ListStudiesRequest,
      context: Optional[grpc.ServicerContext] = None
  ) -> vizier_service_pb2.ListStudiesResponse:
    """Lists all the studies in a region for an associated project."""
    list_of_studies = self.datastore.list_studies(request.parent)
    return vizier_service_pb2.ListStudiesResponse(studies=list_of_studies)

  def DeleteStudy(
      self,
      request: vizier_service_pb2.DeleteStudyRequest,
      context: Optional[grpc.ServicerContext] = None) -> empty_pb2.Empty:
    """Deletes a Study."""
    lock = self._owner_name_to_lock[resources.StudyResource.from_name(
        request.name).owner_resource.name]
    lock.acquire()
    self.datastore.delete_study(request.name)
    lock.release()
    return empty_pb2.Empty()

  def SuggestTrials(
      self,
      request: vizier_service_pb2.SuggestTrialsRequest,
      context: Optional[grpc.ServicerContext] = None
  ) -> operations_pb2.Operation:
    """Adds one or more Trials to a Study, with parameter values suggested by a Pythia policy.

    Args:
      request:
      context:

    Returns:
      A long-running operation associated with the generation of Trial
      suggestions. When this long-running operation succeeds, it will contain a
      [SuggestTrialsResponse].
    """

    lock = self._study_name_to_lock[request.parent]
    lock.acquire()

    study_resource = resources.StudyResource.from_name(request.parent)
    study_id = study_resource.study_id
    study = self.datastore.load_study(request.parent)
    owner_id = study_resource.owner_id
    owner_name = study_resource.owner_resource.name

    # Checks for a non-done operation in the database with this name.
    filter_fn = lambda op: not op.done
    active_op_list = self.datastore.list_suggestion_operations(
        owner_name, request.client_id, filter_fn)

    if active_op_list:
      return active_op_list[0]  # We've found the active one!

    # If that failed, create a new Op.
    new_op_number = self.datastore.max_suggestion_operation_number(
        owner_name, request.client_id) + 1
    new_op_name = resources.SuggestionOperationResource(owner_id,
                                                        request.client_id,
                                                        new_op_number).name
    operation = operations_pb2.Operation(name=new_op_name, done=False)
    self.datastore.create_suggestion_operation(operation)

    policy_supporter = service_policy_supporter.ServicePolicySupporter(self)
    pythia_policy = policy_creator(study.study_spec.algorithm, policy_supporter)
    try:
      pythia_sc = pyvizier.StudyConfig.from_proto(study.study_spec).to_pythia()
      study_descriptor = pythia.StudyDescriptor(config=pythia_sc)
      suggest_request = base.SuggestRequest(
          study_descriptor=study_descriptor, count=request.suggestion_count)
      suggest_decisions = pythia_policy.suggest(suggest_request)

      py_trials = []
      for decision in suggest_decisions:
        py_trials.append(pyvizier.Trial(parameters=decision.parameters))
      output_trials = pyvizier.TrialConverter.to_protos(py_trials)

      start_time = _get_current_time()

      for trial in output_trials:
        trial_id = self.datastore.max_trial_id(request.parent) + 1
        trial.id = str(trial_id)
        trial.name = resources.TrialResource(owner_id, study_id, trial_id).name
        trial.state = study_pb2.Trial.State.ACTIVE
        trial.start_time.CopyFrom(start_time)
        trial.client_id = request.client_id
        self.datastore.create_trial(trial)

      operation.response.value = vizier_service_pb2.SuggestTrialsResponse(
          trials=output_trials, start_time=start_time).SerializeToString()
    # Leaving a broad catch for now since Pythia can raise any exception.
    except Exception as e:  # pylint: disable=broad-except
      operation.error = status_pb2.Status(
          code=code_pb2.Code.INTERNAL, message=str(e))
      logging.exception('Failed to request trials from Pythia for request: %s',
                        request)

    operation.done = True
    lock.release()
    return operation

  def GetOperation(
      self,
      request: operations_pb2.GetOperationRequest,
      context: Optional[grpc.ServicerContext] = None
  ) -> operations_pb2.Operation:
    """Gets the latest state of a SuggestTrials() long-running operation."""
    operation = self.datastore.get_suggestion_operation(request.name)
    return operation

  def CreateTrial(
      self,
      request: vizier_service_pb2.CreateTrialRequest,
      context: Optional[grpc.ServicerContext] = None) -> study_pb2.Trial:
    """Adds user provided Trial to a Study and assigns the correct fields."""
    trial = request.trial
    trial.id = str(self.datastore.max_trial_id(request.parent) + 1)

    if trial.state != study_pb2.Trial.State.SUCCEEDED:
      trial.state = study_pb2.Trial.State.REQUESTED
    trial.ClearField('client_id')

    trial.start_time.CopyFrom(_get_current_time())
    self.datastore.create_trial(trial)
    return trial

  def GetTrial(
      self,
      request: vizier_service_pb2.GetTrialRequest,
      context: Optional[grpc.ServicerContext] = None) -> study_pb2.Trial:
    """Gets a Trial."""
    return self.datastore.get_trial(request.name)

  def ListTrials(
      self,
      request: vizier_service_pb2.ListTrialsRequest,
      context: Optional[grpc.ServicerContext] = None
  ) -> vizier_service_pb2.ListTrialsResponse:
    """Lists the Trials associated with a Study."""
    list_of_trials = self.datastore.list_trials(request.parent)
    return vizier_service_pb2.ListTrialsResponse(trials=list_of_trials)

  def AddTrialMeasurement(
      self,
      request: vizier_service_pb2.AddTrialMeasurementRequest,
      context: Optional[grpc.ServicerContext] = None) -> study_pb2.Trial:
    """Adds a measurement of the objective metrics to a Trial.

    This measurement is assumed to have been taken before the Trial is
    complete.

    Args:
      request:
      context:

    Returns:
      Trial whose measurement was appended.

    """
    trial = self.datastore.get_trial(request.trial_name)
    trial.measurements.extend([request.measurement])
    return trial

  def CompleteTrial(
      self,
      request: vizier_service_pb2.CompleteTrialRequest,
      context: Optional[grpc.ServicerContext] = None) -> study_pb2.Trial:
    """Marks a Trial as complete."""
    trial = self.datastore.get_trial(request.name)
    if request.trial_infeasible:
      trial.state = study_pb2.Trial.State.INFEASIBLE
      trial.infeasible_reason = request.infeasible_reason
    else:
      trial.state = study_pb2.Trial.State.SUCCEEDED

    if request.final_measurement.metrics:
      trial.final_measurement.CopyFrom(request.final_measurement)
    else:
      if trial.measurements:
        # Trial's final measurement auto-selected from latest reported
        # measurement.
        trial.final_measurement.CopyFrom(trial.measurements[-1])
      else:
        raise ValueError(
            "Both the request and trial intermediate measurements are missing. Cannot determine trial's final_measurement."
        )
    return trial

  def DeleteTrial(
      self,
      request: vizier_service_pb2.DeleteTrialRequest,
      context: Optional[grpc.ServicerContext] = None) -> empty_pb2.Empty:
    """Deletes a Trial."""
    self.datastore.delete_trial(request.name)
    return empty_pb2.Empty()

  def CheckTrialEarlyStoppingState(
      self,
      request: vizier_service_pb2.CheckTrialEarlyStoppingStateRequest,
      context: Optional[grpc.ServicerContext] = None
  ) -> vizier_oss_pb2.EarlyStoppingOperation:
    """Checks whether a Trial should stop or not.

    Args:
      request:
      context:

    Returns:
      A long-running operation. When the operation is successful, it
      will contain a [CheckTrialEarlyStoppingStateResponse].
    """
    trial_resource = resources.TrialResource.from_name(request.trial_name)
    lock = self._study_name_to_lock[trial_resource.study_resource.name]
    lock.acquire()

    operation_name = resources.EarlyStoppingOperationResource(
        trial_resource.owner_id, trial_resource.study_id,
        trial_resource.trial_id).name

    try:
      output_operation = self.datastore.get_early_stopping_operation(
          operation_name)

    except KeyError:
      output_operation = vizier_oss_pb2.EarlyStoppingOperation(
          name=operation_name,
          status=vizier_oss_pb2.EarlyStoppingOperation.Status.ACTIVE)

      study = self.datastore.load_study(trial_resource.study_resource.name)
      policy_supporter = service_policy_supporter.ServicePolicySupporter(self)
      pythia_policy = policy_creator(study.study_spec.algorithm,
                                     policy_supporter)
      pythia_sc = pyvizier.StudyConfig.from_proto(study.study_spec).to_pythia()
      study_descriptor = pythia.StudyDescriptor(config=pythia_sc)
      early_stop_request = base.EarlyStopRequest(
          study_descriptor=study_descriptor,
          trial_ids=[trial_resource.trial_id])
      early_stopping_decisions = pythia_policy.early_stop(early_stop_request)

      for early_stopping_decision in early_stopping_decisions:
        if early_stopping_decision.should_stop:
          op_name = resources.EarlyStoppingOperationResource(
              trial_resource.owner_id, trial_resource.study_id,
              early_stopping_decision.id).name
          try:
            inner_operation = self.datastore.get_early_stopping_operation(
                op_name)
            inner_operation.should_stop = True
            inner_operation.status = vizier_oss_pb2.EarlyStoppingOperation.Status.DONE
          except KeyError:
            # inner_operation not found.
            pass

    lock.release()
    return output_operation

  def StopTrial(
      self,
      request: vizier_service_pb2.StopTrialRequest,
      context: Optional[grpc.ServicerContext] = None) -> study_pb2.Trial:
    trial = self.datastore.get_trial(request.name)
    trial.state = study_pb2.Trial.STOPPING
    return trial

  def ListOptimalTrials(
      self,
      request: vizier_service_pb2.ListOptimalTrialsRequest,
      context: Optional[grpc.ServicerContext] = None
  ) -> vizier_service_pb2.ListOptimalTrialsResponse:
    """The definition of pareto-optimal can be checked in wiki page.

    https://en.wikipedia.org/wiki/Pareto_efficiency.

    Args:
      request:
      context:

    Returns:
      A list containing pareto-optimal Trials for multi-objective Study or the
      optimal Trials for single-objective Study.
    """
    raw_trial_list = self.datastore.list_trials(request.parent)
    study_spec = self.datastore.load_study(request.parent).study_spec

    metric_id_to_goal = {m.metric_id: m.goal for m in study_spec.metrics}
    required_metric_ids = set(metric_id_to_goal.keys())

    considered_trials = []
    considered_trial_objective_vectors = []

    for trial in raw_trial_list:
      trial_metric_id_to_value = {
          m.metric_id: m.value for m in trial.final_measurement.metrics
      }
      trial_metric_ids = set(trial_metric_id_to_value.keys())
      # Add trials ONLY if they succeeded and contain all supposed metrics.
      if trial.state == study_pb2.Trial.State.SUCCEEDED and required_metric_ids.issubset(
          trial_metric_ids):
        objective_vector = []
        for metric_id, goal in metric_id_to_goal.items():
          # Flip sign for convenience when computing optimality.
          if goal == study_pb2.StudySpec.MetricSpec.GoalType.MINIMIZE:
            vector_value = -1.0 * trial_metric_id_to_value[metric_id]
          else:
            vector_value = trial_metric_id_to_value[metric_id]
          objective_vector.append(vector_value)

        considered_trials.append(trial)
        considered_trial_objective_vectors.append(objective_vector)

    # Find Pareto optimal trials.
    ys = np.array(considered_trial_objective_vectors)
    n = ys.shape[0]
    dominated = np.asarray(
        [[np.all(ys[i] <= ys[j]) & np.any(ys[j] > ys[i])
          for i in range(n)]
         for j in range(n)])
    optimal_booleans = np.logical_not(np.any(dominated, axis=0))
    optimal_trials = []
    for i, boolean in enumerate(list(optimal_booleans)):
      if boolean:
        optimal_trials.append(considered_trials[i])
    return vizier_service_pb2.ListOptimalTrialsResponse(
        optimal_trials=optimal_trials)
