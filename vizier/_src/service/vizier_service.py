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

"""RPC functions implemented from vizier_service.proto."""

import collections
import datetime
import threading
from typing import Optional

from absl import logging
import grpc
import numpy as np
import sqlalchemy as sqla
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.service import constants
from vizier._src.service import custom_errors
from vizier._src.service import grpc_util
from vizier._src.service import pythia_service
from vizier._src.service import ram_datastore
from vizier._src.service import resources
from vizier._src.service import sql_datastore
from vizier._src.service import stubs_util
from vizier._src.service import study_pb2
from vizier._src.service import types
from vizier._src.service import vizier_oss_pb2
from vizier._src.service import vizier_service_pb2
from vizier._src.service import vizier_service_pb2_grpc
from vizier.service import pyvizier as svz

from google.longrunning import operations_pb2
from google.protobuf import empty_pb2
from google.protobuf import timestamp_pb2
from google.rpc import code_pb2
from google.rpc import status_pb2


def _get_current_time() -> timestamp_pb2.Timestamp:
  now = timestamp_pb2.Timestamp()
  now.GetCurrentTime()
  return now


StudyResource = resources.StudyResource
TrialResource = resources.TrialResource


# TODO: remove context = None
# TODO: remove context = None
class VizierServicer(vizier_service_pb2_grpc.VizierServiceServicer):
  """Implements the GRPC functions outlined in vizier_service.proto."""

  # Trial states in which a trial can be modified.
  _TRIAL_MUTABLE_STATES = (
      study_pb2.Trial.State.ACTIVE,
      study_pb2.Trial.State.STOPPING,
  )

  def __init__(
      self,
      database_url: Optional[str] = constants.SQL_LOCAL_URL,
      early_stop_recycle_period: datetime.timedelta = datetime.timedelta(
          seconds=60
      ),
      default_pythia_service: Optional[types.PythiaService] = None,
  ):
    """Initializes the service.

    Creates the datastore and relevant locks for multhreading. Note that the
    datastore input/output is assumed to always be pass-by-value.

    Args:
      database_url: URL to the SQL database. If None, it connects to our custom
        RAM Datastore.
      early_stop_recycle_period: Amount of time needed to pass before recycling
        an early stopping operation. See `CheckEarlyStoppingState` for more
        details.
      default_pythia_service: Default PythiaService to use when
        StudyConfig.pythia_endpoint is unset. If None, creates a local
        PythiaServicer.
    """
    # By default, uses a local PythiaServicer instance.
    self.default_pythia_service: types.PythiaService = (
        default_pythia_service
        or pythia_service.PythiaServicer(vizier_service=self)
    )

    if database_url is None:
      self.datastore = ram_datastore.NestedDictRAMDataStore()
    else:
      engine = sqla.create_engine(
          database_url,
          echo=False,  # Set True to log transactions for debugging.
          connect_args={'check_same_thread': False},
          poolclass=sqla.pool.StaticPool,
      )
      self.datastore = sql_datastore.SQLDataStore(engine)

    # For database edits using owner names.
    self._owner_name_to_lock = collections.defaultdict(threading.Lock)
    # For database edits using study names.
    self._study_name_to_lock = collections.defaultdict(threading.Lock)
    # For calls to Pythia (SuggestTrials and CheckTrialEarlyStoppingState).
    self._operation_lock = collections.defaultdict(threading.Lock)

    self._early_stop_recycle_period = early_stop_recycle_period

  def _select_pythia_service(
      self, endpoint: Optional[str] = None
  ) -> types.PythiaService:
    """Selects PythiaService to use."""
    # TODO: Add test for StudyConfig endpointing.
    if endpoint is None:
      logging.info('Using default PythiaServicer.')
      return self.default_pythia_service

    logging.info('Connecting to Pythia endpoint: %s', endpoint)
    pythia_stub = stubs_util.create_pythia_server_stub(endpoint)
    logging.info('Created Pythia Server stub: %s', pythia_stub)
    return pythia_stub

  def _study_is_immutable(self, study_name: str) -> bool:
    """Checks if study is immutable to block study-related mutations."""
    study = self.datastore.load_study(study_name)
    return study.state not in (
        study_pb2.Study.State.ACTIVE,
        study_pb2.Study.State.STATE_UNSPECIFIED,
    )

  def CreateStudy(
      self,
      request: vizier_service_pb2.CreateStudyRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> study_pb2.Study:
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
      e = ValueError(
          'Maximum number of studies reached for owner {}.'.format(owner_id)
      )
      grpc_util.handle_exception(e, context)
    if not request.study.display_name:
      grpc_util.handle_exception(
          ValueError('Study display_name must be specified.'), context
      )

    with self._owner_name_to_lock[request.parent]:
      # Database creates a new active study or loads existing study using the
      # display name.
      try:
        possible_candidate_studies = self.datastore.list_studies(request.parent)
      except custom_errors.NotFoundError:
        possible_candidate_studies = []

      # Check if all possible study_id's have been taken.
      if len(possible_candidate_studies) >= constants.MAX_STUDY_ID:
        e = ValueError(
            'Maximum number of studies reached for owner {}.'.format(owner_id)
        )
        grpc_util.handle_exception(e, context)

      for candidate_study in possible_candidate_studies:
        if candidate_study.display_name == request.study.display_name:
          logging.info(
              'Found existing study of owner=%s display_name=%s!',
              owner_id,
              candidate_study.display_name,
          )
          return candidate_study

      # No study in the database matches the resource name. Making a new one.
      # study_id must be unique among existing studies.
      study_id = study.display_name

      # Finally create study in database and return it.
      study.name = StudyResource(owner_id, study_id).name
      self.datastore.create_study(study)
    return study

  def GetStudy(
      self,
      request: vizier_service_pb2.GetStudyRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> study_pb2.Study:
    """Gets a Study by name. If the study does not exist, return error."""
    return self.datastore.load_study(request.name)

  def ListStudies(
      self,
      request: vizier_service_pb2.ListStudiesRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> vizier_service_pb2.ListStudiesResponse:
    """Lists all the studies in a region for an associated project."""
    studies = self.datastore.list_studies(request.parent)
    return vizier_service_pb2.ListStudiesResponse(studies=studies)

  def DeleteStudy(
      self,
      request: vizier_service_pb2.DeleteStudyRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> empty_pb2.Empty:
    """Deletes a Study."""
    self.datastore.delete_study(request.name)
    return empty_pb2.Empty()

  def SetStudyState(
      self,
      request: vizier_service_pb2.SetStudyStateRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> study_pb2.Study:
    with self._study_name_to_lock[request.parent]:
      study = self.datastore.load_study(request.parent)
      study.state = request.state
      self.datastore.update_study(study)
    return study

  def SuggestTrials(
      self,
      request: vizier_service_pb2.SuggestTrialsRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> operations_pb2.Operation:
    """Adds one or more Trials to a Study, with parameter values suggested by a Pythia policy.

    The logic is as follows:
    1. If there is already an active (not done) operation, simply return that.
    2. Else, create a new active operation.
    3. We need the requested number of ACTIVE trials to return. These will come
    from 3 sources:
      A. ACTIVE trials already assigned to the client.
      B. REQUESTED trials which not been assigned to any client.
      C. Pythia-computed suggestions.

    We first look from source A, then source B, then source C. If we ever reach
    source C and Pythia over-delivers too many suggestions (e.g. evolutionary
    algorithms sometimes do so because of batched behaviors), we put the extra
    suggestions into the REQUESTED pool (i.e. source B) for future use.

    The returned operation can either be:
    1. Done (Successfully contains the requested number of suggestions)
    2. Error-ed, which will happen if Pythia had an issue producing the right
    number of trials (via numerical crash or under-delivering).

    Args:
      request:
      context:

    Returns:
      A long-running operation associated with the generation of Trial
      suggestions. When this long-running operation succeeds, it will contain a
      [SuggestTrialsResponse].

    Raises:
      ImmutableStudyError: If study was already immutable.
    """
    # Convenient names and id's to be used below.
    study_name = request.parent
    if self._study_is_immutable(study_name):
      e = custom_errors.ImmutableStudyError(
          'Study {} is immutable. Cannot suggest trial.'.format(study_name)
      )
      grpc_util.handle_exception(e, context)

    study_resource = StudyResource.from_name(study_name)
    study_id = study_resource.study_id
    owner_id = study_resource.owner_id

    # Don't allow simultaneous SuggestTrial or EarlyStopping calls to be
    # processed.
    with self._operation_lock[request.parent]:
      study = self.datastore.load_study(request.parent)

      # Checks for a non-done operation in the database with this name.
      active_op_filter_fn = lambda op: not op.done
      try:
        active_op_list = self.datastore.list_suggestion_operations(
            study_name, request.client_id, active_op_filter_fn
        )
      except custom_errors.NotFoundError:
        active_op_list = []
      if active_op_list:
        return active_op_list[0]  # We've found the active one!

      start_time = _get_current_time()
      # Create a new Op if there aren't any active (not done) ops.
      try:
        old_op_number = self.datastore.max_suggestion_operation_number(
            study_name, request.client_id
        )
      except custom_errors.NotFoundError:
        old_op_number = 0
      new_op_number = old_op_number + 1
      new_op_name = resources.SuggestionOperationResource(
          owner_id, study_id, request.client_id, new_op_number
      ).name
      output_op = operations_pb2.Operation(name=new_op_name, done=False)
      self.datastore.create_suggestion_operation(output_op)

      # Check how many ACTIVE trials already exist for this client only.
      all_trials = self.datastore.list_trials(study_name)
      active_trials = [
          t
          for t in all_trials
          if t.state == study_pb2.Trial.State.ACTIVE
          and t.client_id == request.client_id
      ]
      if len(active_trials) >= request.suggestion_count:
        output_op.response.value = vizier_service_pb2.SuggestTrialsResponse(
            trials=active_trials[: request.suggestion_count],
            start_time=start_time,
        ).SerializeToString()
        output_op.done = True
        self.datastore.update_suggestion_operation(output_op)
        return output_op

      # Get suggestions from the pool of requested trials.
      output_trials = active_trials
      requested_trials = [
          t for t in all_trials if t.state == study_pb2.Trial.State.REQUESTED
      ]
      while requested_trials and request.suggestion_count > len(output_trials):
        assigned_trial = requested_trials.pop()
        assigned_trial.state = study_pb2.Trial.State.ACTIVE
        assigned_trial.client_id = request.client_id
        assigned_trial.start_time.CopyFrom(start_time)
        self.datastore.update_trial(assigned_trial)
        output_trials.append(assigned_trial)

      if len(output_trials) == request.suggestion_count:
        # We've finished collecting enough trials from the REQUESTED pool.
        output_op.response.value = vizier_service_pb2.SuggestTrialsResponse(
            trials=output_trials, start_time=start_time
        ).SerializeToString()
        output_op.done = True
        self.datastore.update_suggestion_operation(output_op)
        return output_op

      # Still need more suggestions. Pythia begins computing missing amount.
      study_config = svz.StudyConfig.from_proto(study.study_spec)
      study_descriptor = vz.StudyDescriptor(
          config=study_config,
          guid=study_name,
          max_trial_id=self.datastore.max_trial_id(study_name),
      )
      suggest_request = pythia.SuggestRequest(
          study_descriptor=study_descriptor,
          count=request.suggestion_count - len(output_trials),
      )

      # Convert request, send to Pythia, and obtain suggestions.
      suggest_request_proto = svz.SuggestConverter.to_request_proto(
          suggest_request
      )
      suggest_request_proto.algorithm = study.study_spec.algorithm

      try:
        temp_pythia_service = self._select_pythia_service(
            study_config.pythia_endpoint
        )
        suggest_decision_proto = temp_pythia_service.Suggest(
            suggest_request_proto
        )
      # Pythia can raise any exception, captured inside grpc.RpcError.
      except grpc.RpcError as e:
        output_op.error.CopyFrom(
            status_pb2.Status(code=code_pb2.Code.INTERNAL, message=str(e))
        )
        logging.exception(
            'Failed to request trials from Pythia for request: %s', request
        )
        output_op.done = True
        self.datastore.update_suggestion_operation(output_op)
        return output_op

      # Check if we received enough suggestions.
      if len(
          suggest_decision_proto.suggestions
      ) < request.suggestion_count - len(output_trials):
        logging.warning(
            'Requested at least %d suggestions but Pythia only produced %d.',
            request.suggestion_count - len(output_trials),
            len(suggest_decision_proto.suggestions),
        )
      suggest_decision = svz.SuggestConverter.from_decision_proto(
          suggest_decision_proto
      )

      # Write the metadata update to the datastore.
      try:
        self.datastore.update_metadata(
            study_name,
            svz.metadata_util.make_key_value_list(
                suggest_decision.metadata.on_study
            ),
            svz.metadata_util.trial_metadata_to_update_list(
                suggest_decision.metadata.on_trials
            ),
        )
      except KeyError as e:
        output_op.error.CopyFrom(
            status_pb2.Status(code=code_pb2.Code.INTERNAL, message=str(e))
        )
        logging.exception(
            'Failed to write metadata update to datastore: %s',
            suggest_decision.metadata,
        )
        output_op.done = True
        self.datastore.update_suggestion_operation(output_op)
        return output_op

      new_py_trials = [
          decision.to_trial() for decision in suggest_decision.suggestions
      ]
      new_trials = svz.TrialConverter.to_protos(new_py_trials)

      while request.suggestion_count > len(output_trials):
        new_trial = new_trials.pop()
        trial_id = self.datastore.max_trial_id(request.parent) + 1
        new_trial.id = str(trial_id)
        new_trial.name = TrialResource(owner_id, study_id, trial_id).name
        new_trial.state = study_pb2.Trial.State.ACTIVE
        new_trial.start_time.CopyFrom(start_time)
        new_trial.client_id = request.client_id
        self.datastore.create_trial(new_trial)
        output_trials.append(new_trial)

      output_op.response.value = vizier_service_pb2.SuggestTrialsResponse(
          trials=output_trials, start_time=start_time
      ).SerializeToString()

      # Store remaining trials as REQUESTED if Pythia over-delivered.
      for remain_trial in new_trials:
        trial_id = self.datastore.max_trial_id(request.parent) + 1
        remain_trial.id = str(trial_id)
        remain_trial.name = TrialResource(owner_id, study_id, trial_id).name
        remain_trial.state = study_pb2.Trial.State.REQUESTED
        self.datastore.create_trial(remain_trial)

      output_op.done = True
      self.datastore.update_suggestion_operation(output_op)
      return output_op

  def GetOperation(
      self,
      request: operations_pb2.GetOperationRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> operations_pb2.Operation:
    """Gets the latest state of a SuggestTrials() long-running operation."""
    return self.datastore.get_suggestion_operation(request.name)

  def CreateTrial(
      self,
      request: vizier_service_pb2.CreateTrialRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> study_pb2.Trial:
    """Adds user provided Trial to a Study and assigns the correct fields."""
    if self._study_is_immutable(request.parent):
      e = custom_errors.ImmutableStudyError(
          'Study {} is immutable. Cannot create trial.'.format(request.parent)
      )
      grpc_util.handle_exception(e, context)

    trial = request.trial
    with self._study_name_to_lock[request.parent]:
      trial.id = str(self.datastore.max_trial_id(request.parent) + 1)
      study_resource = StudyResource.from_name(request.parent)
      trial.name = (study_resource.trial_resource(trial.id)).name

      if trial.state != study_pb2.Trial.State.SUCCEEDED:
        trial.state = study_pb2.Trial.State.REQUESTED
      trial.ClearField('client_id')

      trial.start_time.CopyFrom(_get_current_time())
      self.datastore.create_trial(trial)
    return trial

  def GetTrial(
      self,
      request: vizier_service_pb2.GetTrialRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> Optional[study_pb2.Trial]:
    """Gets a Trial."""
    return self.datastore.get_trial(request.name)

  def ListTrials(
      self,
      request: vizier_service_pb2.ListTrialsRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> vizier_service_pb2.ListTrialsResponse:
    """Lists the Trials associated with a Study."""
    list_of_trials = self.datastore.list_trials(request.parent)
    return vizier_service_pb2.ListTrialsResponse(trials=list_of_trials)

  def AddTrialMeasurement(
      self,
      request: vizier_service_pb2.AddTrialMeasurementRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> study_pb2.Trial:
    """Adds a measurement of the objective metrics to a Trial.

    This measurement is assumed to have been taken before the Trial is
    complete.

    Args:
      request:
      context:

    Returns:
      Trial whose measurement was appended.

    Raises:
      ImmutableStudyError: If study was already immutable.
      ImmutableTrialError: If the trial cannot be modified.
    """
    study_name = TrialResource.from_name(request.trial_name).study_resource.name
    if self._study_is_immutable(study_name):
      e = custom_errors.ImmutableStudyError(
          'Study {} is immutable. Cannot add measurement.'.format(study_name)
      )
      grpc_util.handle_exception(e, context)

    with self._study_name_to_lock[study_name]:
      trial = self.datastore.get_trial(request.trial_name)
      if trial.state == study_pb2.Trial.State.INFEASIBLE:
        return trial
      if trial.state not in self._TRIAL_MUTABLE_STATES:
        e = custom_errors.ImmutableTrialError(
            'Trial {} has state {}. Measurements can only be added to trials in'
            ' state ACTIVE or STOPPING'.format(
                request.trial_name, study_pb2.Trial.State.Name(trial.state)
            )
        )
        grpc_util.handle_exception(e, context)

      trial.measurements.extend([request.measurement])
      self.datastore.update_trial(trial)
    return trial

  # TODO: Auto selection defaults to the last measurement.
  # Add support for "best measurement" behavior.
  def CompleteTrial(
      self,
      request: vizier_service_pb2.CompleteTrialRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> study_pb2.Trial:
    """Marks a Trial as complete."""
    study_name = TrialResource.from_name(request.name).study_resource.name
    if self._study_is_immutable(study_name):
      e = custom_errors.ImmutableStudyError(
          'Study {} is immutable. Cannot complete trial.'.format(study_name)
      )
      grpc_util.handle_exception(e, context)

    with self._study_name_to_lock[study_name]:
      trial = self.datastore.get_trial(request.name)
      if trial.state not in self._TRIAL_MUTABLE_STATES:
        e = custom_errors.ImmutableTrialError(
            'Trial {} has state {}. Only trials in state ACTIVE or STOPPING '
            'can be completed.'.format(
                request.name, study_pb2.Trial.State.Name(trial.state)
            )
        )
        grpc_util.handle_exception(e, context)

      trial.state = study_pb2.Trial.State.SUCCEEDED
      if request.final_measurement.metrics:
        trial.final_measurement.CopyFrom(request.final_measurement)
      elif not request.trial_infeasible:
        # Trial's final measurement auto-selected from latest reported
        # measurement.
        if not trial.measurements:
          e = ValueError(
              'Both the request and trial intermediate measurements are'
              " missing. Cannot determine trial's final_measurement."
          )
          grpc_util.handle_exception(e, context)
        trial.final_measurement.CopyFrom(trial.measurements[-1])

      # Handle infeasibility.
      if request.trial_infeasible:
        trial.state = study_pb2.Trial.State.INFEASIBLE
        trial.infeasible_reason = request.infeasible_reason

      self.datastore.update_trial(trial)
    return trial

  def DeleteTrial(
      self,
      request: vizier_service_pb2.DeleteTrialRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> empty_pb2.Empty:
    """Deletes a Trial."""
    study_name = TrialResource.from_name(request.name).study_resource.name
    if self._study_is_immutable(study_name):
      e = custom_errors.ImmutableStudyError(
          'Study {} is immutable. Cannot delete trial.'.format(study_name)
      )
      grpc_util.handle_exception(e, context)

    self.datastore.delete_trial(request.name)
    return empty_pb2.Empty()

  # TODO: This currently uses the same algorithm as suggestion.
  def CheckTrialEarlyStoppingState(
      self,
      request: vizier_service_pb2.CheckTrialEarlyStoppingStateRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> vizier_service_pb2.CheckTrialEarlyStoppingStateResponse:
    """Checks whether a Trial should stop or not.

    The logic is as follows:

    1. We check if there already exists an operation. If not, create it. But if
    it already exists:
      a. If it's still ACTIVE (i.e. early stopping is still being computed by
      Pythia), just return it.
      b. If it's done and the call is very recent, just return it.
      c. If it's been a long time, we can recycle the operation, and remark it
      as ACTIVE to be used again.

    2. Send the operation to Pythia, which will return a set of early stopping
    decisions. The Pythia policy is guaranteed to return a decision for the
    requested trial. However, Pythia policy may also signal whether other trials
    should stop or not (e.g. in algorithms in which proposed trials are batched
    and correlated).

    3. Transfer these decisions' `should_stop` into the relevant operations
    (some of which will also be created in this call, for future use), including
    the original operation which needs to be returned. Mark them all as done.

    4. Return the original trial's `should_stop` decision.

    Args:
      request:
      context:

    Returns:
      A CheckTrialEarlyStoppingStateResponse, containing a bool to denote
      whether the requested trial should stop.

    Raises:
      ImmutableStudyError: If study was already immutable.
      ImmutableTrialError: If the trial cannot be modified.
    """
    trial_resource = TrialResource.from_name(request.trial_name)
    study_name = trial_resource.study_resource.name
    if self._study_is_immutable(study_name):
      e = custom_errors.ImmutableStudyError(
          'Study {} is immutable. Cannot early stop trial.'.format(study_name)
      )
      grpc_util.handle_exception(e, context)

    with self._study_name_to_lock[study_name]:
      trial = self.datastore.get_trial(request.trial_name)
      if trial.state not in self._TRIAL_MUTABLE_STATES:
        e = custom_errors.ImmutableTrialError(
            'Trial {} has state {}. Only trials in state ACTIVE or STOPPING '
            'can be completed.'.format(
                request.trial_name, study_pb2.Trial.State.Name(trial.state)
            )
        )
        grpc_util.handle_exception(e, context)
    outer_op_name = trial_resource.early_stopping_operation_resource.name
    # Don't allow simultaneous SuggestTrial or EarlyStopping calls to be
    # processed.
    with self._operation_lock[study_name]:
      try:
        # Reuse any existing early stopping op, since the Pythia policy may have
        # already signaled this trial to stop.
        output_operation = self.datastore.get_early_stopping_operation(
            outer_op_name
        )
      except KeyError:
        output_operation = None

      if output_operation is None:
        # Create fresh new operation.
        output_operation = vizier_oss_pb2.EarlyStoppingOperation(
            name=outer_op_name,
            status=vizier_oss_pb2.EarlyStoppingOperation.Status.ACTIVE,
            should_stop=False,
        )
        output_operation.creation_time.CopyFrom(_get_current_time())
        self.datastore.create_early_stopping_operation(output_operation)
      else:
        if (
            output_operation.status
            == vizier_oss_pb2.EarlyStoppingOperation.Status.ACTIVE
            or datetime.datetime.utcnow()
            - output_operation.completion_time.ToDatetime()
            < self._early_stop_recycle_period
        ):
          # Operation is already active or very recent. Just return it.
          return vizier_service_pb2.CheckTrialEarlyStoppingStateResponse(
              should_stop=output_operation.should_stop
          )

        # Recycle the operation to ACTIVE again and start Pythia for
        # recomputation.
        output_operation.status = (
            vizier_oss_pb2.EarlyStoppingOperation.Status.ACTIVE
        )
        output_operation.should_stop = False  # Defaulted back to False.
        self.datastore.update_early_stopping_operation(output_operation)

      study = self.datastore.load_study(study_name)
      study_config = svz.StudyConfig.from_proto(study.study_spec)
      study_descriptor = vz.StudyDescriptor(
          config=study_config,
          guid=study_name,
          max_trial_id=self.datastore.max_trial_id(study_name),
      )
      early_stop_request = pythia.EarlyStopRequest(
          study_descriptor=study_descriptor, trial_ids=[trial_resource.trial_id]
      )
      early_stop_request_proto = svz.EarlyStopConverter.to_request_proto(
          early_stop_request
      )
      spec_name = (
          study.study_spec.WhichOneof('automated_stopping_spec')
          or 'default_stopping_spec'
      )
      if spec_name == 'default_stopping_spec':
        # TODO: Add TSGP algorithm when open sourced.
        early_stop_request_proto.algorithm = 'RANDOM_SEARCH'
      else:
        raise ValueError(
            f'Misconfigured automated_stopping_spec: {study.study_spec}'
        )

      # Send request to Pythia.
      temp_pythia_service = self._select_pythia_service(
          study_config.pythia_endpoint
      )
      early_stopping_decisions_proto = temp_pythia_service.EarlyStop(
          early_stop_request_proto
      )
      early_stopping_decisions = svz.EarlyStopConverter.from_decisions_proto(
          early_stopping_decisions_proto
      )
      # Update metadata from result.
      self.datastore.update_metadata(
          study_name,
          svz.metadata_util.make_key_value_list(
              early_stopping_decisions.metadata.on_study
          ),
          svz.metadata_util.trial_metadata_to_update_list(
              early_stopping_decisions.metadata.on_trials
          ),
      )

      # Pythia does not guarantee that the output_operation's id
      # will be in the decisions.
      for early_stopping_decision in early_stopping_decisions.decisions:
        inner_op_name = resources.EarlyStoppingOperationResource(
            trial_resource.owner_id,
            trial_resource.study_id,
            early_stopping_decision.id,
        ).name
        try:
          inner_operation = self.datastore.get_early_stopping_operation(
              inner_op_name
          )
        except KeyError:
          # Create the operation to store early stopping data for future use.
          inner_operation = vizier_oss_pb2.EarlyStoppingOperation(
              name=inner_op_name,
              status=vizier_oss_pb2.EarlyStoppingOperation.Status.ACTIVE,
              should_stop=False,
          )
          inner_operation.creation_time.CopyFrom(_get_current_time())
          self.datastore.create_early_stopping_operation(inner_operation)

        inner_operation.should_stop = early_stopping_decision.should_stop
        inner_operation.status = (
            vizier_oss_pb2.EarlyStoppingOperation.Status.DONE
        )
        inner_operation.completion_time.CopyFrom(_get_current_time())
        self.datastore.update_early_stopping_operation(inner_operation)

      # Operation to be outputted may have changed.
      output_operation = self.datastore.get_early_stopping_operation(
          output_operation.name
      )
      return vizier_service_pb2.CheckTrialEarlyStoppingStateResponse(
          should_stop=output_operation.should_stop
      )

  def StopTrial(
      self,
      request: vizier_service_pb2.StopTrialRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> study_pb2.Trial:
    """Sets the trial state to STOPPING.

    Args:
      request:
      context:

    Returns:
      The stopped Trial

    Raises:
      ImmutableStudyError: If study was already immutable.
      ImmutableTrialError: If the trial cannot be modified.
    """
    study_name = TrialResource.from_name(request.name).study_resource.name
    if self._study_is_immutable(study_name):
      e = custom_errors.ImmutableStudyError(
          'Study {} is immutable. Cannot stop trial.'.format(study_name)
      )
      grpc_util.handle_exception(e, context)

    with self._study_name_to_lock[study_name]:
      trial = self.datastore.get_trial(request.name)
      if trial.state == study_pb2.Trial.ACTIVE:
        trial.state = study_pb2.Trial.STOPPING
        self.datastore.update_trial(trial)
      elif trial.state in (study_pb2.Trial.STOPPING, study_pb2.Trial.SUCCEEDED):
        logging.warn(
            'Trial %s has state %s. StopTrial will be a no-op.',
            request.name,
            study_pb2.Trial.State.Name(trial.state),
        )
      else:
        e = custom_errors.ImmutableTrialError(
            'Trial {} has state {}. Raising Error.'.format(
                request.name, study_pb2.Trial.State.Name(trial.state)
            )
        )
        grpc_util.handle_exception(e, context)
    return trial

  def ListOptimalTrials(
      self,
      request: vizier_service_pb2.ListOptimalTrialsRequest,
      context: Optional[grpc.ServicerContext] = None,
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
    if not raw_trial_list:
      return vizier_service_pb2.ListOptimalTrialsResponse(optimal_trials=[])

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
      if (
          trial.state == study_pb2.Trial.State.SUCCEEDED
          and required_metric_ids.issubset(trial_metric_ids)
      ):
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

    if not considered_trials:
      return vizier_service_pb2.ListOptimalTrialsResponse(optimal_trials=[])

    # Find Pareto optimal trials.
    ys = np.array(considered_trial_objective_vectors)
    n = ys.shape[0]
    dominated = np.asarray([
        [np.all(ys[i] <= ys[j]) & np.any(ys[j] > ys[i]) for i in range(n)]
        for j in range(n)
    ])
    optimal_booleans = np.logical_not(np.any(dominated, axis=0))
    optimal_trials = []
    for i, boolean in enumerate(list(optimal_booleans)):
      if boolean:
        optimal_trials.append(considered_trials[i])

    return vizier_service_pb2.ListOptimalTrialsResponse(
        optimal_trials=optimal_trials
    )

  def UpdateMetadata(
      self,
      request: vizier_service_pb2.UpdateMetadataRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> vizier_service_pb2.UpdateMetadataResponse:
    """Stores the supplied metadata in the database."""
    if self._study_is_immutable(request.name):
      e = custom_errors.ImmutableStudyError(
          'Study {} is immutable. Cannot update metadata.'.format(request.name)
      )
      grpc_util.handle_exception(e, context)

    try:
      self.datastore.update_metadata(
          request.name,
          [x.metadatum for x in request.delta if not x.HasField('trial_id')],
          [x for x in request.delta if x.HasField('trial_id')],
      )
    except KeyError as e:
      return vizier_service_pb2.UpdateMetadataResponse(
          error_details=';'.join(e.args)
      )
    return vizier_service_pb2.UpdateMetadataResponse()
