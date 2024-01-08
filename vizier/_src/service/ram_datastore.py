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

"""Basic datastore written using Python primitives.

For debugging/testing purposes mainly.
"""
import collections
import copy
import dataclasses
import threading
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional
from absl import logging

from vizier._src.service import custom_errors
from vizier._src.service import datastore
from vizier._src.service import key_value_pb2
from vizier._src.service import resources
from vizier._src.service import study_pb2
from vizier._src.service import vizier_oss_pb2
from vizier._src.service import vizier_service_pb2
from vizier.service import pyvizier as vz

from google.longrunning import operations_pb2

UnitMetadataUpdate = vizier_service_pb2.UnitMetadataUpdate


@dataclasses.dataclass(frozen=True)
class ClientNode:
  """Only contains suggestion operations associated with this client."""

  # Keys are `operation_id`.
  suggestion_operations: Dict[str, operations_pb2.Operation] = (
      dataclasses.field(default_factory=dict)
  )


# Specific dataclasses used for NestedDictRAMDataStore.
@dataclasses.dataclass(frozen=True)
class StudyNode:
  """Contains original study and currently stored trials."""

  study_proto: study_pb2.Study

  # Keys are `trial_id`.
  trial_protos: Dict[int, study_pb2.Trial] = dataclasses.field(
      default_factory=dict
  )

  # Keys are `operation_id`.
  early_stopping_operations: Dict[
      str, vizier_oss_pb2.EarlyStoppingOperation
  ] = dataclasses.field(default_factory=dict)

  # Key is `client_id`, which distinguishes clients, so you can have several
  # clients interacting with the same Vizier server.
  clients: Dict[str, ClientNode] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class OwnerNode:
  """First level of the entire RAM Datastore."""

  # Key is a `study_id` that pick's out one of the owner's Studies.
  studies: Dict[str, StudyNode] = dataclasses.field(default_factory=dict)


class NestedDictRAMDataStore(datastore.DataStore):
  """Basic Datastore class using nested dictionaries."""

  def __init__(self):
    """Organization as follows."""
    # Key is `owner_id`, which corresponds to the (perhaps human) owner
    # of the Study.
    self._owners: Dict[str, OwnerNode] = {}
    # TODO: Use more fine-grained (study/trial/etc.) locks.
    self._lock = threading.Lock()

  def create_study(self, study: study_pb2.Study) -> resources.StudyResource:
    resource = resources.StudyResource.from_name(study.name)
    temp_dict = {resource.study_id: StudyNode(study_proto=copy.deepcopy(study))}

    with self._lock:
      if resource.owner_id not in self._owners:
        self._owners[resource.owner_id] = OwnerNode(studies=temp_dict)
      else:
        study_dict = self._owners[resource.owner_id].studies
        if resource.study_id not in study_dict:
          study_dict.update(temp_dict)
        else:
          raise custom_errors.AlreadyExistsError(
              'Study with that name already exists.', study.name
          )
    return resource

  def load_study(self, study_name: str) -> study_pb2.Study:
    resource = resources.StudyResource.from_name(study_name)
    try:
      with self._lock:
        return copy.deepcopy(
            self._owners[resource.owner_id]
            .studies[resource.study_id]
            .study_proto
        )
    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Could not get Study with name:', resource.name
      ) from err

  def update_study(self, study: study_pb2.Study) -> resources.StudyResource:
    resource = resources.StudyResource.from_name(study.name)
    try:
      with self._lock:
        self._owners[resource.owner_id].studies[
            resource.study_id
        ].study_proto.CopyFrom(study)
      return resource
    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Could not update Study with name:', resource.name
      ) from err

  def delete_study(self, study_name: str) -> None:
    resource = resources.StudyResource.from_name(study_name)
    try:
      with self._lock:
        del self._owners[resource.owner_id].studies[resource.study_id]
    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Study does not exist:', study_name
      ) from err

  def list_studies(self, owner_name: str) -> List[study_pb2.Study]:
    resource = resources.OwnerResource.from_name(owner_name)
    try:
      with self._lock:
        study_nodes = list(self._owners[resource.owner_id].studies.values())
        return copy.deepcopy(
            [study_node.study_proto for study_node in study_nodes]
        )
    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Owner does not exist:', owner_name
      ) from err

  def create_trial(self, trial: study_pb2.Trial) -> resources.TrialResource:
    resource = resources.TrialResource.from_name(trial.name)
    with self._lock:
      trial_protos = (
          self._owners[resource.owner_id]
          .studies[resource.study_id]
          .trial_protos
      )
      if resource.trial_id in trial_protos:
        raise custom_errors.AlreadyExistsError(
            'Trial %s already exists' % trial.name
        )
      else:
        trial_protos[resource.trial_id] = copy.deepcopy(trial)
    return resource

  def get_trial(self, trial_name: str) -> study_pb2.Trial:
    resource = resources.TrialResource.from_name(trial_name)
    try:
      with self._lock:
        return copy.deepcopy(
            self._owners[resource.owner_id]
            .studies[resource.study_id]
            .trial_protos[resource.trial_id]
        )
    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Could not get Trial with name:', resource.name
      ) from err

  def update_trial(self, trial: study_pb2.Trial) -> resources.TrialResource:
    resource = resources.TrialResource.from_name(trial.name)
    try:
      with self._lock:
        trial_protos = (
            self._owners[resource.owner_id]
            .studies[resource.study_id]
            .trial_protos
        )
        if resource.trial_id not in trial_protos:
          raise custom_errors.NotFoundError(
              'Trial %s does not exist.' % trial.name
          )
        trial_protos[resource.trial_id] = copy.deepcopy(trial)
      return resource
    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Could not update Trial with name:', resource.name
      ) from err

  def list_trials(self, study_name: str) -> List[study_pb2.Trial]:
    resource = resources.StudyResource.from_name(study_name)
    try:
      with self._lock:
        return copy.deepcopy(
            list(
                self._owners[resource.owner_id]
                .studies[resource.study_id]
                .trial_protos.values()
            )
        )
    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Study does not exist:', study_name
      ) from err

  def delete_trial(self, trial_name: str) -> None:
    resource = resources.TrialResource.from_name(trial_name)
    try:
      with self._lock:
        del (
            self._owners[resource.owner_id]
            .studies[resource.study_id]
            .trial_protos[resource.trial_id]
        )
    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Trial does not exist:', trial_name
      ) from err

  def max_trial_id(self, study_name: str) -> int:
    resource = resources.StudyResource.from_name(study_name)
    try:
      with self._lock:
        trial_ids = copy.deepcopy(
            list(
                self._owners[resource.owner_id]
                .studies[resource.study_id]
                .trial_protos.keys()
            )
        )
    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Study does not exist:', study_name
      ) from err

    if trial_ids:
      return max(trial_ids)
    else:
      return 0

  def create_suggestion_operation(
      self, operation: operations_pb2.Operation
  ) -> resources.SuggestionOperationResource:
    resource = resources.SuggestionOperationResource.from_name(operation.name)
    with self._lock:
      if (
          resource.client_id
          not in self._owners[resource.owner_id]
          .studies[resource.study_id]
          .clients
      ):
        self._owners[resource.owner_id].studies[resource.study_id].clients[
            resource.client_id
        ] = ClientNode()
      suggestion_operations = (
          self._owners[resource.owner_id]
          .studies[resource.study_id]
          .clients[resource.client_id]
          .suggestion_operations
      )

      if resource.operation_id in suggestion_operations:
        raise custom_errors.AlreadyExistsError(
            'Operation already exists:', resource.operation_id
        )

      suggestion_operations[resource.operation_id] = copy.deepcopy(operation)
    return resource

  def get_suggestion_operation(
      self, operation_name: str
  ) -> operations_pb2.Operation:
    resource = resources.SuggestionOperationResource.from_name(operation_name)
    try:
      with self._lock:
        return copy.deepcopy(
            self._owners[resource.owner_id]
            .studies[resource.study_id]
            .clients[resource.client_id]
            .suggestion_operations[resource.operation_id]
        )

    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Could not find SuggestionOperation with name:', resource.name
      ) from err

  def update_suggestion_operation(
      self, operation: operations_pb2.Operation
  ) -> resources.SuggestionOperationResource:
    resource = resources.SuggestionOperationResource.from_name(operation.name)
    try:
      with self._lock:
        self._owners[resource.owner_id].studies[resource.study_id].clients[
            resource.client_id
        ].suggestion_operations[resource.operation_id] = copy.deepcopy(
            operation
        )
      return resource
    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Could not update SuggestionOperation with name:', resource.name
      ) from err

  def list_suggestion_operations(
      self,
      study_name: str,
      client_id: str,
      filter_fn: Optional[Callable[[operations_pb2.Operation], bool]] = None,
  ) -> List[operations_pb2.Operation]:
    resource = resources.StudyResource.from_name(study_name)
    try:
      with self._lock:
        operations_list = copy.deepcopy(
            list(
                self._owners[resource.owner_id]
                .studies[resource.study_id]
                .clients[client_id]
                .suggestion_operations.values()
            )
        )
    except KeyError as err:
      raise custom_errors.NotFoundError(
          '(study_name, client_id) does not exist:', (study_name, client_id)
      ) from err

    if filter_fn is not None:
      return copy.deepcopy([op for op in operations_list if filter_fn(op)])
    else:
      return copy.deepcopy(operations_list)

  def max_suggestion_operation_number(
      self, study_name: str, client_id: str
  ) -> int:
    resource = resources.StudyResource.from_name(study_name)
    try:
      with self._lock:
        ops = (
            self._owners[resource.owner_id]
            .studies[resource.study_id]
            .clients[client_id]
            .suggestion_operations
        )
        return len(ops)
    except KeyError as err:
      raise custom_errors.NotFoundError(
          '(study_name, client_id) does not exist:', (study_name, client_id)
      ) from err

  def create_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    resource = resources.EarlyStoppingOperationResource.from_name(
        operation.name
    )
    with self._lock:
      early_stopping_ops = (
          self._owners[resource.owner_id]
          .studies[resource.study_id]
          .early_stopping_operations
      )
      if resource.operation_id in early_stopping_ops:
        raise custom_errors.AlreadyExistsError(
            'Operation already exists:', resource.operation_id
        )

      early_stopping_ops[resource.operation_id] = copy.deepcopy(operation)
    return resource

  def get_early_stopping_operation(
      self, operation_name: str
  ) -> vizier_oss_pb2.EarlyStoppingOperation:
    resource = resources.EarlyStoppingOperationResource.from_name(
        operation_name
    )
    try:
      with self._lock:
        return copy.deepcopy(
            self._owners[resource.owner_id]
            .studies[resource.study_id]
            .early_stopping_operations[resource.operation_id]
        )
    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Could not find EarlyStoppingOperation with name:', resource.name
      ) from err

  def update_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    resource = resources.EarlyStoppingOperationResource.from_name(
        operation.name
    )
    try:
      with self._lock:
        self._owners[resource.owner_id].studies[
            resource.study_id
        ].early_stopping_operations[resource.operation_id] = copy.deepcopy(
            operation
        )
      return resource
    except KeyError as err:
      raise custom_errors.NotFoundError(
          'Could not update EarlyStoppingOperation with name:', resource.name
      ) from err

  def update_metadata(
      self,
      study_name: str,
      study_metadata: Iterable[key_value_pb2.KeyValue],
      trial_metadata: Iterable[UnitMetadataUpdate],
  ) -> None:
    # TODO:
    """Writes the supplied metadata to the database.

    Args:
      study_name:
      study_metadata: Metadata that's associated with the Study as a whole.
      trial_metadata: Metadata that's associated with trials.  (Note that its an
        error to attach metadata to a Trial that doesn't exist.)
    """
    s_resource = resources.StudyResource.from_name(study_name)
    logging.debug('database.update_metadata s_resource= %s', s_resource)

    with self._lock:
      try:
        study_node = self._owners[s_resource.owner_id].studies[
            s_resource.study_id
        ]
      except KeyError as e:
        raise custom_errors.NotFoundError(
            'No such study:', s_resource.name
        ) from e
      # Store Study-related metadata into the database.
      vz.metadata_util.merge_study_metadata(
          study_node.study_proto.study_spec, copy.deepcopy(study_metadata)
      )
      # Split the trial-related metadata by Trial.
      split_metadata: DefaultDict[str, List[UnitMetadataUpdate]] = (
          collections.defaultdict(list)
      )
      for md in copy.deepcopy(trial_metadata):
        split_metadata[md.trial_id].append(md)
      # Now, we update one Trial at a time:
      for trial_id, md_list in split_metadata.items():
        t_resource = s_resource.trial_resource(trial_id)
        trial_proto = study_node.trial_protos[t_resource.trial_id]
        vz.metadata_util.merge_trial_metadata(trial_proto, md_list)
