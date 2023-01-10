# Copyright 2023 Google LLC.
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

"""Contains all datastore classes for saving study + trial data.

See resources.py for naming conventions.
"""
import abc
import collections
import copy
import dataclasses
import threading
from typing import Callable, DefaultDict, Dict, Iterable, List, Optional, Tuple
from absl import logging

from vizier.service import custom_errors
from vizier.service import key_value_pb2
from vizier.service import resources
from vizier.service import study_pb2
from vizier.service import vizier_oss_pb2
from vizier.service import vizier_service_pb2
from google.longrunning import operations_pb2

UnitMetadataUpdate = vizier_service_pb2.UnitMetadataUpdate


class DataStore(abc.ABC):
  """Abstract class for data storage.

  Input/Outputs should always be pass-by-value.
  """

  @abc.abstractmethod
  def create_study(self, study: study_pb2.Study) -> resources.StudyResource:
    """Creates study in database. If preexisting, raises AlreadyExistsError."""

  @abc.abstractmethod
  def load_study(self, study_name: str) -> study_pb2.Study:
    """Loads study from the database. If nonexistent, raises NotFoundError."""

  @abc.abstractmethod
  def update_study(self, study: study_pb2.Study) -> resources.StudyResource:
    """Updates pre-existing study. If nonexistent, raises NotFoundError."""

  @abc.abstractmethod
  def delete_study(self, study_name: str) -> None:
    """Deletes study from database. If nonexistent, raises NotFoundError."""

  @abc.abstractmethod
  def list_studies(self, owner_name: str) -> List[study_pb2.Study]:
    """Lists all studies under owner.

    Args:
      owner_name: Name of owner.

    Returns:
      List of Studies associated with owner.

    Raises:
      NotFoundError: If the owner does not exist.
    """

  @abc.abstractmethod
  def create_trial(self, trial: study_pb2.Trial) -> resources.TrialResource:
    """Stores trial in database.

    Args:
      trial: Trial to be stored.

    Returns:
      The newly created trial resource name.

    Raises:
      AlreadyExistsError: If trial already exists.
    """

  @abc.abstractmethod
  def get_trial(self, trial_name: str) -> study_pb2.Trial:
    """Retrieves trial from database. If nonexistent, raises NotFoundError."""

  @abc.abstractmethod
  def update_trial(self, trial: study_pb2.Trial) -> resources.TrialResource:
    """Updates pre-existing trial. If nonexistent, raises NotFoundError."""

  @abc.abstractmethod
  def list_trials(self, study_name: str) -> List[study_pb2.Trial]:
    """List all trials for study. If study nonexistent, raises NotFoundError."""

  @abc.abstractmethod
  def delete_trial(self, trial_name: str) -> None:
    """Deletes trial from database. If nonexistent, raises NotFoundError."""

  @abc.abstractmethod
  def max_trial_id(self, study_name: str) -> int:
    """Maximal trial ID in study (defaults to 0 if no trials exist).

    Args:
      study_name: Name of study.

    Returns:
      Maximum trial ID as an int.

    Raises:
      NotFoundError: If the study does not exist.
    """

  @abc.abstractmethod
  def create_suggestion_operation(
      self, operation: operations_pb2.Operation
  ) -> resources.SuggestionOperationResource:
    """Stores suggestion operation.

    Args:
      operation:

    Returns:
      Resource for created op.

    Raises:
      AlreadyExistsError: If the suggest op already exists.
    """

  @abc.abstractmethod
  def get_suggestion_operation(
      self, operation_name: str
  ) -> operations_pb2.Operation:
    """Retrieves suggestion operation. If nonexistent, raises NotFoundError."""

  @abc.abstractmethod
  def update_suggestion_operation(
      self, operation: operations_pb2.Operation
  ) -> resources.SuggestionOperationResource:
    """Updates pre-existing suggestion op.

    Args:
      operation: New suggestion speration.

    Returns:
      Resource to operation.

    Raises:
      NotFoundError if suggest op is nonexistent.
    """

  @abc.abstractmethod
  def list_suggestion_operations(
      self,
      study_name: str,
      client_id: str,
      filter_fn: Optional[Callable[[operations_pb2.Operation], bool]] = None,
  ) -> List[operations_pb2.Operation]:
    """Retrieve all suggestion op from client.

    Args:
      study_name: Associated study for the suggest op.
      client_id: Associated client for the suggest op.
      filter_fn: Optional function to filter out the suggest ops.

    Raises:
      NotFoundError: If study or client is nonexistent.
    """

  @abc.abstractmethod
  def max_suggestion_operation_number(
      self, study_name: str, client_id: str
  ) -> int:
    """Maximal suggestion number for given client.

    Args:
      study_name: Name of associated owners.
      client_id: ID of associated client.

    Returns:
      Maximum suggest op number as an int.

    Raises:
      NotFoundError: If owner or client is nonexistent.
    """

  @abc.abstractmethod
  def create_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    """Stores early stopping operation.

    Args:
      operation: Operation to be stored.

    Returns:
      The newly created op resource name.

    Raises:
      AlreadyExistsError: If the suggest op already exists.
    """

  @abc.abstractmethod
  def get_early_stopping_operation(
      self, operation_name: str
  ) -> vizier_oss_pb2.EarlyStoppingOperation:
    """Retrieves early stopping op. If nonexistent, raises NotFoundError."""

  @abc.abstractmethod
  def update_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    """Updates preexisting early stopping op.

    Args:
      operation: New version of EarlyStoppingOp.

    Returns:
      Resource to the EarlyStoppingOp.

    Raises:
      NotFoundError: If early stopping op is nonexistent.
    """

  # TODO: Simplify the API by taking MetadataUpdateRequest proto
  # as input.
  @abc.abstractmethod
  def update_metadata(
      self,
      study_name: str,
      study_metadata: Iterable[key_value_pb2.KeyValue],
      trial_metadata: Iterable[UnitMetadataUpdate],
  ) -> None:
    """Store the supplied metadata in the database.

    Args:
      study_name: (Typically derived from a StudyResource.)
      study_metadata: Metadata to attach to the Study as a whole.
      trial_metadata: Metadata to attach to Trials.

    Raises:
      NotFoundError: If the update fails because of an attempt to attach
        metadata to a nonexistant Trial.
    """


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


def merge_study_metadata(
    study_spec: study_pb2.StudySpec,
    new_metadata: Iterable[key_value_pb2.KeyValue],
) -> None:
  """Merges $new_metadata into a Study's existing metadata."""
  metadata_dict: Dict[Tuple[str, str], key_value_pb2.KeyValue] = {}
  for kv in study_spec.metadata:
    metadata_dict[(kv.ns, kv.key)] = kv
  for kv in new_metadata:
    metadata_dict[(kv.ns, kv.key)] = kv
  study_spec.ClearField('metadata')
  study_spec.metadata.extend(
      sorted(metadata_dict.values(), key=lambda kv: (kv.ns, kv.key))
  )


def merge_trial_metadata(
    trial_proto: study_pb2.Trial, new_metadata: Iterable[UnitMetadataUpdate]
) -> None:
  """Merges $new_metadata into a Trial's existing metadata.

  Args:
    trial_proto: A representation of a Trial; this will be modified.
    new_metadata: Metadata that will add or update metadata in the Trial.
  NOTE: the metadata updates in $new_metadata should have the same ID as
    $trial_proto.
  """
  metadata_dict: Dict[Tuple[str, str], key_value_pb2.KeyValue] = {}
  for kv in trial_proto.metadata:
    metadata_dict[(kv.ns, kv.key)] = kv
  for md_update in new_metadata:
    if md_update.trial_id == trial_proto.id:
      metadata_dict[(md_update.metadatum.ns, md_update.metadatum.key)] = (
          md_update.metadatum
      )
    else:
      logging.warning(
          'Metadata associated with wrong trial: %s instead of %s',
          md_update.trial_id,
          trial_proto.id,
      )
  trial_proto.ClearField('metadata')
  trial_proto.metadata.extend(
      sorted(metadata_dict.values(), key=lambda kv: (kv.ns, kv.key))
  )


class NestedDictRAMDataStore(DataStore):
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
      merge_study_metadata(
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
        merge_trial_metadata(trial_proto, md_list)
