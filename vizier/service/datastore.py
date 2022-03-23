"""Contains all datastore classes for saving study + trial data.

See resources.py for naming conventions.
"""
import abc
import dataclasses
import logging
from typing import Callable, Dict, Iterable, List, Optional

from vizier.service import key_value_pb2
from vizier.service import resources
from vizier.service import study_pb2
from vizier.service import vizier_oss_pb2
from vizier.service import vizier_service_pb2
from google.longrunning import operations_pb2


_KeyValuePlus = vizier_service_pb2.UpdateMetadataRequest.KeyValuePlus


class DataStore(abc.ABC):
  """Abstract class for data storing."""

  @abc.abstractmethod
  def create_study(self, study: study_pb2.Study) -> resources.StudyResource:
    """Creates study in the database. If already exists, raises ValueError."""

  @abc.abstractmethod
  def load_study(self, study_name: str) -> study_pb2.Study:
    """Loads a study from the database."""

  @abc.abstractmethod
  def delete_study(self, study_name: str) -> None:
    """Deletes study from database. If Study doesn't exist, raises KeyError."""

  @abc.abstractmethod
  def list_studies(self, owner_name: str) -> List[study_pb2.Study]:
    """Lists all studies under a given owner."""

  @abc.abstractmethod
  def create_trial(self, trial: study_pb2.Trial) -> resources.TrialResource:
    """Stores trial in database."""

  @abc.abstractmethod
  def get_trial(self, trial_name: str) -> study_pb2.Trial:
    """Retrieves trial from database."""

  @abc.abstractmethod
  def list_trials(self, study_name: str) -> List[study_pb2.Trial]:
    """List all trials given a study."""

  @abc.abstractmethod
  def delete_trial(self, trial_name: str) -> None:
    """Deletes trial from database."""

  @abc.abstractmethod
  def max_trial_id(self, study_name: str) -> int:
    """Maximal trial ID in study. Returns 0 if no trials exist."""

  @abc.abstractmethod
  def create_suggestion_operation(
      self, operation: operations_pb2.Operation
  ) -> resources.SuggestionOperationResource:
    """Stores suggestion operation."""

  @abc.abstractmethod
  def create_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    """Stores early stopping operation."""

  @abc.abstractmethod
  def get_suggestion_operation(self,
                               operation_name: str) -> operations_pb2.Operation:
    """Retrieves suggestion operation."""

  @abc.abstractmethod
  def get_early_stopping_operation(
      self, operation_name: str) -> vizier_oss_pb2.EarlyStoppingOperation:
    """Retrieves early stopping operation."""

  @abc.abstractmethod
  def list_suggestion_operations(
      self,
      owner_name: str,
      client_id: str,
      filter_fn: Optional[Callable[[operations_pb2.Operation], bool]] = None
  ) -> List[operations_pb2.Operation]:
    """Retrieve all suggestion operations from a client."""

  @abc.abstractmethod
  def max_suggestion_operation_number(self, owner_name: str,
                                      client_id: str) -> int:
    """Maximal suggestion number for a given client."""

  @abc.abstractmethod
  def update_metadata(
      self,
      study_name: str,
      study_metadata: Iterable[key_value_pb2.KeyValue],
      trial_metadata: Iterable[_KeyValuePlus],
  ) -> None:
    """Store the supplied metadata in the database.

    Args:
        study_name: (Typically derived from a StudyResource.)
        study_metadata: Metadata to attach to the Study as a whole.
        trial_metadata: Metadata to attach to Trials.

    Raises:
      KeyError: if the update fails because of an attempt to attach metadata to
        a nonexistant Trial.
    """


# Specific dataclasses used for NestedDictRAMDataStore.
@dataclasses.dataclass(frozen=True)
class StudyNode:
  """Contains original study and currently stored trials."""
  study_proto: study_pb2.Study

  # Keys are `trial_id`.
  trial_protos: Dict[int,
                     study_pb2.Trial] = dataclasses.field(default_factory=dict)

  # Keys are `operation_id`.
  early_stopping_operations: Dict[
      str, vizier_oss_pb2.EarlyStoppingOperation] = dataclasses.field(
          default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ClientNode:
  """Only contains suggestion operations associated with this client."""
  # Keys are `operation_id`.
  suggestion_operations: Dict[str,
                              operations_pb2.Operation] = dataclasses.field(
                                  default_factory=dict)


@dataclasses.dataclass(frozen=True)
class OwnerNode:
  """First level of the entire RAM Datastore."""
  # Key is a `study_id` that pick's out one of the owner's Studies.
  studies: Dict[str, StudyNode] = dataclasses.field(default_factory=dict)
  # Key is `client_id`, which distinguishes clients, so you can have several
  # clients interacting with the same Vizier server.
  clients: Dict[str, ClientNode] = dataclasses.field(default_factory=dict)


class NestedDictRAMDataStore(DataStore):
  """Basic Datastore class using nested dictionaries."""

  def __init__(self):
    """Organization as follows."""
    # Key is `owner_id`, which corresponds to the (perhaps human) owner
    # of the Study.
    self._owners: Dict[str, OwnerNode] = {}

  def create_study(self, study: study_pb2.Study) -> resources.StudyResource:
    resource = resources.StudyResource.from_name(study.name)
    temp_dict = {resource.study_id: StudyNode(study_proto=study)}

    if resource.owner_id not in self._owners:
      self._owners[resource.owner_id] = OwnerNode(studies=temp_dict)
    else:
      study_dict = self._owners[resource.owner_id].studies
      if resource.study_id not in study_dict:
        study_dict.update(temp_dict)
      else:
        raise ValueError('Study with that name already exists.', study.name)
    return resource

  def load_study(self, study_name: str) -> study_pb2.Study:
    resource = resources.StudyResource.from_name(study_name)
    return self._owners[resource.owner_id].studies[
        resource.study_id].study_proto

  def delete_study(self, study_name: str) -> None:
    resource = resources.StudyResource.from_name(study_name)
    del self._owners[resource.owner_id].studies[resource.study_id]

  def list_studies(self, owner_name: str) -> List[study_pb2.Study]:
    resource = resources.OwnerResource.from_name(owner_name)
    study_nodes = list(self._owners[resource.owner_id].studies.values())
    return [study_node.study_proto for study_node in study_nodes]

  def create_trial(self, trial: study_pb2.Trial) -> resources.TrialResource:
    resource = resources.TrialResource.from_name(trial.name)
    self._owners[resource.owner_id].studies[resource.study_id].trial_protos[
        resource.trial_id] = trial
    return resource

  def get_trial(self, trial_name: str) -> study_pb2.Trial:
    resource = resources.TrialResource.from_name(trial_name)
    return self._owners[resource.owner_id].studies[
        resource.study_id].trial_protos[resource.trial_id]

  def list_trials(self, study_name: str) -> List[study_pb2.Trial]:
    resource = resources.StudyResource.from_name(study_name)
    return list(self._owners[resource.owner_id].studies[
        resource.study_id].trial_protos.values())

  def update_metadata(self, study_name: str,
                      study_metadata: Iterable[key_value_pb2.KeyValue],
                      trial_metadata: Iterable[_KeyValuePlus]) -> None:
    """Writes the supplied metadata to the database.

    Args:
      study_name:
      study_metadata: Metadata that's associated with the Study as a whole.
      trial_metadata: Metadata that's associated with trials.  (Note that its an
        error to attach metadata to a Trial that doesn't exist.)
    """
    s_resource = resources.StudyResource.from_name(study_name)
    logging.debug('database.update_metadata s_resource= %s', s_resource)
    try:
      study = self._owners[s_resource.owner_id].studies[s_resource.study_id]
    except KeyError as e:
      raise KeyError('No such study:', s_resource.name) from e
    # Store Study-related metadata into the database.
    study.study_proto.study_spec.ClearField('metadata')
    for metadata in study_metadata:
      study.study_proto.study_spec.metadata.append(metadata)
    # Store trial-related metadata in the database.  We first create a table of
    # the relevant `trial_resources` that will be touched.   We clear them, then
    # loop through the metadata, converting to protos.
    trial_resources: Dict[str, resources.TrialResource] = {}
    for metadata in trial_metadata:
      try:
        t_resource = trial_resources[metadata.trial_id]
      except KeyError:
        # If we don't have a t_resource entry already, create one and clear the
        # relevant Trial's metadata.
        t_resource = s_resource.trial_resource(metadata.trial_id)
        trial_resources[metadata.trial_id] = t_resource
        try:
          study.trial_protos[t_resource.trial_id].ClearField('metadata')
        except KeyError as e:
          raise KeyError(f'No such trial ({metadata.trial_id}):',
                         t_resource.name) from e
      study.trial_protos[t_resource.trial_id].metadata.append(metadata.k_v)

  def delete_trial(self, trial_name: str) -> None:
    resource = resources.TrialResource.from_name(trial_name)
    del self._owners[resource.owner_id].studies[resource.study_id].trial_protos[
        resource.trial_id]

  def max_trial_id(self, study_name: str) -> int:
    resource = resources.StudyResource.from_name(study_name)
    trial_ids = list(self._owners[resource.owner_id].studies[
        resource.study_id].trial_protos.keys())
    if trial_ids:
      return max(trial_ids)
    else:
      return 0

  def create_suggestion_operation(
      self, operation: operations_pb2.Operation
  ) -> resources.SuggestionOperationResource:
    resource = resources.SuggestionOperationResource.from_name(operation.name)
    if resource.client_id not in self._owners[resource.owner_id].clients:
      self._owners[resource.owner_id].clients[resource.client_id] = ClientNode()

    suggestion_operations = self._owners[resource.owner_id].clients[
        resource.client_id].suggestion_operations
    if resource.operation_id in suggestion_operations:
      raise ValueError('Operation already exists:', resource.operation_id)

    suggestion_operations[resource.operation_id] = operation
    return resource

  def create_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    resource = resources.EarlyStoppingOperationResource.from_name(
        operation.name)
    self._owners[resource.owner_id].studies[
        resource.study_id].early_stopping_operations[
            resource.operation_id] = operation
    return resource

  def get_suggestion_operation(self,
                               operation_name: str) -> operations_pb2.Operation:
    resource = resources.SuggestionOperationResource.from_name(operation_name)
    try:
      return self._owners[resource.owner_id].clients[
          resource.client_id].suggestion_operations[resource.operation_id]

    except KeyError as err:
      raise KeyError('Could not find SuggestionOperation with name:',
                     resource.name) from err

  def get_early_stopping_operation(
      self, operation_name: str) -> vizier_oss_pb2.EarlyStoppingOperation:
    resource = resources.EarlyStoppingOperationResource.from_name(
        operation_name)
    try:
      return self._owners[resource.owner_id].studies[
          resource.study_id].early_stopping_operations[resource.operation_id]
    except KeyError as err:
      raise KeyError('Could not find EarlyStoppingOperation with name:',
                     resource.name) from err

  def list_suggestion_operations(
      self,
      owner_name: str,
      client_id: str,
      filter_fn: Optional[Callable[[operations_pb2.Operation], bool]] = None
  ) -> List[operations_pb2.Operation]:
    resource = resources.OwnerResource.from_name(owner_name)
    if client_id not in self._owners[resource.owner_id].clients:
      return []
    else:
      operations_list = list(self._owners[
          resource.owner_id].clients[client_id].suggestion_operations.values())
      if filter_fn is not None:
        output_list = []
        for operation in operations_list:
          if filter_fn(operation):
            output_list.append(operation)
        return output_list
      else:
        return operations_list

  def max_suggestion_operation_number(self, owner_name: str,
                                      client_id: str) -> int:
    resource = resources.OwnerResource.from_name(owner_name)
    if client_id not in self._owners[resource.owner_id].clients:
      return 0
    else:
      return len(self._owners[
          resource.owner_id].clients[client_id].suggestion_operations)
