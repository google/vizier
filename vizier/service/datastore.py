"""Contains all datastore classes for saving study + trial data.

See resource_util.py for naming conventions.
"""
import abc
import dataclasses
from typing import Callable, Dict, List, Optional

from vizier.service import resources
from vizier.service import study_pb2
from vizier.service import vizier_oss_pb2
from google.longrunning import operations_pb2


class DataStore(abc.ABC):
  """Abstract class for data storing."""

  @abc.abstractmethod
  def create_study(self, study: study_pb2.Study) -> resources.StudyResource:
    """Creates study in the database. If already exists, raises ValueError."""
    pass

  @abc.abstractmethod
  def load_study(self, study_name: str) -> study_pb2.Study:
    """Loads a study from the database."""
    pass

  @abc.abstractmethod
  def delete_study(self, study_name: str) -> None:
    """Deletes study from database. If Study doesn't exist, raises KeyError."""
    pass

  @abc.abstractmethod
  def list_studies(self, owner_name: str) -> List[study_pb2.Study]:
    """Lists all studies under a given owner."""
    pass

  @abc.abstractmethod
  def create_trial(self, trial: study_pb2.Trial) -> resources.TrialResource:
    """Stores trial in database."""
    pass

  @abc.abstractmethod
  def get_trial(self, trial_name: str) -> study_pb2.Trial:
    """Retrieves trial from database."""
    pass

  @abc.abstractmethod
  def list_trials(self, study_name: str) -> List[study_pb2.Trial]:
    """List all trials given a study."""
    pass

  @abc.abstractmethod
  def delete_trial(self, trial_name: str) -> None:
    """Deletes trial from database."""
    pass

  @abc.abstractmethod
  def max_trial_id(self, study_name: str) -> int:
    """Maximal trial ID in study. Returns 0 if no trials exist."""
    pass

  @abc.abstractmethod
  def create_suggestion_operation(
      self, operation: operations_pb2.Operation
  ) -> resources.SuggestionOperationResource:
    """Stores suggestion operation."""
    pass

  @abc.abstractmethod
  def create_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    """Stores early stopping operation."""
    pass

  @abc.abstractmethod
  def get_suggestion_operation(self,
                               operation_name: str) -> operations_pb2.Operation:
    """Retrieves suggestion operation."""
    pass

  @abc.abstractmethod
  def get_early_stopping_operation(
      self, operation_name: str) -> vizier_oss_pb2.EarlyStoppingOperation:
    """Retrieves early stopping operation."""
    pass

  @abc.abstractmethod
  def list_suggestion_operations(
      self,
      owner_name: str,
      client_id: str,
      filter_fn: Optional[Callable[[operations_pb2.Operation], bool]] = None
  ) -> List[operations_pb2.Operation]:
    """Retrieve all suggestion operations from a client."""
    pass

  @abc.abstractmethod
  def max_suggestion_operation_number(self, owner_name: str,
                                      client_id: str) -> int:
    """Maximal suggestion number for a given client."""
    pass


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
  # Keys are `owner_id`.
  studies: Dict[str, StudyNode] = dataclasses.field(default_factory=dict)
  # Keys are `client_id`.
  clients: Dict[str, ClientNode] = dataclasses.field(default_factory=dict)


class NestedDictRAMDataStore(DataStore):
  """Basic Datastore class using nested dictionaries."""

  def __init__(self):
    """Organization as follows."""
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
        raise ValueError('Study with that name already exists.')
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
      raise ValueError(f'Operation {resource.operation_id} already exists.')

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
      raise KeyError(
          f'Could not find SuggestionOperation using resource with full name {resource.name}.'
      ) from err

  def get_early_stopping_operation(
      self, operation_name: str) -> vizier_oss_pb2.EarlyStoppingOperation:
    resource = resources.EarlyStoppingOperationResource.from_name(
        operation_name)
    try:
      return self._owners[resource.owner_id].studies[
          resource.study_id].early_stopping_operations[resource.operation_id]
    except KeyError as err:
      raise KeyError(
          f'Could not find EarlyStoppingOperation using resource with full name {resource.name}.'
      ) from err

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
