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

"""Contains datastore abstraction, used for saving study + trial data.

See resources.py for naming conventions.
"""
import abc
from typing import Callable, Iterable, List, Optional

from vizier._src.service import key_value_pb2
from vizier._src.service import resources
from vizier._src.service import study_pb2
from vizier._src.service import vizier_oss_pb2
from vizier._src.service import vizier_service_pb2
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
        metadata to a nonexistent Trial.
    """
