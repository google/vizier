"""Partial progress implementation of SQL Datastore."""
from typing import Callable, Iterable, List, Optional
import sqlalchemy.v1_2 as sqla

from vizier.service import datastore
from vizier.service import key_value_pb2
from vizier.service import resources
from vizier.service import study_pb2
from vizier.service import vizier_oss_pb2
from google.longrunning import operations_pb2


# TODO: Finish off implementation of this class.
class SQLDataStore(datastore.DataStore):
  """SQL Datastore."""

  def __init__(self, engine):
    self._engine = engine
    self._connection = self._engine.connect()
    self._root_metadata = sqla.MetaData()
    self._studies_table = sqla.Table(
        'studies',
        self._root_metadata,
        sqla.Column('study_name', sqla.String, primary_key=True),
        sqla.Column('owner_id', sqla.String),
        sqla.Column('study_id', sqla.String),
        sqla.Column('serialized_study', sqla.String),
    )
    self._trials_table = sqla.Table(
        'trials',
        self._root_metadata,
        sqla.Column('trial_name', sqla.String, primary_key=True),
        sqla.Column('owner_id', sqla.String),
        sqla.Column('study_id', sqla.String),
        sqla.Column('trial_id', sqla.INTEGER),
        sqla.Column('serialized_trial', sqla.String),
    )
    self._suggestion_operations_table = sqla.Table(
        'suggestion_operations',
        self._root_metadata,
        sqla.Column('operation_name', sqla.String, primary_key=True),
        sqla.Column('owner_id', sqla.String),
        sqla.Column('client_id', sqla.String),
        sqla.Column('operation_number', sqla.INTEGER),
        sqla.Column('serialized_op', sqla.String),
    )
    self._early_stopping_operations_table = sqla.Table(
        'early_stopping_operations',
        self._root_metadata,
        sqla.Column('operation_name', sqla.String, primary_key=True),
        sqla.Column('owner_id', sqla.String),
        sqla.Column('study_id', sqla.String),
        sqla.Column('trial_id', sqla.INTEGER),
        sqla.Column('serialized_op', sqla.String),
    )
    self._root_metadata.create_all(self._engine)

  def create_study(self, study: study_pb2.Study) -> resources.StudyResource:
    """Creates study in the database. If already exists, raises ValueError."""
    study_resource = resources.StudyResource.from_name(study.name)
    query = self._studies_table.insert().values(
        study_name=study.name,
        owner_id=study_resource.owner_id,
        study_id=study_resource.study_id,
        serialized_study=study.SerializeToString())
    self._connection.execute(query)
    return study_resource

  def load_study(self, study_name: str) -> study_pb2.Study:
    """Loads a study from the database."""
    query = sqla.select([self._studies_table
                        ]).where(self._studies_table.c.study_name == study_name)
    result = self._connection.execute(query)
    row = result.fetchone()
    return study_pb2.Study.FromString(row['serialized_study'])

  def delete_study(self, study_name: str) -> None:
    """Deletes study from database. If Study doesn't exist, raises KeyError."""
    query = self._studies_table.delete().where(
        self._studies_table.c.study_name == study_name)
    self._connection.execute(query)

    study_resource = resources.StudyResource.from_name(study_name)
    trial_query = self._trials_table.delete().where(
        self._trials_table.c.owner_id == study_resource.owner_id).where(
            self._trials_table.c.study_id == study_resource.study_id)
    self._connection.execute(trial_query)

  def list_studies(self, owner_name: str) -> List[study_pb2.Study]:
    """Lists all studies under a given owner."""
    owner_resource = resources.OwnerResource.from_name(owner_name)
    query = sqla.select([
        self._studies_table
    ]).where(self._studies_table.c.owner_id == owner_resource.owner_id)
    result = self._connection.execute(query)
    return [
        study_pb2.Study.FromString(row['serialized_study']) for row in result
    ]

  def create_trial(self, trial: study_pb2.Trial) -> resources.TrialResource:
    """Stores trial in database."""
    trial_resource = resources.TrialResource.from_name(trial.name)
    query = self._trials_table.insert().values(
        trial_name=trial.name,
        owner_id=trial_resource.owner_id,
        study_id=trial_resource.study_id,
        trial_id=trial_resource.trial_id,
        serialized_trial=trial.SerializeToString())
    self._connection.execute(query)
    return trial_resource

  def get_trial(self, trial_name: str) -> study_pb2.Trial:
    """Retrieves trial from database."""
    query = sqla.select([self._trials_table
                        ]).where(self._trials_table.c.trial_name == trial_name)
    result = self._connection.execute(query)
    row = result.fetchone()
    return study_pb2.Trial.FromString(row['serialized_trial'])

  def list_trials(self, study_name: str) -> List[study_pb2.Trial]:
    """List all trials given a study."""
    study_resource = resources.StudyResource.from_name(study_name)
    query = sqla.select([
        self._trials_table
    ]).where(self._trials_table.c.owner_id == study_resource.owner_id).where(
        self._trials_table.c.study_id == study_resource.study_id)
    result = self._connection.execute(query)
    return [
        study_pb2.Trial.FromString(row['serialized_trial']) for row in result
    ]

  def delete_trial(self, trial_name: str) -> None:
    """Deletes trial from database."""
    query = self._trials_table.delete().where(
        self._trials_table.c.trial_name == trial_name)
    self._connection.execute(query)

  def max_trial_id(self, study_name: str) -> int:
    """Maximal trial ID in study. Returns 0 if no trials exist."""
    study_resource = resources.StudyResource.from_name(study_name)
    query = sqla.select([
        sqla.func.max(self._trials_table.c.trial_id, type_=sqla.INT)
    ]).where(self._trials_table.c.owner_id == study_resource.owner_id).where(
        self._trials_table.c.study_id == study_resource.study_id)
    return self._connection.execute(query).fetchone()[0]

  def create_suggestion_operation(
      self, operation: operations_pb2.Operation
  ) -> resources.SuggestionOperationResource:
    """Stores suggestion operation."""
    resource = resources.SuggestionOperationResource.from_name(operation.name)
    query = self._suggestion_operations_table.insert().values(
        operation_name=operation.name,
        owner_id=resource.owner_id,
        client_id=resource.client_id,
        operation_number=resource.operation_number,
        serialized_op=operation.SerializeToString())
    self._connection.execute(query)
    return resource

  def get_suggestion_operation(self,
                               operation_name: str) -> operations_pb2.Operation:
    """Retrieves suggestion operation."""
    query = sqla.select([self._suggestion_operations_table]).where(
        self._suggestion_operations_table.c.operation_name == operation_name)
    result = self._connection.execute(query)
    row = result.fetchone()
    return operations_pb2.Operation.FromString(row['serialized_op'])

  def list_suggestion_operations(
      self,
      owner_name: str,
      client_id: str,
      filter_fn: Optional[Callable[[operations_pb2.Operation], bool]] = None
  ) -> List[operations_pb2.Operation]:
    """Retrieve all suggestion operations from a client."""
    owner_resource = resources.OwnerResource.from_name(owner_name)
    query = sqla.select([
        self._suggestion_operations_table
    ]).where(self._suggestion_operations_table.c.owner_id ==
             owner_resource.owner_id).where(
                 self._suggestion_operations_table.c.client_id == client_id)
    result = self._connection.execute(query)
    all_ops = [
        operations_pb2.Operation.FromString(row['serialized_op'])
        for row in result
    ]
    if filter_fn is not None:
      output_list = []
      for op in all_ops:
        if filter_fn(op):
          output_list.append(op)
      return output_list
    else:
      return all_ops

  def max_suggestion_operation_number(self, owner_name: str,
                                      client_id: str) -> int:
    """Maximal suggestion number for a given client."""
    resource = resources.OwnerResource.from_name(owner_name)
    query = sqla.select([
        sqla.func.max(
            self._suggestion_operations_table.c.operation_number,
            type_=sqla.INT)
    ]).where(self._suggestion_operations_table.c.owner_id == resource.owner_id
            ).where(self._suggestion_operations_table.c.client_id == client_id)
    return self._connection.execute(query).fetchone()[0]

  def create_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    """Stores early stopping operation."""
    resource = resources.EarlyStoppingOperationResource.from_name(
        operation.name)
    query = self._early_stopping_operations_table.insert().values(
        operation_name=operation.name,
        owner_id=resource.owner_id,
        study_id=resource.study_id,
        trial_id=resource.trial_id,
        serialized_op=operation.SerializeToString())
    self._connection.execute(query)
    return resource

  def get_early_stopping_operation(
      self, operation_name: str) -> vizier_oss_pb2.EarlyStoppingOperation:
    """Retrieves early stopping operation."""
    query = sqla.select([
        self._early_stopping_operations_table
    ]).where(self._early_stopping_operations_table.c.operation_name ==
             operation_name)
    result = self._connection.execute(query)
    row = result.fetchone()
    return vizier_oss_pb2.EarlyStoppingOperation.FromString(
        row['serialized_op'])

  def update_metadata(
      self,
      study_name: str,
      study_metadata: Iterable[key_value_pb2.KeyValue],
      trial_metadata: Iterable[datastore._KeyValuePlus],  # pylint:disable=protected-access
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
    raise NotImplementedError()
