"""Implementation of SQL Datastore."""
from typing import Callable, Dict, Iterable, List, Optional
from absl import logging

import sqlalchemy as sqla

from vizier.service import datastore
from vizier.service import key_value_pb2
from vizier.service import resources
from vizier.service import study_pb2
from vizier.service import vizier_oss_pb2
from google.longrunning import operations_pb2


# TODO: Consider using ORM API (when fixed) to reduce code length.
class SQLDataStore(datastore.DataStore):
  """SQL Datastore."""

  def __init__(self, engine):
    self._engine = engine
    self._connection = self._engine.connect()
    self._root_metadata = sqla.MetaData()
    self._owners_table = sqla.Table(
        'owners',
        self._root_metadata,
        sqla.Column('owner_name', sqla.String, primary_key=True),
    )
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
    study_resource = resources.StudyResource.from_name(study.name)
    owner_name = study_resource.owner_resource.name
    owner_query = self._owners_table.insert().values(owner_name=owner_name)
    try:
      self._connection.execute(owner_query)
    except sqla.exc.IntegrityError:
      logging.info('Owner with name %s currently exists.', owner_name)

    query = self._studies_table.insert().values(
        study_name=study.name,
        owner_id=study_resource.owner_id,
        study_id=study_resource.study_id,
        serialized_study=study.SerializeToString())
    try:
      self._connection.execute(query)
      return study_resource
    except sqla.exc.IntegrityError as integrity_error:
      raise datastore.AlreadyExistsError('Study with name %s already exists.' %
                                         study.name) from integrity_error

  def load_study(self, study_name: str) -> study_pb2.Study:
    query = sqla.select([self._studies_table])
    query = query.where(self._studies_table.c.study_name == study_name)
    result = self._connection.execute(query)
    row = result.fetchone()
    if not row:
      raise datastore.NotFoundError('Failed to find study name: %s' %
                                    study_name)

    return study_pb2.Study.FromString(row['serialized_study'])

  def delete_study(self, study_name: str) -> None:
    study_resource = resources.StudyResource.from_name(study_name)

    exists_query = sqla.select([self._studies_table])
    exists_query = exists_query.where(
        self._studies_table.c.study_name == study_name)
    exists_query = sqla.exists(exists_query).select()
    exists = self._connection.execute(exists_query).fetchone()[0]
    if not exists:
      raise datastore.NotFoundError('Study %s does not exist.' % study_name)

    query = self._studies_table.delete().where(
        self._studies_table.c.study_name == study_name)
    self._connection.execute(query)

    trial_query = self._trials_table.delete().where(
        self._trials_table.c.owner_id == study_resource.owner_id).where(
            self._trials_table.c.study_id == study_resource.study_id)
    self._connection.execute(trial_query)

  def list_studies(self, owner_name: str) -> List[study_pb2.Study]:
    owner_id = resources.OwnerResource.from_name(owner_name).owner_id

    exists_query = sqla.select([self._owners_table])
    exists_query = exists_query.where(
        self._owners_table.c.owner_name == owner_name)
    exists_query = sqla.exists(exists_query).select()
    exists = self._connection.execute(exists_query).fetchone()[0]
    if not exists:
      raise datastore.NotFoundError('Owner name %s does not exist.' %
                                    owner_name)

    query = sqla.select([self._studies_table])
    query = query.where(self._studies_table.c.owner_id == owner_id)
    result = self._connection.execute(query).fetchall()
    return [
        study_pb2.Study.FromString(row['serialized_study']) for row in result
    ]

  def create_trial(self, trial: study_pb2.Trial) -> resources.TrialResource:
    trial_resource = resources.TrialResource.from_name(trial.name)
    query = self._trials_table.insert().values(
        trial_name=trial.name,
        owner_id=trial_resource.owner_id,
        study_id=trial_resource.study_id,
        trial_id=trial_resource.trial_id,
        serialized_trial=trial.SerializeToString())
    try:
      self._connection.execute(query)
      return trial_resource
    except sqla.exc.IntegrityError as integrity_error:
      raise datastore.AlreadyExistsError('Trial with name %s already exists.' %
                                         trial.name) from integrity_error

  def get_trial(self, trial_name: str) -> study_pb2.Trial:
    query = sqla.select([self._trials_table])
    query = query.where(self._trials_table.c.trial_name == trial_name)
    result = self._connection.execute(query)
    row = result.fetchone()

    if not row:
      raise datastore.NotFoundError('Failed to find trial name: %s' %
                                    trial_name)

    return study_pb2.Trial.FromString(row['serialized_trial'])

  def update_trial(self, trial: study_pb2.Trial) -> resources.TrialResource:
    trial_resource = resources.TrialResource.from_name(trial.name)

    exists_query = sqla.select([self._trials_table])
    exists_query = exists_query.where(
        self._trials_table.c.trial_name == trial.name)
    exists_query = sqla.exists(exists_query).select()
    exists = self._connection.execute(exists_query).fetchone()[0]
    if not exists:
      raise datastore.NotFoundError('Trial %s does not exist.' % trial.name)

    query = sqla.update(self._trials_table).where(
        self._trials_table.c.trial_name == trial.name).values(
            trial_name=trial.name,
            owner_id=trial_resource.owner_id,
            study_id=trial_resource.study_id,
            trial_id=trial_resource.trial_id,
            serialized_trial=trial.SerializeToString())
    self._connection.execute(query)
    return trial_resource

  def list_trials(self, study_name: str) -> List[study_pb2.Trial]:
    study_resource = resources.StudyResource.from_name(study_name)

    exists_query = sqla.select([self._studies_table])
    exists_query = exists_query.where(
        self._studies_table.c.study_name == study_name)
    exists_query = sqla.exists(exists_query).select()
    exists = self._connection.execute(exists_query).fetchone()[0]
    if not exists:
      raise datastore.NotFoundError('Study name %s does not exist.' %
                                    study_name)

    query = sqla.select([self._trials_table])
    query = query.where(
        self._trials_table.c.owner_id == study_resource.owner_id)
    query = query.where(
        self._trials_table.c.study_id == study_resource.study_id)
    result = self._connection.execute(query)
    return [
        study_pb2.Trial.FromString(row['serialized_trial']) for row in result
    ]

  def delete_trial(self, trial_name: str) -> None:
    exists_query = sqla.select([self._trials_table])
    exists_query = exists_query.where(
        self._trials_table.c.trial_name == trial_name)
    exists_query = sqla.exists(exists_query).select()
    exists = self._connection.execute(exists_query).fetchone()[0]
    if not exists:
      raise datastore.NotFoundError('Trial %s does not exist.' % trial_name)

    query = self._trials_table.delete().where(
        self._trials_table.c.trial_name == trial_name)
    self._connection.execute(query)

  def max_trial_id(self, study_name: str) -> int:
    study_resource = resources.StudyResource.from_name(study_name)

    exists_query = sqla.select([self._studies_table])
    exists_query = exists_query.where(
        self._studies_table.c.study_name == study_name)
    exists_query = sqla.exists(exists_query).select()
    exists = self._connection.execute(exists_query).fetchone()[0]
    if not exists:
      raise datastore.NotFoundError('Study %s does not exist.' % study_name)

    query = sqla.select(
        [sqla.func.max(self._trials_table.c.trial_id, type_=sqla.INT)])
    query = query.where(
        self._trials_table.c.owner_id == study_resource.owner_id)
    query = query.where(
        self._trials_table.c.study_id == study_resource.study_id)
    potential_trial_id = self._connection.execute(query).fetchone()[0]
    if potential_trial_id is None:
      return 0
    return potential_trial_id

  def create_suggestion_operation(
      self, operation: operations_pb2.Operation
  ) -> resources.SuggestionOperationResource:
    resource = resources.SuggestionOperationResource.from_name(operation.name)
    query = self._suggestion_operations_table.insert().values(
        operation_name=operation.name,
        owner_id=resource.owner_id,
        client_id=resource.client_id,
        operation_number=resource.operation_number,
        serialized_op=operation.SerializeToString())

    try:
      self._connection.execute(query)
      return resource
    except sqla.exc.IntegrityError as integrity_error:
      raise datastore.AlreadyExistsError(
          'Suggest Op with name %s already exists.' %
          operation.name) from integrity_error

  def get_suggestion_operation(self,
                               operation_name: str) -> operations_pb2.Operation:
    query = sqla.select([self._suggestion_operations_table]).where(
        self._suggestion_operations_table.c.operation_name == operation_name)
    result = self._connection.execute(query)
    row = result.fetchone()

    if not row:
      raise datastore.NotFoundError('Failed to find suggest op name: %s' %
                                    operation_name)

    return operations_pb2.Operation.FromString(row['serialized_op'])

  def update_suggestion_operation(
      self, operation: operations_pb2.Operation
  ) -> resources.SuggestionOperationResource:
    resource = resources.SuggestionOperationResource.from_name(operation.name)

    exists_query = sqla.select([self._suggestion_operations_table])
    exists_query = exists_query.where(
        self._suggestion_operations_table.c.operation_name == operation.name)
    exists_query = sqla.exists(exists_query).select()
    exists = self._connection.execute(exists_query).fetchone()[0]
    if not exists:
      raise datastore.NotFoundError('Suggest op %s does not exist.' %
                                    operation.name)

    query = sqla.update(self._suggestion_operations_table).where(
        self._suggestion_operations_table.c.operation_name ==
        operation.name).values(
            operation_name=operation.name,
            owner_id=resource.owner_id,
            client_id=resource.client_id,
            operation_number=resource.operation_number,
            serialized_op=operation.SerializeToString())
    self._connection.execute(query)
    return resource

  def list_suggestion_operations(
      self,
      owner_name: str,
      client_id: str,
      filter_fn: Optional[Callable[[operations_pb2.Operation], bool]] = None
  ) -> List[operations_pb2.Operation]:
    owner_resource = resources.OwnerResource.from_name(owner_name)
    query = sqla.select([self._suggestion_operations_table])
    query = query.where(
        self._suggestion_operations_table.c.owner_id == owner_resource.owner_id)
    query = query.where(
        self._suggestion_operations_table.c.client_id == client_id)

    exists_query = sqla.exists(query).select()
    exists = self._connection.execute(exists_query).fetchone()[0]
    if not exists:
      raise datastore.NotFoundError('Could not find (owner_name, client_id):',
                                    (owner_name, client_id))

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
    resource = resources.OwnerResource.from_name(owner_name)

    exists_query = sqla.select([self._suggestion_operations_table])
    exists_query = exists_query.where(
        self._suggestion_operations_table.c.owner_id == resource.owner_id)
    exists_query = exists_query.where(
        self._suggestion_operations_table.c.client_id == client_id)
    exists_query = sqla.exists(exists_query).select()
    exists = self._connection.execute(exists_query).fetchone()[0]
    if not exists:
      raise datastore.NotFoundError('Could not find (owner_name, client_id):',
                                    (owner_name, client_id))

    query = sqla.select([
        sqla.func.max(
            self._suggestion_operations_table.c.operation_number,
            type_=sqla.INT)
    ])
    query = query.where(
        self._suggestion_operations_table.c.owner_id == resource.owner_id)
    query = query.where(
        self._suggestion_operations_table.c.client_id == client_id)
    return self._connection.execute(query).fetchone()[0]

  def create_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    resource = resources.EarlyStoppingOperationResource.from_name(
        operation.name)
    query = self._early_stopping_operations_table.insert().values(
        operation_name=operation.name,
        owner_id=resource.owner_id,
        study_id=resource.study_id,
        trial_id=resource.trial_id,
        serialized_op=operation.SerializeToString())
    try:
      self._connection.execute(query)
      return resource
    except sqla.exc.IntegrityError as integrity_error:
      raise datastore.AlreadyExistsError(
          'Early stopping op with name %s already exists.' %
          operation.name) from integrity_error

  def get_early_stopping_operation(
      self, operation_name: str) -> vizier_oss_pb2.EarlyStoppingOperation:
    query = sqla.select([self._early_stopping_operations_table])
    query = query.where(self._early_stopping_operations_table.c.operation_name
                        == operation_name)
    result = self._connection.execute(query)
    row = result.fetchone()

    if not row:
      raise datastore.NotFoundError(
          'Failed to find early stopping op name: %s' % operation_name)

    return vizier_oss_pb2.EarlyStoppingOperation.FromString(
        row['serialized_op'])

  def update_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    resource = resources.EarlyStoppingOperationResource.from_name(
        operation.name)

    exists_query = sqla.select([self._early_stopping_operations_table])
    exists_query = exists_query.where(self._early_stopping_operations_table.c
                                      .operation_name == operation.name)
    exists_query = sqla.exists(exists_query).select()
    exists = self._connection.execute(exists_query).fetchone()[0]
    if not exists:
      raise datastore.NotFoundError('Early stopping op %s does not exist.' %
                                    operation.name)

    query = sqla.update(self._early_stopping_operations_table).where(
        self._early_stopping_operations_table.c.operation_name ==
        operation.name).values(
            operation_name=operation.name,
            owner_id=resource.owner_id,
            study_id=resource.study_id,
            trial_id=resource.trial_id,
            serialized_op=operation.SerializeToString())
    self._connection.execute(query)
    return resource

  def update_metadata(
      self,
      study_name: str,
      study_metadata: Iterable[key_value_pb2.KeyValue],
      trial_metadata: Iterable[datastore._KeyValuePlus],  # pylint:disable=protected-access
  ) -> None:
    """Store the supplied metadata into the SQL database."""
    # TODO: Add exception logic when this function is updated.
    s_resource = resources.StudyResource.from_name(study_name)
    logging.debug('database.update_metadata s_resource= %s', s_resource)

    # Obtain original study.
    get_study_query = sqla.select([
        self._studies_table
    ]).where(self._studies_table.c.study_name == study_name)
    study_result = self._connection.execute(get_study_query)
    row = study_result.fetchone()
    original_study = study_pb2.Study.FromString(row['serialized_study'])

    # Update the study with new study_metadata and update database.
    original_study.study_spec.ClearField('metadata')
    for metadata in study_metadata:
      original_study.study_spec.metadata.append(metadata)
    update_study_query = sqla.update(self._studies_table).where(
        self._studies_table.c.study_name == study_name).values(
            serialized_study=original_study.SerializeToString())
    self._connection.execute(update_study_query)

    # Store trial-related metadata in the database. We first create a dict of
    # the relevant `trial_resources` that will be touched.   We clear them, then
    # loop through the metadata, converting to protos.
    trial_resources: Dict[str, resources.TrialResource] = {}
    for metadata in trial_metadata:
      clear_metadata_bool = False
      if metadata.trial_id in trial_resources:
        t_resource = trial_resources[metadata.trial_id]
      else:
        # If we don't have a t_resource entry already, create one and clear the
        # relevant Trial's metadata.
        t_resource = s_resource.trial_resource(metadata.trial_id)
        trial_resources[metadata.trial_id] = t_resource
        clear_metadata_bool = True

      # Obtain original trial.
      trial_name = t_resource.name
      original_trial_query = sqla.select([
          self._trials_table
      ]).where(self._trials_table.c.trial_name == trial_name)
      trial_result = self._connection.execute(original_trial_query)
      row = trial_result.fetchone()
      original_trial = study_pb2.Trial.FromString(row['serialized_trial'])

      # Edit trial metadata and update database.
      if clear_metadata_bool:
        original_trial.ClearField('metadata')
      original_trial.metadata.append(metadata.k_v)
      update_trial_query = sqla.update(self._trials_table).where(
          self._trials_table.c.trial_name == trial_name).values(
              serialized_trial=original_trial.SerializeToString())
      self._connection.execute(update_trial_query)
