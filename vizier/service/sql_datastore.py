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

"""Implementation of SQL Datastore."""
import collections
import threading
from typing import Callable, DefaultDict, Iterable, List, Optional
from absl import logging

import sqlalchemy as sqla

from vizier.service import custom_errors
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
        sqla.Column('study_id', sqla.String),
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
    # This lock is meant to lock `execute()` calls for database types which
    # don't support multi-threading, like SQLite.
    self._lock = threading.Lock()
    self._root_metadata.create_all(self._engine)

  def create_study(self, study: study_pb2.Study) -> resources.StudyResource:
    study_resource = resources.StudyResource.from_name(study.name)
    owner_name = study_resource.owner_resource.name
    owner_query = self._owners_table.insert().values(owner_name=owner_name)
    study_query = self._studies_table.insert().values(
        study_name=study.name,
        owner_id=study_resource.owner_id,
        study_id=study_resource.study_id,
        serialized_study=study.SerializeToString(),
    )

    with self._lock:
      try:
        self._connection.execute(owner_query)
      except sqla.exc.IntegrityError:
        logging.info('Owner with name %s currently exists.', owner_name)
      try:
        self._connection.execute(study_query)
        return study_resource
      except sqla.exc.IntegrityError as integrity_error:
        raise custom_errors.AlreadyExistsError(
            'Study with name %s already exists.' % study.name
        ) from integrity_error

  def load_study(self, study_name: str) -> study_pb2.Study:
    query = sqla.select([self._studies_table])
    query = query.where(self._studies_table.c.study_name == study_name)

    with self._lock:
      result = self._connection.execute(query)

    row = result.fetchone()
    if not row:
      raise custom_errors.NotFoundError(
          'Failed to find study name: %s' % study_name
      )
    return study_pb2.Study.FromString(row['serialized_study'])

  def update_study(self, study: study_pb2.Study) -> resources.StudyResource:
    study_resource = resources.StudyResource.from_name(study.name)
    exists_query = sqla.exists(
        sqla.select([self._studies_table]).where(
            self._studies_table.c.study_name == study.name
        )
    ).select()
    update_query = (
        sqla.update(self._studies_table)
        .where(self._studies_table.c.study_name == study.name)
        .values(
            study_name=study.name,
            owner_id=study_resource.owner_id,
            study_id=study_resource.study_id,
            serialized_study=study.SerializeToString(),
        )
    )

    with self._lock:
      exists = self._connection.execute(exists_query).fetchone()[0]
      if not exists:
        raise custom_errors.NotFoundError(
            'Study %s does not exist.' % study.name
        )
      self._connection.execute(update_query)
    return study_resource

  def delete_study(self, study_name: str) -> None:
    study_resource = resources.StudyResource.from_name(study_name)

    exists_query = sqla.select([self._studies_table])
    exists_query = exists_query.where(
        self._studies_table.c.study_name == study_name
    )
    exists_query = sqla.exists(exists_query).select()
    delete_study_query = self._studies_table.delete().where(
        self._studies_table.c.study_name == study_name
    )
    delete_trials_query = (
        self._trials_table.delete()
        .where(self._trials_table.c.owner_id == study_resource.owner_id)
        .where(self._trials_table.c.study_id == study_resource.study_id)
    )

    with self._lock:
      exists = self._connection.execute(exists_query).fetchone()[0]
      if not exists:
        raise custom_errors.NotFoundError(
            'Study %s does not exist.' % study_name
        )
      self._connection.execute(delete_study_query)
      self._connection.execute(delete_trials_query)

  def list_studies(self, owner_name: str) -> List[study_pb2.Study]:
    owner_id = resources.OwnerResource.from_name(owner_name).owner_id
    exists_query = sqla.exists(
        sqla.select([self._owners_table]).where(
            self._owners_table.c.owner_name == owner_name
        )
    ).select()
    list_query = sqla.select([self._studies_table]).where(
        self._studies_table.c.owner_id == owner_id
    )

    with self._lock:
      exists = self._connection.execute(exists_query).fetchone()[0]
      if not exists:
        raise custom_errors.NotFoundError(
            'Owner name %s does not exist.' % owner_name
        )
      result = self._connection.execute(list_query).fetchall()

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
        serialized_trial=trial.SerializeToString(),
    )

    with self._lock:
      try:
        self._connection.execute(query)
        return trial_resource
      except sqla.exc.IntegrityError as integrity_error:
        raise custom_errors.AlreadyExistsError(
            'Trial with name %s already exists.' % trial.name
        ) from integrity_error

  def get_trial(self, trial_name: str) -> study_pb2.Trial:
    query = sqla.select([self._trials_table])
    query = query.where(self._trials_table.c.trial_name == trial_name)

    with self._lock:
      result = self._connection.execute(query)

    row = result.fetchone()
    if not row:
      raise custom_errors.NotFoundError(
          'Failed to find trial name: %s' % trial_name
      )
    return study_pb2.Trial.FromString(row['serialized_trial'])

  def update_trial(self, trial: study_pb2.Trial) -> resources.TrialResource:
    trial_resource = resources.TrialResource.from_name(trial.name)
    exists_query = sqla.exists(
        sqla.select([self._trials_table]).where(
            self._trials_table.c.trial_name == trial.name
        )
    ).select()
    update_query = (
        sqla.update(self._trials_table)
        .where(self._trials_table.c.trial_name == trial.name)
        .values(
            trial_name=trial.name,
            owner_id=trial_resource.owner_id,
            study_id=trial_resource.study_id,
            trial_id=trial_resource.trial_id,
            serialized_trial=trial.SerializeToString(),
        )
    )

    with self._lock:
      exists = self._connection.execute(exists_query).fetchone()[0]
      if not exists:
        raise custom_errors.NotFoundError(
            'Trial %s does not exist.' % trial.name
        )
      self._connection.execute(update_query)

    return trial_resource

  def list_trials(self, study_name: str) -> List[study_pb2.Trial]:
    study_resource = resources.StudyResource.from_name(study_name)
    exists_query = sqla.exists(
        sqla.select([self._studies_table]).where(
            self._studies_table.c.study_name == study_name
        )
    ).select()
    list_query = (
        sqla.select([self._trials_table])
        .where(self._trials_table.c.owner_id == study_resource.owner_id)
        .where(self._trials_table.c.study_id == study_resource.study_id)
    )

    with self._lock:
      exists = self._connection.execute(exists_query).fetchone()[0]
      if not exists:
        raise custom_errors.NotFoundError(
            'Study name %s does not exist.' % study_name
        )
      result = self._connection.execute(list_query)

    return [
        study_pb2.Trial.FromString(row['serialized_trial']) for row in result
    ]

  def delete_trial(self, trial_name: str) -> None:
    exists_query = sqla.exists(
        sqla.select([self._trials_table]).where(
            self._trials_table.c.trial_name == trial_name
        )
    ).select()
    delete_query = self._trials_table.delete().where(
        self._trials_table.c.trial_name == trial_name
    )
    with self._lock:
      exists = self._connection.execute(exists_query).fetchone()[0]
      if not exists:
        raise custom_errors.NotFoundError(
            'Trial %s does not exist.' % trial_name
        )
      self._connection.execute(delete_query)

  def max_trial_id(self, study_name: str) -> int:
    study_resource = resources.StudyResource.from_name(study_name)
    exists_query = sqla.exists(
        sqla.select([self._studies_table]).where(
            self._studies_table.c.study_name == study_name
        )
    ).select()
    trial_id_query = (
        sqla.select(
            [sqla.func.max(self._trials_table.c.trial_id, type_=sqla.INT)]
        )
        .where(self._trials_table.c.owner_id == study_resource.owner_id)
        .where(self._trials_table.c.study_id == study_resource.study_id)
    )

    with self._lock:
      exists = self._connection.execute(exists_query).fetchone()[0]
      if not exists:
        raise custom_errors.NotFoundError(
            'Study %s does not exist.' % study_name
        )
      potential_trial_id = self._connection.execute(trial_id_query).fetchone()[
          0
      ]

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
        study_id=resource.study_id,
        client_id=resource.client_id,
        operation_number=resource.operation_number,
        serialized_op=operation.SerializeToString(),
    )

    try:
      with self._lock:
        self._connection.execute(query)
      return resource
    except sqla.exc.IntegrityError as integrity_error:
      raise custom_errors.AlreadyExistsError(
          'Suggest Op with name %s already exists.' % operation.name
      ) from integrity_error

  def get_suggestion_operation(
      self, operation_name: str
  ) -> operations_pb2.Operation:
    query = sqla.select([self._suggestion_operations_table]).where(
        self._suggestion_operations_table.c.operation_name == operation_name
    )

    with self._lock:
      result = self._connection.execute(query)

    row = result.fetchone()
    if not row:
      raise custom_errors.NotFoundError(
          'Failed to find suggest op name: %s' % operation_name
      )
    return operations_pb2.Operation.FromString(row['serialized_op'])

  def update_suggestion_operation(
      self, operation: operations_pb2.Operation
  ) -> resources.SuggestionOperationResource:
    resource = resources.SuggestionOperationResource.from_name(operation.name)

    exists_query = sqla.exists(
        sqla.select([self._suggestion_operations_table]).where(
            self._suggestion_operations_table.c.operation_name == operation.name
        )
    ).select()
    update_query = (
        sqla.update(self._suggestion_operations_table)
        .where(
            self._suggestion_operations_table.c.operation_name == operation.name
        )
        .values(
            operation_name=operation.name,
            owner_id=resource.owner_id,
            study_id=resource.study_id,
            client_id=resource.client_id,
            operation_number=resource.operation_number,
            serialized_op=operation.SerializeToString(),
        )
    )

    with self._lock:
      exists = self._connection.execute(exists_query).fetchone()[0]
      if not exists:
        raise custom_errors.NotFoundError(
            'Suggest op %s does not exist.' % operation.name
        )
      self._connection.execute(update_query)
    return resource

  def list_suggestion_operations(
      self,
      study_name: str,
      client_id: str,
      filter_fn: Optional[Callable[[operations_pb2.Operation], bool]] = None,
  ) -> List[operations_pb2.Operation]:
    study_resource = resources.StudyResource.from_name(study_name)
    query = sqla.select([self._suggestion_operations_table])
    query = query.where(
        self._suggestion_operations_table.c.owner_id == study_resource.owner_id
    )
    query = query.where(
        self._suggestion_operations_table.c.study_id == study_resource.study_id
    )
    query = query.where(
        self._suggestion_operations_table.c.client_id == client_id
    )

    exists_query = sqla.exists(query).select()
    with self._lock:
      exists = self._connection.execute(exists_query).fetchone()[0]
      if not exists:
        raise custom_errors.NotFoundError(
            'Could not find (study_name, client_id):',
            (study_resource.name, client_id),
        )
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

  def max_suggestion_operation_number(
      self, study_name: str, client_id: str
  ) -> int:
    resource = resources.StudyResource.from_name(study_name)

    exists_query = sqla.exists(
        sqla.select([self._suggestion_operations_table])
        .where(
            self._suggestion_operations_table.c.owner_id == resource.owner_id
        )
        .where(
            self._suggestion_operations_table.c.study_id == resource.study_id
        )
        .where(self._suggestion_operations_table.c.client_id == client_id)
    ).select()
    max_query = (
        sqla.select(
            [
                sqla.func.max(
                    self._suggestion_operations_table.c.operation_number,
                    type_=sqla.INT,
                )
            ]
        )
        .where(
            self._suggestion_operations_table.c.owner_id == resource.owner_id
        )
        .where(
            self._suggestion_operations_table.c.study_id == resource.study_id
        )
        .where(self._suggestion_operations_table.c.client_id == client_id)
    )

    with self._lock:
      exists = self._connection.execute(exists_query).fetchone()[0]
      if not exists:
        raise custom_errors.NotFoundError(
            'Could not find (study_name, client_id):', (study_name, client_id)
        )
      return self._connection.execute(max_query).fetchone()[0]

  def create_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    resource = resources.EarlyStoppingOperationResource.from_name(
        operation.name
    )
    query = self._early_stopping_operations_table.insert().values(
        operation_name=operation.name,
        owner_id=resource.owner_id,
        study_id=resource.study_id,
        trial_id=resource.trial_id,
        serialized_op=operation.SerializeToString(),
    )

    try:
      with self._lock:
        self._connection.execute(query)
      return resource
    except sqla.exc.IntegrityError as integrity_error:
      raise custom_errors.AlreadyExistsError(
          'Early stopping op with name %s already exists.' % operation.name
      ) from integrity_error

  def get_early_stopping_operation(
      self, operation_name: str
  ) -> vizier_oss_pb2.EarlyStoppingOperation:
    query = sqla.select([self._early_stopping_operations_table]).where(
        self._early_stopping_operations_table.c.operation_name == operation_name
    )

    with self._lock:
      result = self._connection.execute(query)

    row = result.fetchone()
    if not row:
      raise custom_errors.NotFoundError(
          'Failed to find early stopping op name: %s' % operation_name
      )
    return vizier_oss_pb2.EarlyStoppingOperation.FromString(
        row['serialized_op']
    )

  def update_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    resource = resources.EarlyStoppingOperationResource.from_name(
        operation.name
    )
    exists_query = sqla.exists(
        sqla.select([self._early_stopping_operations_table]).where(
            self._early_stopping_operations_table.c.operation_name
            == operation.name
        )
    ).select()
    update_query = (
        sqla.update(self._early_stopping_operations_table)
        .where(
            self._early_stopping_operations_table.c.operation_name
            == operation.name
        )
        .values(
            operation_name=operation.name,
            owner_id=resource.owner_id,
            study_id=resource.study_id,
            trial_id=resource.trial_id,
            serialized_op=operation.SerializeToString(),
        )
    )

    with self._lock:
      exists = self._connection.execute(exists_query).fetchone()[0]
      if not exists:
        raise custom_errors.NotFoundError(
            'Early stopping op %s does not exist.' % operation.name
        )
      self._connection.execute(update_query)
      return resource

  def update_metadata(
      self,
      study_name: str,
      study_metadata: Iterable[key_value_pb2.KeyValue],
      trial_metadata: Iterable[datastore.UnitMetadataUpdate],
  ) -> None:
    """Store the supplied metadata into the SQL database."""
    s_resource = resources.StudyResource.from_name(study_name)
    logging.debug('database.update_metadata s_resource= %s', s_resource)
    # Obtain original study.
    get_study_query = sqla.select([self._studies_table]).where(
        self._studies_table.c.study_name == study_name
    )

    with self._lock:
      study_result = self._connection.execute(get_study_query)
      row = study_result.fetchone()
      if not row:
        raise custom_errors.NotFoundError('No such study:', s_resource.name)
      original_study = study_pb2.Study.FromString(row['serialized_study'])

      # Store Study-related metadata into the database.
      datastore.merge_study_metadata(original_study.study_spec, study_metadata)
      update_study_query = (
          sqla.update(self._studies_table)
          .where(self._studies_table.c.study_name == study_name)
          .values(serialized_study=original_study.SerializeToString())
      )
      self._connection.execute(update_study_query)

      # Split the trial-related metadata by Trial.
      split_metadata: DefaultDict[str, List[datastore.UnitMetadataUpdate]] = (
          collections.defaultdict(list)
      )
      for md in trial_metadata:
        split_metadata[md.trial_id].append(md)

      # Now, we update one Trial at a time:
      for trial_id, md_list in split_metadata.items():
        t_resource = s_resource.trial_resource(trial_id)

        # Obtain original trial.
        trial_name = t_resource.name
        original_trial_query = sqla.select([self._trials_table]).where(
            self._trials_table.c.trial_name == trial_name
        )
        trial_result = self._connection.execute(original_trial_query)
        row = trial_result.fetchone()
        if not row:
          raise custom_errors.NotFoundError('No such trial:', trial_name)
        original_trial = study_pb2.Trial.FromString(row['serialized_trial'])

        # Update Trial.
        datastore.merge_trial_metadata(original_trial, md_list)
        update_trial_query = (
            sqla.update(self._trials_table)
            .where(self._trials_table.c.trial_name == trial_name)
            .values(serialized_trial=original_trial.SerializeToString())
        )
        self._connection.execute(update_trial_query)
