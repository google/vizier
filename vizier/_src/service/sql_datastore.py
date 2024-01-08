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

"""Implementation of SQL Datastore."""

import collections
import threading
from typing import Callable, Iterable, List, Optional

from absl import logging
import sqlalchemy as sqla
from vizier._src.service import custom_errors
from vizier._src.service import datastore
from vizier._src.service import key_value_pb2
from vizier._src.service import resources
from vizier._src.service import study_pb2
from vizier._src.service import vizier_oss_pb2
from vizier.service import pyvizier as vz

from google.longrunning import operations_pb2

NotFoundError = custom_errors.NotFoundError
AlreadyExistsError = custom_errors.AlreadyExistsError


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
      except sqla.exc.IntegrityError as e:
        raise AlreadyExistsError(
            'Study with name %s already exists.' % study.name
        ) from e

  def load_study(self, study_name: str) -> study_pb2.Study:
    query = sqla.select(self._studies_table)
    query = query.where(self._studies_table.c.study_name == study_name)

    with self._lock:
      result = self._connection.execute(query)

    row = result.fetchone()
    if not row:
      raise NotFoundError('Failed to find study name: %s' % study_name)
    return study_pb2.Study.FromString(row.serialized_study)

  def update_study(self, study: study_pb2.Study) -> resources.StudyResource:
    study_resource = resources.StudyResource.from_name(study.name)

    # Exist query
    eq = sqla.select(self._studies_table)
    eq = eq.where(self._studies_table.c.study_name == study.name)
    eq = sqla.exists(eq).select()

    # Update query
    uq = sqla.update(self._studies_table)
    uq = uq.where(self._studies_table.c.study_name == study.name)
    uq = uq.values(
        study_name=study.name,
        owner_id=study_resource.owner_id,
        study_id=study_resource.study_id,
        serialized_study=study.SerializeToString(),
    )

    with self._lock:
      if not self._connection.execute(eq).fetchone()[0]:
        raise NotFoundError('Study %s does not exist.' % study.name)
      self._connection.execute(uq)
    return study_resource

  def delete_study(self, study_name: str) -> None:
    study_resource = resources.StudyResource.from_name(study_name)

    # Exist query
    eq = sqla.select(self._studies_table)
    eq = eq.where(self._studies_table.c.study_name == study_name)
    eq = sqla.exists(eq).select()

    # Delete study query
    dsq = self._studies_table.delete()
    dsq = dsq.where(self._studies_table.c.study_name == study_name)

    # Delete trials query
    dtq = self._trials_table.delete()
    dtq = dtq.where(self._trials_table.c.owner_id == study_resource.owner_id)
    dtq = dtq.where(self._trials_table.c.study_id == study_resource.study_id)

    with self._lock:
      if not self._connection.execute(eq).fetchone()[0]:
        raise NotFoundError('Study %s does not exist.' % study_name)
      self._connection.execute(dsq)
      self._connection.execute(dtq)

  def list_studies(self, owner_name: str) -> List[study_pb2.Study]:
    owner_id = resources.OwnerResource.from_name(owner_name).owner_id

    # Exist query
    eq = sqla.select(self._owners_table)
    eq = eq.where(self._owners_table.c.owner_name == owner_name)
    eq = sqla.exists(eq).select()

    # List query
    lq = sqla.select(self._studies_table)
    lq = lq.where(self._studies_table.c.owner_id == owner_id)

    with self._lock:
      if not self._connection.execute(eq).fetchone()[0]:
        raise NotFoundError('Owner name %s does not exist.' % owner_name)
      result = self._connection.execute(lq).fetchall()

    return [study_pb2.Study.FromString(row.serialized_study) for row in result]

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
      except sqla.exc.IntegrityError as e:
        raise AlreadyExistsError(
            'Trial with name %s already exists.' % trial.name
        ) from e

  def get_trial(self, trial_name: str) -> study_pb2.Trial:
    query = sqla.select(self._trials_table)
    query = query.where(self._trials_table.c.trial_name == trial_name)

    with self._lock:
      result = self._connection.execute(query)

    row = result.fetchone()
    if not row:
      raise NotFoundError('Failed to find trial name: %s' % trial_name)
    return study_pb2.Trial.FromString(row.serialized_trial)

  def update_trial(self, trial: study_pb2.Trial) -> resources.TrialResource:
    trial_resource = resources.TrialResource.from_name(trial.name)

    # Exist query
    eq = sqla.select(self._trials_table)
    eq = eq.where(self._trials_table.c.trial_name == trial.name)
    eq = sqla.exists(eq).select()

    # Update query
    uq = sqla.update(self._trials_table)
    uq = uq.where(self._trials_table.c.trial_name == trial.name)
    uq = uq.values(
        trial_name=trial.name,
        owner_id=trial_resource.owner_id,
        study_id=trial_resource.study_id,
        trial_id=trial_resource.trial_id,
        serialized_trial=trial.SerializeToString(),
    )

    with self._lock:
      if not self._connection.execute(eq).fetchone()[0]:
        raise NotFoundError('Trial %s does not exist.' % trial.name)
      self._connection.execute(uq)

    return trial_resource

  def list_trials(self, study_name: str) -> List[study_pb2.Trial]:
    study_resource = resources.StudyResource.from_name(study_name)

    # Exist query
    eq = sqla.select(self._studies_table)
    eq = eq.where(self._studies_table.c.study_name == study_name)
    eq = sqla.exists(eq).select()

    # List query
    lq = sqla.select(self._trials_table)
    lq = lq.where(self._trials_table.c.owner_id == study_resource.owner_id)
    lq = lq.where(self._trials_table.c.study_id == study_resource.study_id)

    with self._lock:
      if not self._connection.execute(eq).fetchone()[0]:
        raise NotFoundError('Study name %s does not exist.' % study_name)
      result = self._connection.execute(lq)

    return [study_pb2.Trial.FromString(row.serialized_trial) for row in result]

  def delete_trial(self, trial_name: str) -> None:
    # Exist query
    eq = sqla.select(self._trials_table)
    eq = eq.where(self._trials_table.c.trial_name == trial_name)
    eq = sqla.exists(eq).select()

    # Delete query
    dq = self._trials_table.delete()
    dq = dq.where(self._trials_table.c.trial_name == trial_name)

    with self._lock:
      if not self._connection.execute(eq).fetchone()[0]:
        raise NotFoundError('Trial %s does not exist.' % trial_name)
      self._connection.execute(dq)

  def max_trial_id(self, study_name: str) -> int:
    study_resource = resources.StudyResource.from_name(study_name)

    # Exist query
    eq = sqla.select(self._studies_table)
    eq = eq.where(self._studies_table.c.study_name == study_name)
    eq = sqla.exists(eq).select()

    # Trial ID query
    tq = sqla.func.max(self._trials_table.c.trial_id, type_=sqla.INT)
    tq = sqla.select(tq)
    tq = tq.where(self._trials_table.c.owner_id == study_resource.owner_id)
    tq = tq.where(self._trials_table.c.study_id == study_resource.study_id)

    with self._lock:
      if not self._connection.execute(eq).fetchone()[0]:
        raise NotFoundError('Study %s does not exist.' % study_name)
      potential_trial_id = self._connection.execute(tq).fetchone()[0]

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
    except sqla.exc.IntegrityError as e:
      raise AlreadyExistsError(
          'Suggest Op with name %s already exists.' % operation.name
      ) from e

  def get_suggestion_operation(
      self, operation_name: str
  ) -> operations_pb2.Operation:
    q = sqla.select(self._suggestion_operations_table)
    q = q.where(
        self._suggestion_operations_table.c.operation_name == operation_name
    )

    with self._lock:
      result = self._connection.execute(q)

    row = result.fetchone()
    if not row:
      raise NotFoundError('Failed to find suggest op name: %s' % operation_name)
    return operations_pb2.Operation.FromString(row.serialized_op)

  def update_suggestion_operation(
      self, operation: operations_pb2.Operation
  ) -> resources.SuggestionOperationResource:
    resource = resources.SuggestionOperationResource.from_name(operation.name)

    # Exist query
    eq = sqla.select(self._suggestion_operations_table)
    eq = eq.where(
        self._suggestion_operations_table.c.operation_name == operation.name
    )
    eq = sqla.exists(eq).select()

    # Update query
    uq = sqla.update(self._suggestion_operations_table)
    uq = uq.where(
        self._suggestion_operations_table.c.operation_name == operation.name
    )
    uq = uq.values(
        operation_name=operation.name,
        owner_id=resource.owner_id,
        study_id=resource.study_id,
        client_id=resource.client_id,
        operation_number=resource.operation_number,
        serialized_op=operation.SerializeToString(),
    )

    with self._lock:
      if not self._connection.execute(eq).fetchone()[0]:
        raise NotFoundError('Suggest op %s does not exist.' % operation.name)
      self._connection.execute(uq)
    return resource

  def list_suggestion_operations(
      self,
      study_name: str,
      client_id: str,
      filter_fn: Optional[Callable[[operations_pb2.Operation], bool]] = None,
  ) -> List[operations_pb2.Operation]:
    study_resource = resources.StudyResource.from_name(study_name)
    q = sqla.select(self._suggestion_operations_table)
    q = q.where(
        self._suggestion_operations_table.c.owner_id == study_resource.owner_id
    )
    q = q.where(
        self._suggestion_operations_table.c.study_id == study_resource.study_id
    )
    q = q.where(self._suggestion_operations_table.c.client_id == client_id)

    eq = sqla.exists(q).select()
    with self._lock:
      if not self._connection.execute(eq).fetchone()[0]:
        raise NotFoundError(
            'Could not find (study_name, client_id):',
            (study_resource.name, client_id),
        )
      result = self._connection.execute(q)

    all_ops = [
        operations_pb2.Operation.FromString(row.serialized_op) for row in result
    ]

    if filter_fn is None:
      return all_ops
    return [op for op in all_ops if filter_fn(op)]

  def max_suggestion_operation_number(
      self, study_name: str, client_id: str
  ) -> int:
    resource = resources.StudyResource.from_name(study_name)

    # Exist query
    eq = sqla.select(self._suggestion_operations_table)
    eq = eq.where(
        self._suggestion_operations_table.c.owner_id == resource.owner_id
    )
    eq = eq.where(
        self._suggestion_operations_table.c.study_id == resource.study_id
    )
    eq = eq.where(self._suggestion_operations_table.c.client_id == client_id)
    eq = sqla.exists(eq).select()

    # Max query
    mq = sqla.func.max(
        self._suggestion_operations_table.c.operation_number,
        type_=sqla.INT,
    )
    mq = sqla.select(mq)
    mq = mq.where(
        self._suggestion_operations_table.c.owner_id == resource.owner_id
    )
    mq = mq.where(
        self._suggestion_operations_table.c.study_id == resource.study_id
    )
    mq = mq.where(self._suggestion_operations_table.c.client_id == client_id)

    with self._lock:
      if not self._connection.execute(eq).fetchone()[0]:
        raise NotFoundError(
            'Could not find (study_name, client_id):', (study_name, client_id)
        )
      return self._connection.execute(mq).fetchone()[0]

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
    except sqla.exc.IntegrityError as e:
      raise AlreadyExistsError(
          'Early stopping op with name %s already exists.' % operation.name
      ) from e

  def get_early_stopping_operation(
      self, operation_name: str
  ) -> vizier_oss_pb2.EarlyStoppingOperation:
    q = sqla.select(self._early_stopping_operations_table)
    q = q.where(
        self._early_stopping_operations_table.c.operation_name == operation_name
    )

    with self._lock:
      result = self._connection.execute(q)

    row = result.fetchone()
    if not row:
      raise NotFoundError(
          'Failed to find early stopping op name: %s' % operation_name
      )
    return vizier_oss_pb2.EarlyStoppingOperation.FromString(row.serialized_op)

  def update_early_stopping_operation(
      self, operation: vizier_oss_pb2.EarlyStoppingOperation
  ) -> resources.EarlyStoppingOperationResource:
    resource = resources.EarlyStoppingOperationResource.from_name(
        operation.name
    )

    # Exist query
    eq = sqla.select(self._early_stopping_operations_table)
    eq = eq.where(
        self._early_stopping_operations_table.c.operation_name == operation.name
    )
    eq = sqla.exists(eq).select()

    # Update query
    uq = sqla.update(self._early_stopping_operations_table)
    uq = uq.where(
        self._early_stopping_operations_table.c.operation_name == operation.name
    )
    uq = uq.values(
        operation_name=operation.name,
        owner_id=resource.owner_id,
        study_id=resource.study_id,
        trial_id=resource.trial_id,
        serialized_op=operation.SerializeToString(),
    )

    with self._lock:
      if not self._connection.execute(eq).fetchone()[0]:
        raise NotFoundError(
            'Early stopping op %s does not exist.' % operation.name
        )
      self._connection.execute(uq)
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
    sq = sqla.select(self._studies_table)
    sq = sq.where(self._studies_table.c.study_name == study_name)

    with self._lock:
      study_result = self._connection.execute(sq)
      row = study_result.fetchone()
      if not row:
        raise NotFoundError('No such study:', s_resource.name)
      original_study = study_pb2.Study.FromString(row.serialized_study)

      # Store Study-related metadata into the database.
      vz.metadata_util.merge_study_metadata(
          original_study.study_spec, study_metadata
      )

      usq = sqla.update(self._studies_table)
      usq = usq.where(self._studies_table.c.study_name == study_name)
      usq = usq.values(serialized_study=original_study.SerializeToString())
      self._connection.execute(usq)

      # Split the trial-related metadata by Trial.
      split_metadata = collections.defaultdict(list)
      for md in trial_metadata:
        split_metadata[md.trial_id].append(md)

      # Now, we update one Trial at a time:
      for trial_id, md_list in split_metadata.items():
        t_resource = s_resource.trial_resource(trial_id)
        trial_name = t_resource.name

        # Obtain original trial.
        otq = sqla.select(self._trials_table)
        otq = otq.where(self._trials_table.c.trial_name == trial_name)
        trial_result = self._connection.execute(otq)
        row = trial_result.fetchone()
        if not row:
          raise NotFoundError('No such trial:', trial_name)
        original_trial = study_pb2.Trial.FromString(row.serialized_trial)

        # Update Trial.
        vz.metadata_util.merge_trial_metadata(original_trial, md_list)
        utq = sqla.update(self._trials_table)
        utq = utq.where(self._trials_table.c.trial_name == trial_name)
        utq = utq.values(serialized_trial=original_trial.SerializeToString())
        self._connection.execute(utq)
