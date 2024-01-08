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

"""Tests for sql_datastore."""
import os
import sqlalchemy as sqla

from vizier._src.service import constants
from vizier._src.service import datastore_test_lib
from vizier._src.service import sql_datastore
from vizier._src.service.testing import util as test_util
from absl.testing import absltest


class SQLDataStoreTest(datastore_test_lib.DataStoreTestCase):

  def setUp(self):
    self.owner_id = 'my_username'
    self.study_id = '123123123'
    self.client_id = 'client_0'
    self.example_study = test_util.generate_study(self.owner_id, self.study_id)
    self.example_trials = test_util.generate_trials(
        [1, 2], owner_id=self.owner_id, study_id=self.study_id
    )
    self.example_suggestion_operations = (
        test_util.generate_suggestion_operations(
            [1, 2, 3, 4], self.owner_id, self.study_id, self.client_id
        )
    )
    self.example_early_stopping_operations = (
        test_util.generate_early_stopping_operations(
            [1, 2], self.owner_id, self.study_id
        )
    )

    engine = sqla.create_engine(constants.SQL_MEMORY_URL, echo=True)
    self.datastore = sql_datastore.SQLDataStore(engine)
    super().setUp()

  def test_study_api(self):
    self.assertStudyAPI(self.datastore, self.example_study)

  def test_trial(self):
    self.assertTrialAPI(self.datastore, self.example_study, self.example_trials)

  def test_suggestion_operation(self):
    self.assertSuggestOpAPI(
        self.datastore,
        self.example_study,
        self.client_id,
        self.example_suggestion_operations,
    )

  def test_early_stopping_operation(self):
    self.assertEarlyStoppingAPI(
        self.datastore,
        self.example_study,
        self.example_trials,
        self.example_early_stopping_operations,
    )

  def test_update_metadata(self):
    self.assertUpdateMetadataAPI(
        self.datastore, self.example_study, self.example_trials
    )


class SQLDataStoreAdditionalTest(absltest.TestCase):
  """For additional tests outside of regular database functionality."""

  def setUp(self):
    super().setUp()
    self.owner_id = 'my_username'
    self.study_id = '123123123'
    self.example_study = test_util.generate_study(self.owner_id, self.study_id)

  @absltest.skip("Github workflow tests don't allow using directories.")
  def test_local_hdd_persistence(self):
    db_path = os.path.join(absltest.get_default_test_tmpdir(), 'local.db')
    sql_url = f'sqlite:///{db_path}'

    engine = sqla.create_engine(sql_url, echo=True)
    datastore = sql_datastore.SQLDataStore(engine)
    datastore.create_study(self.example_study)
    del datastore

    engine2 = sqla.create_engine(sql_url, echo=True)
    datastore2 = sql_datastore.SQLDataStore(engine2)
    study = datastore2.load_study(self.example_study.name)

    self.assertEqual(self.example_study, study)


if __name__ == '__main__':
  absltest.main()
