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

from vizier._src.service import datastore_test_lib
from vizier._src.service import ram_datastore
from vizier._src.service import vizier_service_pb2
from vizier._src.service.testing import util as test_util

from absl.testing import absltest

UnitMetadataUpdate = vizier_service_pb2.UnitMetadataUpdate


class NestedDictRAMDataStoreTest(datastore_test_lib.DataStoreTestCase):

  def setUp(self):
    self.owner_id = 'my_username'
    self.study_id = '123123123'
    self.client_id = 'client_0'
    self.datastore = ram_datastore.NestedDictRAMDataStore()
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


if __name__ == '__main__':
  absltest.main()
