"""Tests for vizier.service.datastore."""
from vizier.service import datastore
from vizier.service import datastore_test_lib
from vizier.service import test_util
from vizier.service import vizier_service_pb2

from absl.testing import absltest

_KeyValuePlus = vizier_service_pb2.UpdateMetadataRequest.KeyValuePlus


class NestedDictRAMDataStoreTest(datastore_test_lib.DataStoreTestCase):

  def setUp(self):
    self.owner_id = 'my_username'
    self.study_id = '123123123'
    self.client_id = 'client_0'
    self.datastore = datastore.NestedDictRAMDataStore()
    self.example_study = test_util.generate_study(self.owner_id, self.study_id)
    self.example_trials = test_util.generate_trials([1, 2],
                                                    owner_id=self.owner_id,
                                                    study_id=self.study_id)
    self.example_suggestion_operations = test_util.generate_suggestion_operations(
        [1, 2, 3, 4], self.owner_id, self.client_id)
    self.example_early_stopping_operations = test_util.generate_early_stopping_operations(
        [1, 2], self.owner_id, self.study_id)
    super().setUp()

  def test_study_api(self):
    self.assertStudyAPI(self.datastore, self.example_study)

  def test_trial(self):
    self.assertTrialAPI(self.datastore, self.example_study, self.example_trials)

  def test_suggestion_operation(self):
    self.assertSuggestOpAPI(self.datastore, self.example_study, self.client_id,
                            self.example_suggestion_operations)

  def test_early_stopping_operation(self):
    self.assertEarlyStoppingAPI(self.datastore, self.example_study,
                                self.example_trials,
                                self.example_early_stopping_operations)

  def test_update_metadata(self):
    self.assertUpdateMetadataAPI(self.datastore, self.example_study,
                                 self.example_trials)


if __name__ == '__main__':
  absltest.main()
