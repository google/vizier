"""Tests for vizier.service.datastore."""
from vizier.service import datastore
from vizier.service import datastore_test_lib
from vizier.service import key_value_pb2
from vizier.service import resources
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
    self.datastore.create_study(self.example_study)
    for operation in self.example_suggestion_operations:
      self.datastore.create_suggestion_operation(operation)

    self.assertLen(
        self.example_suggestion_operations,
        self.datastore.max_suggestion_operation_number(
            resources.OwnerResource(self.owner_id).name, self.client_id))

    list_of_operations = self.datastore.list_suggestion_operations(
        resources.OwnerResource(self.owner_id).name, self.client_id)
    self.assertEqual(list_of_operations, self.example_suggestion_operations)

    output_operation = self.datastore.get_suggestion_operation(
        resources.SuggestionOperationResource(
            self.owner_id, self.client_id, operation_number=1).name)
    self.assertEqual(output_operation, self.example_suggestion_operations[0])

  def test_early_stopping_operation(self):
    self.datastore.create_study(self.example_study)

    for trial in self.example_trials:
      self.datastore.create_trial(trial)

    for operation in self.example_early_stopping_operations:
      self.datastore.create_early_stopping_operation(operation)

    output_operation = self.datastore.get_early_stopping_operation(
        resources.EarlyStoppingOperationResource(self.owner_id, self.study_id,
                                                 1).name)
    self.assertEqual(output_operation,
                     self.example_early_stopping_operations[0])

  def test_update_metadata(self):
    self.datastore.create_study(self.example_study)
    for trial in self.example_trials:
      self.datastore.create_trial(trial)
    study_metadata = [key_value_pb2.KeyValue(key='a', ns='b', value='C')]
    trial_metadata = [
        _KeyValuePlus(
            trial_id='1',
            k_v=key_value_pb2.KeyValue(key='d', ns='e', value='F'))
    ]
    s_resource = resources.StudyResource(self.owner_id, self.study_id)
    self.datastore.update_metadata(s_resource.name, study_metadata,
                                   trial_metadata)
    mutated_study_config = self.datastore.load_study(s_resource.name).study_spec
    self.assertEqual(list(mutated_study_config.metadata), study_metadata)
    mutated_trial = self.datastore.get_trial(self.example_trials[0].name)
    self.assertEqual(mutated_trial.id, str(trial_metadata[0].trial_id))
    self.assertEqual(list(mutated_trial.metadata), [trial_metadata[0].k_v])


if __name__ == '__main__':
  absltest.main()
