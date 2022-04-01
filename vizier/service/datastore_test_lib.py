"""Library functions for testing databases."""
from typing import List

from vizier.service import datastore
from vizier.service import resources
from vizier.service import study_pb2
from vizier.service import vizier_oss_pb2

from google.longrunning import operations_pb2
from absl.testing import parameterized


class DataStoreTestCase(parameterized.TestCase):
  """Base class for testing datastores."""

  def assertStudyAPI(self, ds: datastore.DataStore, study: study_pb2.Study):
    """Tests if the datastore handles studies correctly."""
    ds.create_study(study)
    output_study = ds.load_study(study.name)
    self.assertEqual(output_study, study)

    owner_name = resources.StudyResource.from_name(
        study.name).owner_resource.name
    list_of_one_study = ds.list_studies(owner_name)
    self.assertLen(list_of_one_study, 1)
    self.assertEqual(list_of_one_study[0], study)

    ds.delete_study(study.name)
    empty_list = ds.list_studies(owner_name)
    self.assertEmpty(empty_list)

  def assertTrialAPI(self, ds: datastore.DataStore, study: study_pb2.Study,
                     trials: List[study_pb2.Trial]):
    """Tests if the datastore handles trials correctly."""
    ds.create_study(study)
    for trial in trials:
      ds.create_trial(trial)

    self.assertLen(trials, ds.max_trial_id(study.name))

    list_of_trials = ds.list_trials(study.name)
    self.assertEqual(list_of_trials, trials)

    first_trial = trials[0]
    output_trial = ds.get_trial(first_trial.name)
    self.assertEqual(output_trial, first_trial)

    ds.delete_trial(first_trial.name)
    leftover_trials = ds.list_trials(study.name)
    self.assertEqual(leftover_trials, trials[1:])

  def assertSuggestOpAPI(self, ds: datastore.DataStore, study: study_pb2.Study,
                         client_id: str,
                         suggestion_ops: List[operations_pb2.Operation]):
    """Tests if the datastore handles suggest ops correctly."""
    study_resource = resources.StudyResource.from_name(study.name)

    ds.create_study(study)
    for operation in suggestion_ops:
      ds.create_suggestion_operation(operation)

    self.assertLen(
        suggestion_ops,
        ds.max_suggestion_operation_number(
            resources.OwnerResource(study_resource.owner_id).name, client_id))

    list_of_operations = ds.list_suggestion_operations(
        resources.OwnerResource(study_resource.owner_id).name, client_id)
    self.assertEqual(list_of_operations, suggestion_ops)

    output_operation = ds.get_suggestion_operation(
        resources.SuggestionOperationResource(
            study_resource.owner_id, client_id, operation_number=1).name)
    self.assertEqual(output_operation, suggestion_ops[0])

  def assertEarlyStoppingAPI(
      self, ds: datastore.DataStore, study: study_pb2.Study,
      trials: List[study_pb2.Trial],
      early_stopping_ops: List[vizier_oss_pb2.EarlyStoppingOperation]):
    """Tests if the datastore handles early stopping ops correctly."""
    study_resource = resources.StudyResource.from_name(study.name)
    ds.create_study(study)

    for trial in trials:
      ds.create_trial(trial)

    for operation in early_stopping_ops:
      ds.create_early_stopping_operation(operation)

    output_operation = ds.get_early_stopping_operation(
        resources.EarlyStoppingOperationResource(study_resource.owner_id,
                                                 study_resource.study_id,
                                                 1).name)
    self.assertEqual(output_operation, early_stopping_ops[0])
