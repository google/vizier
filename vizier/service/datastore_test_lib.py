"""Library functions for testing databases."""
from typing import List

from vizier.service import datastore
from vizier.service import resources
from vizier.service import study_pb2

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
