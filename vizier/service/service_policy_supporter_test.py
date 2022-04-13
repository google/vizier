"""Tests for vizier.service.service_policy_supporter."""

import logging
from vizier import pythia
from vizier.service import pyvizier
from vizier.service import resources
from vizier.service import service_policy_supporter
from vizier.service import study_pb2
from vizier.service import test_util
from vizier.service import vizier_server

from absl.testing import absltest


class PythiaSupporterTest(absltest.TestCase):

  def setUp(self):
    self.owner_id = 'my_username'
    self.study_id = '1231223'
    self.study_name = resources.StudyResource(
        owner_id=self.owner_id, study_id=self.study_id).name
    self.vs = vizier_server.VizierService()
    self.example_study = test_util.generate_study(self.owner_id, self.study_id)
    self.vs.datastore.create_study(self.example_study)

    self.active_trials = test_util.generate_trials(
        range(1, 7), self.owner_id, self.study_id)
    self.succeeded_trial = test_util.generate_trials(
        [7],
        self.owner_id,
        self.study_id,
        state=study_pb2.Trial.State.SUCCEEDED,
        final_measurement=study_pb2.Measurement())[0]

    for trial in self.active_trials + [self.succeeded_trial]:
      self.vs.datastore.create_trial(trial)

    self.policy_supporter = service_policy_supporter.ServicePolicySupporter(
        self.study_name, self.vs)

    super().setUp()

  def test_trial_names_filter(self):
    trials = self.policy_supporter.GetTrials(
        study_guid=self.study_name, trial_ids=[3, 4])

    self.assertEqual(trials[0],
                     pyvizier.TrialConverter.from_proto(self.active_trials[2]))
    self.assertEqual(trials[1],
                     pyvizier.TrialConverter.from_proto(self.active_trials[3]))

  def test_min_max_filter(self):
    trials = self.policy_supporter.GetTrials(
        study_guid=self.study_name, min_trial_id=3, max_trial_id=4)

    self.assertEqual(trials[0],
                     pyvizier.TrialConverter.from_proto(self.active_trials[2]))
    self.assertEqual(trials[1],
                     pyvizier.TrialConverter.from_proto(self.active_trials[3]))

  def test_status_match_filter(self):
    trials = self.policy_supporter.GetTrials(
        study_guid=self.study_name, status_matches=pyvizier.TrialStatus.ACTIVE)

    self.assertLen(trials, 6)
    self.assertEqual(trials[0],
                     pyvizier.TrialConverter.from_proto(self.active_trials[0]))

  def test_raise_value_error(self):

    def should_raise_value_error_fn():
      self.policy_supporter.GetTrials(
          study_guid=self.study_name,
          trial_ids=[1],
          status_matches=pyvizier.TrialStatus.ACTIVE)

      with self.assertRaises(ValueError):
        should_raise_value_error_fn()

  def test_get_study_config(self):
    pythia_sc = self.policy_supporter.GetStudyConfig(self.study_name)
    correct_pythia_sc = pyvizier.StudyConfig.from_proto(
        self.example_study.study_spec)
    self.assertEqual(pythia_sc, correct_pythia_sc)

  def test_update_metadata(self):
    on_study_metadata = pyvizier.Metadata()
    on_study_metadata.ns('bar')['foo'] = '.bar.foo.1'
    on_trial1_metadata = pyvizier.Metadata()
    on_trial1_metadata.ns('bax')['nerf'] = '1.bar.nerf.2'
    delta = pythia.MetadataDelta(
        on_study=on_study_metadata, on_trials={1: on_trial1_metadata})
    self.policy_supporter.SendMetadata(delta)
    # Read to see that the results are correct.
    pythia_sc = self.policy_supporter.GetStudyConfig(self.study_name)
    self.assertLen(pythia_sc.metadata.namespaces(), 1)
    logging.info('pythia_sc namespaces: %s', pythia_sc.metadata.namespaces())
    logging.info('pythia_sc = %s', pythia_sc.metadata.ns('bar'))
    logging.info('on_study = %s', on_study_metadata.ns('bar'))
    self.assertNotEmpty(pythia_sc.metadata.ns('bar'))
    self.assertEqual(pythia_sc.metadata.ns('bar'), on_study_metadata.ns('bar'))
    trials = self.policy_supporter.GetTrials(
        study_guid=self.study_name, min_trial_id=1, max_trial_id=1)
    self.assertLen(trials, 1)
    self.assertNotEmpty(trials[0].metadata.ns('bax'))
    self.assertEqual(trials[0].metadata.ns('bax'), on_trial1_metadata.ns('bax'))


if __name__ == '__main__':
  absltest.main()
