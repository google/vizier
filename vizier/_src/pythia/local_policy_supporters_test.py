"""Tests for vizier.pythia.bsae.local_policy_supporters."""

from vizier._src.pythia import local_policy_supporters
from vizier._src.pythia import policy_supporter
from vizier.pyvizier import pythia as vz

from absl.testing import absltest

LocalPolicyRunner = local_policy_supporters.LocalPolicyRunner


def _runner_with_10trials():
  runner = LocalPolicyRunner(vz.StudyConfig())
  runner.AddTrials([vz.Trial() for _ in range(1, 11)])
  return runner


class LocalPolicySupportersTest(absltest.TestCase):

  def test_add_and_get_trials(self):
    runner = _runner_with_10trials()
    trials = runner.GetTrials()
    self.assertLen(trials, 10)
    # The 10 trials are assigned ids 1 through 10 automatically.
    self.assertSequenceEqual([t.id for t in trials], range(1, 11))

  def test_update_metadata(self):
    runner = _runner_with_10trials()
    trial1 = runner.GetTrials(min_trial_id=1, max_trial_id=1)[0]

    with policy_supporter.MetadataUpdate(runner) as mu:
      mu.assign('ns', 'key', 'value')
      mu.assign('ns', 'key', 'value', trial_id=1)
      # Metadata update is not immediate.
      self.assertEmpty(runner.GetStudyConfig().metadata.ns('ns'))
      self.assertEmpty(trial1.metadata.ns('ns'))

    self.assertEqual(runner.GetStudyConfig().metadata.ns('ns').get('key'),
                     'value')
    trial0 = runner.GetTrials(min_trial_id=1, max_trial_id=1)[0]
    self.assertSequenceEqual(trial0.metadata.ns('ns'), {'key': 'value'})

  def test_update_metadata_inplace(self):
    runner = _runner_with_10trials()
    trial1 = runner.GetTrials(min_trial_id=1, max_trial_id=1)[0]
    with policy_supporter.MetadataUpdate(runner) as mu:
      mu.assign('ns', 'key', 'value', trial=trial1)
      self.assertEqual(trial1.metadata.ns('ns').get('key'), 'value')

    # Update is reflected.
    trial1 = runner.GetTrials(min_trial_id=1, max_trial_id=1)[0]
    self.assertSequenceEqual(trial1.metadata.ns('ns'), {'key': 'value'})


if __name__ == '__main__':
  absltest.main()
