"""Tests for vizier.pythia.bsae.local_policy_supporters."""

import numpy as np

from vizier import pyvizier as vz
from vizier._src.pythia import local_policy_supporters
from absl.testing import absltest
from absl.testing import parameterized

LocalPolicyRunner = local_policy_supporters.LocalPolicyRunner


def _runner_with_10trials():
  runner = LocalPolicyRunner(vz.StudyConfig())
  runner.AddTrials([vz.Trial() for _ in range(1, 11)])
  return runner


class LocalPolicySupportersTest(parameterized.TestCase):

  def test_add_and_get_trials(self):
    runner = _runner_with_10trials()
    trials = runner.GetTrials()
    self.assertLen(trials, 10)
    # The 10 trials are assigned ids 1 through 10 automatically.
    self.assertSequenceEqual([t.id for t in trials], range(1, 11))

  def test_update_metadata(self):
    runner = _runner_with_10trials()
    trial1 = runner.GetTrials(min_trial_id=1, max_trial_id=1)[0]

    with runner.MetadataUpdate() as mu:
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
    with runner.MetadataUpdate() as mu:
      mu.assign('ns', 'key', 'value', trial=trial1)
      self.assertEqual(trial1.metadata.ns('ns').get('key'), 'value')

    # Update is reflected.
    trial1 = runner.GetTrials(min_trial_id=1, max_trial_id=1)[0]
    self.assertSequenceEqual(trial1.metadata.ns('ns'), {'key': 'value'})


class LocalPolicySupportersGetBestTrialsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(goal=vz.ObjectiveMetricGoal.MAXIMIZE, count=2, best_values=[9, 8]),
      dict(goal=vz.ObjectiveMetricGoal.MINIMIZE, count=2, best_values=[0, 1]),
      dict(goal=vz.ObjectiveMetricGoal.MINIMIZE, count=None, best_values=[
          0,
      ]))
  def test_get_best_trials_single_objective(self, goal, count, best_values):
    runner = LocalPolicyRunner(
        vz.ProblemStatement(
            vz.SearchSpace(),
            metric_information=vz.MetricsConfig(
                [vz.MetricInformation(name='objective', goal=goal)])))

    trials = [
        vz.Trial().complete(vz.Measurement({'objective': i})) for i in range(10)
    ]
    np.random.shuffle(trials)
    runner.AddTrials(trials)

    objectives = np.array([
        t.final_measurement.metrics['objective'].value
        for t in runner.GetBestTrials(count=count)
    ])
    np.testing.assert_array_equal(objectives, np.array(best_values))

  @parameterized.parameters(
      dict(
          sin_goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          cos_goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          best_r=5),
      dict(
          sin_goal=vz.ObjectiveMetricGoal.MINIMIZE,
          cos_goal=vz.ObjectiveMetricGoal.MINIMIZE,
          best_r=1))
  def test_get_best_trials_multi_objective(self, sin_goal, cos_goal, best_r):
    runner = LocalPolicyRunner(
        vz.StudyConfig(
            vz.SearchSpace(),
            metric_information=vz.MetricsConfig([
                vz.MetricInformation(name='sin', goal=sin_goal),
                vz.MetricInformation(name='cos', goal=cos_goal)
            ])))

    def build_measurement(r, theta):
      return vz.Measurement({
          'r': r,
          'theta': theta,
          'cos': r * np.cos(theta),
          'sin': r * np.sin(theta)
      })

    # Generate many trials for each radius.
    for r in range(1, 6):
      for theta in np.linspace(0, np.pi / 2, 5):
        runner.AddTrials([vz.Trial().complete(build_measurement(r, theta))])

    # Check the radius.
    rs = np.array([
        t.final_measurement.metrics['r'].value for t in runner.GetBestTrials()
    ])
    self.assertEqual(rs.size, 5)
    np.testing.assert_array_equal(rs, np.ones_like(rs) * best_r)


if __name__ == '__main__':
  absltest.main()
