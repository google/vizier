"""Tests for emukit."""

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import emukit
from vizier._src.algorithms.testing import test_runners
from vizier.testing import test_studies

from absl.testing import absltest


class EmukitTest(absltest.TestCase):

  def test_on_flat_space(self):
    config = vz.StudyConfig(
        search_space=test_studies.flat_space_with_all_types(),
        metric_information=[
            vz.MetricInformation(
                name='x1', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
        ])
    designer = emukit.EmukitDesigner(
        config, num_random_samples=10, metadata_ns='emukit')
    trials = test_runners.run_with_random_metrics(
        designer,
        config,
        iters=15,
        batch_size=1,
        verbose=1,
        validate_parameters=True)
    self.assertLen(trials, 15)

    for t in trials[:10]:
      self.assertEqual(
          t.metadata.ns('emukit')['source'],
          'random',
          msg=f'Trial {t} should be gerated at random.')

    for t in trials[10:]:
      self.assertEqual(
          t.metadata.ns('emukit')['source'],
          'bayesopt',
          msg=f'Trial {t} should be gerated by bayesopt.')


if __name__ == '__main__':
  absltest.main()
