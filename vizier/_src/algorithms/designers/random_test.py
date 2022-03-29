"""Tests for random."""

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.testing import test_runners
from vizier.testing import test_studies

from absl.testing import absltest


class RandomTest(absltest.TestCase):

  def test_on_flat_space(self):
    config = vz.StudyConfig(test_studies.flat_space_with_all_types())
    designer = random.RandomDesigner(config.search_space, seed=None)
    self.assertLen(
        test_runners.run_with_random_metrics(
            designer, config, iters=50, batch_size=1), 50)

  def test_reproducible_random(self):
    config = vz.StudyConfig(test_studies.flat_space_with_all_types())
    designer = random.RandomDesigner(config.search_space, seed=5)
    t1 = designer.suggest(10)

    designer = random.RandomDesigner(config.search_space, seed=5)
    t2 = designer.suggest(10)
    self.assertEqual(t1, t2)


if __name__ == '__main__':
  absltest.main()
