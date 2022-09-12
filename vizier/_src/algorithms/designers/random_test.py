"""Tests for random."""

import numpy as np
from vizier import algorithms as vza
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.testing import convergence_runner
from vizier._src.algorithms.testing import test_runners
from vizier._src.benchmarks.experimenters import shifting_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob
from vizier.testing import test_studies

from absl.testing import absltest


class RandomTest(absltest.TestCase):

  def test_on_flat_space(self):
    config = vz.ProblemStatement(test_studies.flat_space_with_all_types())
    designer = random.RandomDesigner(config.search_space, seed=None)
    self.assertLen(
        test_runners.run_with_random_metrics(
            designer, config, iters=50, batch_size=1), 50)

  def test_reproducible_random(self):
    config = vz.ProblemStatement(test_studies.flat_space_with_all_types())
    designer = random.RandomDesigner(config.search_space, seed=5)
    t1 = designer.suggest(10)

    designer = random.RandomDesigner(config.search_space, seed=5)
    t2 = designer.suggest(10)
    self.assertEqual(t1, t2)

  def test_convergence_1d(self):
    problem = bbob.DefaultBBOBProblemStatement(1)
    experimenter = shifting_experimenter.ShiftingExperimenter(
        exptr=benchmarks.NumpyExperimenter(bbob.Sphere, problem),
        shift=np.random.random())

    def _random_designer_factory(problem: vz.ProblemStatement) -> vza.Designer:
      return random.RandomDesigner(problem.search_space)

    benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=_random_designer_factory,
        experimenter=experimenter,
    )
    convergence_test = convergence_runner.BenchmarkConvergenceRunner(
        benchmark_state_factory=benchmark_state_factory,
        trials_per_check=5000,
        repeated_checks=5,
        success_rate_threshold=0.6,
        tolerance=1.0)
    convergence_test.assert_converges()


if __name__ == '__main__':
  absltest.main()
