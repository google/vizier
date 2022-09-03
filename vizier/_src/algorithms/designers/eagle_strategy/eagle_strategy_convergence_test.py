"""Convergence test for Eagle Strategy."""

import numpy as np
from vizier import algorithms as vza
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.testing import convergence_runner
from vizier._src.benchmarks.experimenters import shifting_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest


# TODO: move this class to eagle_strategy_test.py
class EagleStrategyConvergenceTest(absltest.TestCase):

  def _eagle_designer_factory(self,
                              problem: vz.ProblemStatement) -> vza.Designer:
    return eagle_strategy.EagleStrategyDesiger(problem)

  def test_convergence_1d(self):
    problem = bbob.DefaultBBOBProblemStatement(1)
    experimenter = shifting_experimenter.ShiftingExperimenter(
        exptr=benchmarks.NumpyExperimenter(bbob.Sphere, problem),
        shift=np.random.random())
    # Create a benchmark state factory based on a designer factory
    benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=self._eagle_designer_factory,
        experimenter=experimenter,
    )
    # Run Convergence test
    convergence_test = convergence_runner.BenchmarkConvergenceRunner(
        benchmark_state_factory=benchmark_state_factory,
        trials_per_check=1000,
        repeated_checks=5,
        success_rate_threshold=0.8,
        tolerance=1e-2)
    convergence_test.assert_converges()

  def test_convergence_5d(self):
    problem = bbob.DefaultBBOBProblemStatement(5)
    experimenter = shifting_experimenter.ShiftingExperimenter(
        exptr=benchmarks.NumpyExperimenter(bbob.Sphere, problem),
        shift=np.random.random(5))
    # Create a benchmark state factory based on a designer factory
    benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=self._eagle_designer_factory,
        experimenter=experimenter,
    )
    # Run Convergence test
    convergence_test = convergence_runner.BenchmarkConvergenceRunner(
        benchmark_state_factory=benchmark_state_factory,
        trials_per_check=1000,
        repeated_checks=5,
        success_rate_threshold=0.8,
        tolerance=1e-2)
    convergence_test.assert_converges()


if __name__ == '__main__':
  absltest.main()
