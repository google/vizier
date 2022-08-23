"""Tests for base_runner."""

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.benchmarks.experimenters import numpy_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob
from vizier._src.benchmarks.runners import benchmark_runner

from absl.testing import absltest


class BaseRunnerTest(absltest.TestCase):

  def testSimpleSuggestAndEvaluate(self):
    runner = benchmark_runner.BenchmarkRunner(
        benchmark_subroutines=[
            benchmark_runner.GenerateSuggestions(),
            benchmark_runner.EvaluateActiveTrials()
        ],
        num_repeats=7)

    dim = 10
    experimenter = numpy_experimenter.NumpyExperimenter(
        bbob.Sphere, bbob.DefaultBBOBProblemStatement(dim))

    def designer_factory(config: vz.ProblemStatement):
      return random.RandomDesigner(config.search_space, seed=5)

    benchmark_state = benchmark_runner.BenchmarkState.from_designer_factory(
        designer_factory=designer_factory, experimenter=experimenter)

    runner.run(benchmark_state)
    self.assertEmpty(
        benchmark_state.algorithm.supporter.GetTrials(
            status_matches=vz.TrialStatus.ACTIVE))
    all_trials = benchmark_state.algorithm.supporter.GetTrials()
    self.assertLen(all_trials, 7)
    self.assertEqual(all_trials[0].status, vz.TrialStatus.COMPLETED)


if __name__ == '__main__':
  absltest.main()
