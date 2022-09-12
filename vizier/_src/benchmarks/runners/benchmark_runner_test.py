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

    benchmark_state_factory = benchmark_runner.DesignerBenchmarkStateFactory(
        designer_factory=designer_factory, experimenter=experimenter)

    benchmark_state = benchmark_state_factory.create()

    runner.run(benchmark_state)
    self.assertEmpty(
        benchmark_state.algorithm.supporter.GetTrials(
            status_matches=vz.TrialStatus.ACTIVE))
    all_trials = benchmark_state.algorithm.supporter.GetTrials()
    self.assertLen(all_trials, 7)
    self.assertEqual(all_trials[0].status, vz.TrialStatus.COMPLETED)

  def testGenerateAndEvaluate(self):
    num_suggestions = 3
    num_iterations = 7
    runner = benchmark_runner.BenchmarkRunner(
        benchmark_subroutines=[
            benchmark_runner.GenerateAndEvaluate(
                num_suggestions=num_suggestions)
        ],
        num_repeats=num_iterations)

    dim = 10
    experimenter = numpy_experimenter.NumpyExperimenter(
        bbob.Sphere, bbob.DefaultBBOBProblemStatement(dim))

    def designer_factory(config: vz.ProblemStatement):
      return random.RandomDesigner(config.search_space, seed=5)

    benchmark_state_factory = benchmark_runner.DesignerBenchmarkStateFactory(
        experimenter=experimenter, designer_factory=designer_factory)

    benchmark_state = benchmark_state_factory.create()

    runner.run(benchmark_state)
    self.assertEmpty(
        benchmark_state.algorithm.supporter.GetTrials(
            status_matches=vz.TrialStatus.ACTIVE))
    all_trials = benchmark_state.algorithm.supporter.GetTrials()
    self.assertLen(all_trials, num_suggestions * num_iterations)
    for trial in all_trials:
      self.assertEqual(trial.status, vz.TrialStatus.COMPLETED)


if __name__ == '__main__':
  absltest.main()
