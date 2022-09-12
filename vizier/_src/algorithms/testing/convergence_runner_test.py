"""Test suite for convergence test functionality."""

from typing import Optional, Sequence

from vizier import algorithms as vza
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.testing import convergence_runner
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest


class DummyDesigner(vza.Designer):
  """Dummy designer to control convergence."""

  def __init__(self,
               search_space: vz.SearchSpace,
               convegence_value: float = 0.0,
               non_convegence_value: float = 1.0,
               num_trial_to_converge: int = 0):

    self.convergence_suggestion = vz.TrialSuggestion(parameters={
        param.name: convegence_value for param in search_space.parameters
    })
    self.non_convergence_suggestion = vz.TrialSuggestion(parameters={
        param.name: non_convegence_value for param in search_space.parameters
    })
    self.num_trial_to_converge = num_trial_to_converge
    self.current_trial = 0

  def update(self, delta: vza.CompletedTrials) -> None:
    pass

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    self.current_trial += 1
    if self.current_trial < self.num_trial_to_converge:
      return [self.non_convergence_suggestion]
    else:
      return [self.convergence_suggestion]


class ConvergenceTest(absltest.TestCase):

  def test_successful_convergence(self):
    problem = bbob.DefaultBBOBProblemStatement(3)
    experimenter = benchmarks.NumpyExperimenter(bbob.Sphere, problem)

    def _dummy_designer_factory(problem: vz.ProblemStatement) -> vza.Designer:
      return DummyDesigner(problem.search_space, convegence_value=0.0)

    benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        experimenter=experimenter, designer_factory=_dummy_designer_factory)

    convergence_test = convergence_runner.BenchmarkConvergenceRunner(
        benchmark_state_factory=benchmark_state_factory,
        trials_per_check=10,
        repeated_checks=5,
        success_rate_threshold=1.0,
        tolerance=1e-2)
    convergence_test.assert_converges()

  def test_failed_convergence(self):
    problem = bbob.DefaultBBOBProblemStatement(3)
    experimenter = benchmarks.NumpyExperimenter(bbob.Sphere, problem)

    def _dummy_designer_factory(problem: vz.ProblemStatement) -> vza.Designer:
      return DummyDesigner(problem.search_space, convegence_value=1.0)

    benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        experimenter=experimenter, designer_factory=_dummy_designer_factory)

    convergence_test = convergence_runner.BenchmarkConvergenceRunner(
        benchmark_state_factory=benchmark_state_factory,
        trials_per_check=10,
        repeated_checks=5,
        success_rate_threshold=1.0,
        tolerance=1e-2)

    assert_call = convergence_test.assert_converges
    with self.assertRaises(convergence_runner.FailedConvergenceTestError):
      assert_call()


if __name__ == '__main__':
  absltest.main()
