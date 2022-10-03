"""Tests for comparator_runner."""

from typing import Optional, Sequence

import numpy as np
from vizier import algorithms as vza
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.testing import comparator_runner
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest


class DummyDesigner(vza.Designer):
  """Dummy designer to control convergence."""

  def __init__(self,
               search_space: vz.SearchSpace,
               good_value: float = 0.0,
               bad_value: float = 1.0,
               noise: float = 0.1,
               num_trial_to_converge: int = 0):
    self.search_space = search_space
    self.good_value = good_value
    self.bad_value = bad_value
    self.noise = noise
    self.num_trial_to_converge = num_trial_to_converge
    self.num_trials_so_far = 0

  def update(self, delta: vza.CompletedTrials) -> None:
    self.num_trials_so_far += len(delta.completed)

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    if self.num_trials_so_far < self.num_trial_to_converge:
      parameters = {
          param.name: self.bad_value + self.noise * np.random.uniform()
          for param in self.search_space.parameters
      }
    else:
      parameters = {
          param.name: self.good_value + self.noise * np.random.uniform()
          for param in self.search_space.parameters
      }
    return [vz.TrialSuggestion(parameters)]


class ConvergenceTest(absltest.TestCase):

  def test_comparison(self):
    problem = bbob.DefaultBBOBProblemStatement(3)
    experimenter = benchmarks.NumpyExperimenter(bbob.Sphere, problem)

    num_trials = 20

    def _baseline_designer(problem: vz.ProblemStatement) -> vza.Designer:
      return DummyDesigner(
          problem.search_space, num_trial_to_converge=num_trials)

    def _good_designer(problem: vz.ProblemStatement) -> vza.Designer:
      return DummyDesigner(
          problem.search_space, num_trial_to_converge=int(num_trials / 4))

    comparator = comparator_runner.ComparisonTester(
        num_trials=num_trials, num_repeats=5)
    comparator.assert_better_efficiency(
        benchmarks.DesignerBenchmarkStateFactory(
            experimenter=experimenter, designer_factory=_good_designer),
        benchmarks.DesignerBenchmarkStateFactory(
            experimenter=experimenter, designer_factory=_baseline_designer),
        score_threshold=0.1)

    # Test that our baseline is worse.
    with self.assertRaises(comparator_runner.FailedComparisonTestError):  # pylint: disable=g-error-prone-assert-raises
      comparator.assert_better_efficiency(
          benchmarks.DesignerBenchmarkStateFactory(
              experimenter=experimenter, designer_factory=_baseline_designer),
          benchmarks.DesignerBenchmarkStateFactory(
              experimenter=experimenter, designer_factory=_good_designer))


if __name__ == '__main__':
  absltest.main()
