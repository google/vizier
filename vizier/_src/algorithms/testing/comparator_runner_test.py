# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for comparator_runner."""

from typing import Optional, Sequence

import numpy as np
from vizier import algorithms as vza
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.algorithms.testing import comparator_runner
from vizier.pyvizier import converters

from absl.testing import absltest
from absl.testing import parameterized


class DummyVectorizedStrategy(vb.VectorizedStrategy):

  def __init__(self, end_value: float):
    self._end_value = end_value

  def suggest(self) -> np.ndarray:
    return np.ones((5, 2))

  @property
  def suggestion_count(self) -> int:
    return 5

  @property
  def best_results(self) -> list[vb.VectorizedStrategyResult]:
    return [vb.VectorizedStrategyResult(np.ones(2), self._end_value)]

  def update(self, rewards: np.ndarray) -> None:
    pass


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


class EfficiencyConvergenceTest(absltest.TestCase):

  def test_comparison(self):
    experimenter = benchmarks.BBOBExperimenterFactory('Sphere', 3)()

    num_trials = 20

    def _baseline_designer(problem: vz.ProblemStatement) -> vza.Designer:
      return DummyDesigner(
          problem.search_space, num_trial_to_converge=num_trials)

    def _good_designer(problem: vz.ProblemStatement) -> vza.Designer:
      return DummyDesigner(
          problem.search_space, num_trial_to_converge=int(num_trials / 4))

    comparator = comparator_runner.EfficiencyComparisonTester(
        num_trials=num_trials, num_repeats=5)
    comparator.assert_better_efficiency(
        benchmarks.DesignerBenchmarkStateFactory(
            experimenter=experimenter, designer_factory=_good_designer),
        benchmarks.DesignerBenchmarkStateFactory(
            experimenter=experimenter, designer_factory=_baseline_designer),
        score_threshold=0.0)

    # Test that our baseline is worse.
    with self.assertRaises(comparator_runner.FailedComparisonTestError):  # pylint: disable=g-error-prone-assert-raises
      comparator.assert_better_efficiency(
          benchmarks.DesignerBenchmarkStateFactory(
              experimenter=experimenter, designer_factory=_baseline_designer),
          benchmarks.DesignerBenchmarkStateFactory(
              experimenter=experimenter, designer_factory=_good_designer))


class SimpleRegretConvergenceRunnerTest(parameterized.TestCase):
  """Test suite for convergence runner."""

  def setUp(self):
    super(SimpleRegretConvergenceRunnerTest, self).setUp()
    self.experimenter = benchmarks.BBOBExperimenterFactory('Sphere', 3)()
    self.converter = converters.TrialToArrayConverter.from_study_config(
        self.experimenter.problem_statement())

  @parameterized.parameters(
      {
          'candidate_num_trials': 1,
          'candidate_end_value': 5.0,
          'should_pass': True
      }, {
          'candidate_num_trials': 100,
          'candidate_end_value': 5.0,
          'should_pass': True
      }, {
          'candidate_num_trials': 1,
          'candidate_end_value': 0.0,
          'should_pass': False
      }, {
          'candidate_num_trials': 100,
          'candidate_end_value': 0.0,
          'should_pass': False
      })
  def test_designer_convergence(self, candidate_num_trials, candidate_end_value,
                                should_pass):

    def better_designer_factory(problem):
      return DummyDesigner(
          search_space=problem.search_space,
          good_value=candidate_end_value,
          bad_value=0.0,
          noise=0.0)

    def baseline_designer_factory(problem):
      return DummyDesigner(
          search_space=problem.search_space,
          good_value=0.0,
          bad_value=0.0,
          noise=0.0)

    baseline_benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        experimenter=self.experimenter,
        designer_factory=baseline_designer_factory)

    candidate_benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        experimenter=self.experimenter,
        designer_factory=better_designer_factory)

    simple_regret_test = comparator_runner.SimpleRegretComparisonTester(
        baseline_num_trials=1000,
        candidate_num_trials=candidate_num_trials,
        baseline_num_repeats=5,
        candidate_num_repeats=5,
        alpha=0.05)

    if should_pass:
      simple_regret_test.assert_benchmark_state_better_simple_regret(
          baseline_benchmark_state_factory,
          candidate_benchmark_state_factory,
      )
    else:
      with self.assertRaises(  # pylint: disable=g-error-prone-assert-raises
          comparator_runner.FailedSimpleRegretConvergenceTestError):
        simple_regret_test.assert_benchmark_state_better_simple_regret(
            baseline_benchmark_state_factory,
            candidate_benchmark_state_factory,
        )

  @parameterized.parameters({
      'candidate_end_value': 5.0,
      'should_pass': True
  }, {
      'candidate_end_value': 5.0,
      'should_pass': True
  }, {
      'candidate_end_value': 0.0,
      'should_pass': False
  }, {
      'candidate_end_value': 0.0,
      'should_pass': False
  })
  def test_optimizer_convergence(self, candidate_end_value, should_pass):

    score_fn = lambda x: np.sum(x, axis=-1)
    simple_regret_test = comparator_runner.SimpleRegretComparisonTester(
        baseline_num_trials=100,
        candidate_num_trials=100,
        baseline_num_repeats=5,
        candidate_num_repeats=5,
        alpha=0.05)

    # pylint: disable=unused-argument
    def baseline_strategy_factory(converter, count):
      return DummyVectorizedStrategy(end_value=1.0)

      # pylint: disable=unused-argument
    def candidate_strategy_factory(converter, count):
      return DummyVectorizedStrategy(end_value=candidate_end_value)

    baseline_optimizer = vb.VectorizedOptimizer(
        strategy_factory=baseline_strategy_factory)

    candidate_optimizer = vb.VectorizedOptimizer(
        strategy_factory=candidate_strategy_factory)

    if should_pass:
      simple_regret_test.assert_optimizer_better_simple_regret(
          self.converter,
          score_fn,
          baseline_optimizer,
          candidate_optimizer,
      )
    else:
      with self.assertRaises(  # pylint: disable=g-error-prone-assert-raises
          comparator_runner.FailedSimpleRegretConvergenceTestError):
        simple_regret_test.assert_optimizer_better_simple_regret(
            self.converter,
            score_fn,
            baseline_optimizer,
            candidate_optimizer,
        )


if __name__ == '__main__':
  absltest.main()
