# Copyright 2024 Google LLC.
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

from __future__ import annotations

"""Tests for comparator_runner."""

from typing import Optional, Sequence

import jax
from jax import numpy as jnp
import numpy as np
from vizier import algorithms as vza
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.algorithms.testing import comparator_runner
from vizier._src.jax import types
from vizier.benchmarks import experimenters
from vizier.pyvizier import converters

from absl.testing import absltest
from absl.testing import parameterized


class FakeVectorizedStrategy(vb.VectorizedStrategy):
  """Dummy vectorized strategy to control convergence."""

  def __init__(
      self,
      converter: converters.TrialToModelInputConverter,
      good_value: float = 1.0,
      bad_value: float = 0.0,
      num_trial_to_converge: int = 0,
  ):
    self.converter = converter
    self.good_value = good_value
    self.bad_value = bad_value
    self.num_trial_to_converge = num_trial_to_converge
    self.num_trials_so_far = 0

  def init_state(
      self,
      seed: jax.Array,
      n_parallel: int = 1,
      *,
      prior_features: Optional[vb.VectorizedOptimizerInput] = None,
      prior_rewards: Optional[types.Array] = None,
  ) -> None:
    return

  def suggest(
      self,
      seed: jax.Array,
      state: None,
      n_parallel: int = 1,
  ) -> vb.VectorizedOptimizerInput:
    continuous_output_len = sum(
        [spec.num_dimensions for spec in self.converter.output_specs.continuous]
    )
    categorical_output_len = len(self.converter.output_specs.categorical)
    shape = (1, n_parallel, continuous_output_len)
    if self.num_trials_so_far < self.num_trial_to_converge:
      continuous = jnp.ones(shape) * self.bad_value
    else:
      continuous = jnp.ones(shape) * self.good_value
    return vb.VectorizedOptimizerInput(
        continuous=continuous,
        categorical=jnp.zeros(
            (1, n_parallel, categorical_output_len), dtype=types.INT_DTYPE
        ),
    )

  @property
  def suggestion_batch_size(self) -> int:
    return 1

  def update(
      self,
      seed: jax.Array,
      state: None,
      batch_features: vb.VectorizedOptimizerInput,
      batch_rewards: types.Array,
  ) -> None:
    pass


class FakeDesigner(vza.Designer):
  """Dummy designer to control convergence."""

  def __init__(
      self,
      search_space: vz.SearchSpace,
      *,
      good_value: float = 1.0,
      bad_value: float = 0.0,
      noise: float = 0.1,
      num_trial_to_converge: int = 0,
      seed: Optional[int] = None,
  ):
    self.search_space = search_space
    self.good_value = good_value
    self.bad_value = bad_value
    self.noise = noise
    self.num_trial_to_converge = num_trial_to_converge
    self.num_trials_so_far = 0

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    del all_active
    self.num_trials_so_far += len(completed.trials)

  def suggest(
      self, count: Optional[int] = None
  ) -> Sequence[vz.TrialSuggestion]:
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
    experimenter = experimenters.BBOBExperimenterFactory('Sphere', 3)()

    num_trials = 20

    def _baseline_designer(
        problem: vz.ProblemStatement, seed: Optional[int] = None
    ) -> vza.Designer:
      return FakeDesigner(
          problem.search_space,
          num_trial_to_converge=num_trials,
          good_value=0.0,
          bad_value=1.0,
          seed=seed,
      )

    def _good_designer(
        problem: vz.ProblemStatement, seed: Optional[int] = None
    ) -> vza.Designer:
      return FakeDesigner(
          problem.search_space,
          good_value=0.0,
          bad_value=1.0,
          num_trial_to_converge=int(num_trials / 4),
          seed=seed,
      )

    comparator = comparator_runner.EfficiencyComparisonTester(
        num_trials=num_trials, num_repeats=5
    )
    comparator.assert_better_efficiency(
        benchmarks.DesignerBenchmarkStateFactory(
            experimenter=experimenter, designer_factory=_good_designer
        ),
        benchmarks.DesignerBenchmarkStateFactory(
            experimenter=experimenter, designer_factory=_baseline_designer
        ),
        score_threshold=0.3,
    )

    # Test that our baseline is worse.
    with self.assertRaises(comparator_runner.FailedComparisonTestError):  # pylint: disable=g-error-prone-assert-raises
      comparator.assert_better_efficiency(
          benchmarks.DesignerBenchmarkStateFactory(
              experimenter=experimenter, designer_factory=_baseline_designer
          ),
          benchmarks.DesignerBenchmarkStateFactory(
              experimenter=experimenter, designer_factory=_good_designer
          ),
          score_threshold=-0.1,
      )


class SimpleRegretConvergenceRunnerTest(parameterized.TestCase):
  """Test suite for convergence runner."""

  def setUp(self):
    super(SimpleRegretConvergenceRunnerTest, self).setUp()
    self.experimenter = experimenters.BBOBExperimenterFactory('Sphere', 3)()
    self.converter = converters.TrialToModelInputConverter.from_problem(
        self.experimenter.problem_statement()
    )

  @parameterized.parameters(
      {
          'candidate_num_trials': 1,
          'candidate_x_value': 5.0,
          'goal': vz.ObjectiveMetricGoal.MAXIMIZE,
          'should_pass': True,
      },
      {
          'candidate_num_trials': 100,
          'candidate_x_value': 5.0,
          'goal': vz.ObjectiveMetricGoal.MAXIMIZE,
          'should_pass': True,
      },
      {
          'candidate_num_trials': 1,
          'candidate_x_value': 0.0,
          'goal': vz.ObjectiveMetricGoal.MAXIMIZE,
          'should_pass': False,
      },
      {
          'candidate_num_trials': 100,
          'candidate_x_value': 0.0,
          'goal': vz.ObjectiveMetricGoal.MAXIMIZE,
          'should_pass': False,
      },
      {
          'candidate_num_trials': 1,
          'candidate_x_value': 5.0,
          'goal': vz.ObjectiveMetricGoal.MINIMIZE,
          'should_pass': False,
      },
      {
          'candidate_num_trials': 100,
          'candidate_x_value': 5.0,
          'goal': vz.ObjectiveMetricGoal.MINIMIZE,
          'should_pass': False,
      },
      {
          'candidate_num_trials': 1,
          'candidate_x_value': 0.0,
          'goal': vz.ObjectiveMetricGoal.MINIMIZE,
          'should_pass': True,
      },
      {
          'candidate_num_trials': 100,
          'candidate_x_value': 0.0,
          'goal': vz.ObjectiveMetricGoal.MINIMIZE,
          'should_pass': True,
      },
  )
  def test_designer_convergence(
      self, candidate_num_trials, candidate_x_value, goal, should_pass
  ):
    def _better_designer_factory(problem, seed):
      return FakeDesigner(
          search_space=problem.search_space,
          good_value=candidate_x_value,
          bad_value=0.0,
          noise=0.0,
          seed=seed,
      )

    def _baseline_designer_factory(problem, seed):
      return FakeDesigner(
          search_space=problem.search_space,
          good_value=1.0,
          bad_value=1.0,
          noise=0.0,
          seed=seed,
      )

    baseline_benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        experimenter=self.experimenter,
        designer_factory=_baseline_designer_factory,
    )

    candidate_benchmark_state_factory = (
        benchmarks.DesignerBenchmarkStateFactory(
            experimenter=self.experimenter,
            designer_factory=_better_designer_factory,
        )
    )

    simple_regret_test = comparator_runner.SimpleRegretComparisonTester(
        baseline_num_trials=1000,
        candidate_num_trials=candidate_num_trials,
        baseline_suggestion_batch_size=1,
        candidate_suggestion_batch_size=1,
        baseline_num_repeats=5,
        candidate_num_repeats=5,
        alpha=0.05,
        goal=goal,
    )

    if should_pass:
      simple_regret_test.assert_benchmark_state_better_simple_regret(
          baseline_benchmark_state_factory,
          candidate_benchmark_state_factory,
      )
    else:
      with self.assertRaises(  # pylint: disable=g-error-prone-assert-raises
          comparator_runner.FailedSimpleRegretConvergenceTestError
      ):
        simple_regret_test.assert_benchmark_state_better_simple_regret(
            baseline_benchmark_state_factory,
            candidate_benchmark_state_factory,
        )

  @parameterized.parameters(
      {
          'candidate_x_value': 5.0,
          'goal': vz.ObjectiveMetricGoal.MAXIMIZE,
          'should_pass': True,
      },
      {
          'candidate_x_value': 0.0,
          'goal': vz.ObjectiveMetricGoal.MAXIMIZE,
          'should_pass': False,
      },
      {
          'candidate_x_value': 5.0,
          'goal': vz.ObjectiveMetricGoal.MINIMIZE,
          'should_pass': False,
      },
      {
          'candidate_x_value': 0.0,
          'goal': vz.ObjectiveMetricGoal.MINIMIZE,
          'should_pass': True,
      },
  )
  def test_optimizer_convergence(self, candidate_x_value, goal, should_pass):
    score_fn = lambda x, _: jnp.sum(x.continuous.padded_array, axis=-1)
    simple_regret_test = comparator_runner.SimpleRegretComparisonTester(
        baseline_num_trials=100,
        candidate_num_trials=100,
        baseline_suggestion_batch_size=1,
        candidate_suggestion_batch_size=1,
        baseline_num_repeats=5,
        candidate_num_repeats=5,
        alpha=0.05,
        goal=goal,
    )

    # pylint: disable=unused-argument
    def _baseline_strategy_factory(converter, suggestion_batch_size):
      return FakeVectorizedStrategy(
          converter=converter,
          good_value=1.0,
          bad_value=1.0,
          num_trial_to_converge=0,
      )

      # pylint: disable=unused-argument

    def _candidate_strategy_factory(converter, suggestion_batch_size):
      return FakeVectorizedStrategy(
          converter=converter,
          good_value=candidate_x_value,
          bad_value=0.0,
          num_trial_to_converge=0,
      )

    if should_pass:
      simple_regret_test.assert_optimizer_better_simple_regret(
          self.converter,
          score_fn,
          _baseline_strategy_factory,
          _candidate_strategy_factory,
      )
    else:
      with self.assertRaises(  # pylint: disable=g-error-prone-assert-raises
          comparator_runner.FailedSimpleRegretConvergenceTestError
      ):
        simple_regret_test.assert_optimizer_better_simple_regret(
            self.converter,
            score_fn,
            _baseline_strategy_factory,
            _candidate_strategy_factory,
        )


if __name__ == '__main__':
  absltest.main()
