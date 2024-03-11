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

"""Comparison test for algorithms using score analysis.

Ex: Typical Efficiency Convergence Test Example
-----------------------------------------------
baseline_factory = benchmarks.BenchmarkStateFactory(...)
candidate_factory = benchmarks.BenchmarkStateFactory(...)

# Run each algorithm for 100 Trials with 5 repeats each.
comparator = comparator_runner.EfficiencyComparisonTester(
        num_trials=100, num_repeats=5)
comparator.assert_better_efficiency(candidate_factory, baseline_factory)

NOTE: assert_converges_faster is a generic method name that conveys the general
use of the class.
"""

from absl import logging
import attr
from jax import random
import numpy as np
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.benchmarks.analyzers import simple_regret_score
from vizier.benchmarks import analyzers
from vizier.pyvizier import converters


class FailedComparisonTestError(Exception):
  """Exception raised for comparison test fails."""


class FailedSimpleRegretConvergenceTestError(Exception):
  """Exception raised for simple-regret convergence test fails."""


@attr.define
class EfficiencyComparisonTester:
  """Comparison test between algorithms using analysis scores."""
  num_trials: int = attr.field(
      default=1, validator=attr.validators.instance_of(int))
  num_repeats: int = attr.field(
      default=1, validator=attr.validators.instance_of(int))

  def assert_better_efficiency(
      self,
      candidate_state_factory: benchmarks.BenchmarkStateFactory,
      baseline_state_factory: benchmarks.BenchmarkStateFactory,
      score_threshold: float = 0.0) -> None:
    """Asserts that candidate is better than baseline via log_eff_score."""
    # TODO: Consider making this more flexible with more runners
    # And enable multimetric.
    runner = benchmarks.BenchmarkRunner(
        benchmark_subroutines=[benchmarks.GenerateAndEvaluate()],
        num_repeats=self.num_trials)

    baseline_curves = []
    candidate_curves = []
    for _ in range(self.num_repeats):
      baseline_state = baseline_state_factory()
      candidate_state = candidate_state_factory()

      baseline_statement = baseline_state.experimenter.problem_statement()
      if len(baseline_statement.metric_information) > 1:
        raise ValueError('Support for multimetric is not yet')
      if baseline_statement != (
          candidate_statement :=
          candidate_state.experimenter.problem_statement()):
        raise ValueError('Comparison tests done for different statements: '
                         f'{baseline_statement} vs {candidate_statement}')

      runner.run(baseline_state)
      runner.run(candidate_state)
      baseline_curves.append(
          analyzers.ConvergenceCurveConverter(
              baseline_statement.metric_information.item()
          ).convert(baseline_state.algorithm.supporter.GetTrials())
      )
      candidate_curves.append(
          analyzers.ConvergenceCurveConverter(
              baseline_statement.metric_information.item()
          ).convert(candidate_state.algorithm.supporter.GetTrials())
      )

    baseline_curve = analyzers.ConvergenceCurve.align_xs(
        baseline_curves, interpolate_repeats=True
    )[0]
    candidate_curve = analyzers.ConvergenceCurve.align_xs(
        candidate_curves, interpolate_repeats=True
    )[0]
    comparator = analyzers.LogEfficiencyConvergenceCurveComparator(
        baseline_curve=baseline_curve, compared_curve=candidate_curve
    )

    if (log_eff_score := comparator.score()) < score_threshold:
      raise FailedComparisonTestError(
          f'Log efficiency score {log_eff_score} is less than {score_threshold}'
          f' when comparing algorithms: {candidate_state_factory} '
          f'vs baseline of {baseline_state_factory} for {self.num_trials} '
          f' Trials with {self.num_repeats} repeats')


@attr.define(kw_only=True)
class SimpleRegretComparisonTester:
  """Compare two algorithms by their simple regrets.

  The test runs the baseline algorithm 'baseline_num_repeats' times each with
  'baseline_num_trials' trials and computes the simple regret in each trial,
  and similarly for the candidate algorithm.

  A one-sided T-test is performed to compute the p-value of observing the
  difference in the simple regret sample means. The T-test score (p-value) is
  compared against the significance level (alpha) to determine if the test
  passed.
  """
  baseline_num_trials: int
  candidate_num_trials: int
  baseline_suggestion_batch_size: int
  candidate_suggestion_batch_size: int
  baseline_num_repeats: int
  candidate_num_repeats: int
  alpha: float = attr.field(
      validator=attr.validators.and_(
          attr.validators.ge(0), attr.validators.le(0.1)),
      default=0.05)
  goal: vz.ObjectiveMetricGoal

  def assert_optimizer_better_simple_regret(
      self,
      converter: converters.TrialToModelInputConverter,
      score_fn: vb.ArrayScoreFunction,
      baseline_strategy_factory: vb.VectorizedStrategyFactory,
      candidate_strategy_factory: vb.VectorizedStrategyFactory,
  ) -> None:
    """Assert if candidate optimizer has better simple regret than the baseline.
    """
    baseline_obj_values = []
    candidate_obj_values = []

    baseline_optimizer_factory = vb.VectorizedOptimizerFactory(
        baseline_strategy_factory,
        suggestion_batch_size=self.baseline_suggestion_batch_size,
        max_evaluations=self.baseline_num_trials,
    )
    candidate_optimizer_factory = vb.VectorizedOptimizerFactory(
        candidate_strategy_factory,
        suggestion_batch_size=self.candidate_suggestion_batch_size,
        max_evaluations=self.candidate_num_trials,
    )
    baseline_optimizer = baseline_optimizer_factory(converter)
    candidate_optimizer = candidate_optimizer_factory(converter)

    for i in range(self.baseline_num_repeats):
      res = baseline_optimizer(score_fn, count=1, seed=random.PRNGKey(i))  # pytype: disable=wrong-arg-types
      trial = vb.best_candidates_to_trials(res, converter)
      baseline_obj_values.append(
          trial[0].final_measurement_or_die.metrics['acquisition'].value
      )

    for i in range(self.candidate_num_repeats):
      res = candidate_optimizer(score_fn, count=1, seed=random.PRNGKey(i))  # pytype: disable=wrong-arg-types
      trial = vb.best_candidates_to_trials(res, converter)
      candidate_obj_values.append(
          trial[0].final_measurement_or_die.metrics['acquisition'].value
      )

    self._conclude_test(baseline_obj_values, candidate_obj_values)

  def assert_benchmark_state_better_simple_regret(
      self,
      baseline_benchmark_state_factory: benchmarks.BenchmarkStateFactory,
      candidate_benchmark_state_factory: benchmarks.BenchmarkStateFactory,
  ) -> None:
    """Runs simple-regret convergence test for benchmark state."""

    def _run_one(benchmark_state_factory: benchmarks.BenchmarkStateFactory,
                 num_trials: int, batch_size: int, seed: int) -> float:
      """Run one benchmark run and returns simple regret."""
      benchmark_state = benchmark_state_factory(seed=seed)
      baseline_runner = benchmarks.BenchmarkRunner(
          benchmark_subroutines=[benchmarks.GenerateAndEvaluate(batch_size)],
          num_repeats=num_trials // batch_size)
      baseline_runner.run(benchmark_state)
      # Extract best metric
      best_trial = benchmark_state.algorithm.supporter.GetBestTrials(count=1)[0]
      metric_name = benchmark_state.experimenter.problem_statement(
      ).single_objective_metric_name
      return best_trial.final_measurement_or_die.metrics[metric_name].value

    baseline_obj_values = []
    candidate_obj_values = []

    for idx in range(self.baseline_num_repeats):
      baseline_obj_values.append(
          _run_one(
              benchmark_state_factory=baseline_benchmark_state_factory,
              num_trials=self.baseline_num_trials,
              batch_size=self.baseline_suggestion_batch_size,
              seed=idx))

    for idx in range(self.candidate_num_repeats):
      candidate_obj_values.append(
          _run_one(
              benchmark_state_factory=candidate_benchmark_state_factory,
              num_trials=self.candidate_num_trials,
              batch_size=self.candidate_suggestion_batch_size,
              seed=idx))
    self._conclude_test(baseline_obj_values, candidate_obj_values)

  def _conclude_test(self, baseline_obj_values: list[float],
                     candidate_obj_values: list[float]) -> None:
    """Concludes test based on baseline and candidate objective func values."""

    p_value = simple_regret_score.t_test_mean_score(baseline_obj_values,
                                                    candidate_obj_values,
                                                    self.goal)
    msg = self._generate_summary(baseline_obj_values, candidate_obj_values,
                                 p_value)
    if p_value <= self.alpha:
      logging.info('Convergence test PASSED:\n %s', msg)
    else:
      raise FailedSimpleRegretConvergenceTestError(msg)

  def _generate_summary(
      self,
      baseline_obj_values: list[float],
      candidate_obj_values: list[float],
      p_value: float,
  ) -> str:
    """Generate summary message."""
    baseline_mean = np.mean(baseline_obj_values)
    baseline_std = np.std(baseline_obj_values)
    candidate_mean = np.mean(candidate_obj_values)
    candidate_std = np.std(candidate_obj_values)
    return (f'\nObjective goal={self.goal.name}'
            f'\nP-value={p_value}'
            f'\nAlpha={self.alpha}'
            f'\nBaseline Objective Mean: {baseline_mean}'
            f'\nBaseline Objective Std: {baseline_std}'
            f'\nCandidate Objective Mean: {candidate_mean}'
            f'\nCandidate Objective Std: {candidate_std}'
            f'\nBaseline Objective Scores: {baseline_obj_values}'
            f'\nCandidate Objective Scores: {candidate_obj_values}')
