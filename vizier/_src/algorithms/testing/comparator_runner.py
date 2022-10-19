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

"""Comparison test for algorithms using score analysis.

Ex: Typical Convergence Test Example
----------------------------------------

baseline_factory = benchmarks.BenchmarkStateFactory(...)
candidate_factory = benchmarks.BenchmarkStateFactory(...)

# Run each algorithm for 100 Trials with 5 repeats each.
comparator = comparator_runner.ComparisonTester(
        num_trials=100, num_repeats=5)
comparator.assert_converges_faster(candidate_factory, baseline_factory)

NOTE: assert_converges_faster is a generic method name that conveys the general
use of the class.
"""

import attr
from vizier import benchmarks


class FailedComparisonTestError(Exception):
  """Exception raised for comparison test fails."""


@attr.define
class ComparisonTester:
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
          benchmarks.ConvergenceCurveConverter(
              baseline_statement.metric_information.item()).convert(
                  baseline_state.algorithm.supporter.GetTrials()))
      candidate_curves.append(
          benchmarks.ConvergenceCurveConverter(
              baseline_statement.metric_information.item()).convert(
                  candidate_state.algorithm.supporter.GetTrials()))

    baseline_curve = benchmarks.ConvergenceCurve.align_xs(baseline_curves)
    candidate_curve = benchmarks.ConvergenceCurve.align_xs(candidate_curves)
    comparator = benchmarks.ConvergenceCurveComparator(baseline_curve)

    if (log_eff_score :=
        comparator.get_log_efficiency_score(candidate_curve)) < score_threshold:
      raise FailedComparisonTestError(
          f'Log efficiency score {log_eff_score} is less than {score_threshold}'
          f' when comparing algorithms: {candidate_state_factory} '
          f'vs baseline of {baseline_state_factory} for {self.num_trials} '
          f' Trials with {self.num_repeats} repeats')
