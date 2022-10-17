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

"""Runs a simple-regret convergence test.

The test is based on comparing the results to a random search. It performs
a statistical test to compute the probability (p-value) of observing the
best result under the null hypothesis of running a random search and with the
given amount of evaluations. The p-value is compared against a pre-determined
significnce-level (alpha) to decide if the test has passed or not.

Test arguments
--------------
- optimum_features/optimum trial: The objective function's optima point against
which the algorithm convergence is evaluated.

- evaluations: The number of evaluations the algorithm perform.

- num_repeats: The total number of individual convergence tests to perform. The
entire convergence test pass if all individual convergence tests pass.

- Alpha: The significance level of each individual convergence test, which is
indenpendent of the total number of individual tests performed. There's no
significance level correction as we require all sub-tests to pass.


Ex: Typical Convergence Test on Designer
----------------------------------------
class MyDesignerConvegenceTest(absltest.TestCase):

  # Test function assuming experimenter and designer factory.
  def test_convergence(self):

    # Create a benchmark state factory based on a designer factory
    benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=designer_factory,
        experimenter=experimenter,
    )
    # Run the convergence test.
    convergence_runner.assert_benchmark_state_converges_better_than_random(
        benchmark_state_factory=benchmark_state_factory,
        optimum_trial=optimum_trial,
        evaluations=1000,
        alpha=0.05,
        num_repeats=5,
    )
"""

import logging
from typing import Tuple

import numpy as np
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.algorithms.testing import p_value as pv
from vizier.pyvizier import converters


class FailedSimpleRegretConvergenceTestError(Exception):
  """Exception raised for simple-regret convergence test fails."""


def assert_optimizer_converges_better_than_random(
    converter: converters.TrialToArrayConverter,
    optimizer: vb.VectorizedOptimizer,
    score_fn: vb.BatchArrayScoreFunction,
    optimum_features: np.ndarray,
    evaluations: int,
    alpha: float = 0.05,
    num_repeats: int = 3) -> None:
  """Runs simple-regret convergence test on vectorized optimizers."""
  success_count = 0
  for _ in range(num_repeats):
    optimizer.optimize(converter, score_fn)
    best_features = optimizer.strategy.best_features_results[0].features
    best_reward = optimizer.strategy.best_features_results[0].reward
    p_value_continuous, p_value_categorical = pv.compute_array_p_values(
        converter, evaluations, best_features, optimum_features)
    msg = (
        f'P-value continuous: {p_value_continuous}. P-value categorical: {p_value_categorical}. '
        f'Alpha={alpha}. Best reward: {best_reward}.\nOptimum features:\n{optimum_features}'
        f'\nBest features:\n{best_features}.\nAbsolute diff:\n{np.abs(optimum_features - best_features)}'
    )
    if p_value_continuous <= alpha and p_value_categorical <= alpha:
      success_count += 1
      logging.info('Convergence test PASSED:\n %s', msg)
    else:
      logging.warning('Convergence test FAILED:\n %s', msg)

  if success_count < num_repeats:
    raise FailedSimpleRegretConvergenceTestError(
        f'Only {success_count} of the {num_repeats} convergence checks passed.')


def assert_benchmark_state_converges_better_than_random(
    benchmark_state_factory: benchmarks.BenchmarkStateFactory,
    optimum_trial: vz.Trial,
    evaluations: int,
    alpha: float = 0.05,
    num_repeats: int = 3,
) -> None:
  """Runs simple-regret convergence test for benchmark state."""
  success_count = 0
  for _ in range(num_repeats):
    benchmark_state = benchmark_state_factory()

    runner = benchmarks.BenchmarkRunner(
        benchmark_subroutines=[benchmarks.GenerateAndEvaluate()],
        num_repeats=evaluations)
    runner.run(benchmark_state)
    best_trial, best_metric = _get_best_metric_and_trial(benchmark_state)
    p_value_continuous, p_value_categorical = pv.compute_trial_p_values(
        benchmark_state.experimenter.problem_statement().search_space,
        evaluations, best_trial, optimum_trial)
    msg = (
        f'P-value continuous: {p_value_continuous}. P-value categorical: {p_value_categorical}. '
        f'Alpha={alpha}.\nOptimum trial:\n{optimum_trial}.\nBest trial:\n{best_trial}.\nBest metric:\n{best_metric}'
    )
    if p_value_continuous <= alpha and p_value_categorical <= alpha:
      success_count += 1
      logging.info('Convergence test PASSED:\n %s', msg)
    else:
      logging.warning('Convergence test FAILED:\n %s', msg)

  if success_count < num_repeats:
    raise FailedSimpleRegretConvergenceTestError(
        f'Only {success_count} of the {num_repeats} convergence checks passed.'
        f'\n{msg}')


def _get_best_metric_and_trial(
    benchmark_state: benchmarks.BenchmarkState) -> Tuple[vz.Trial, float]:
  """Returns the best trial and metric value after the benchmark has run."""
  best_trial = benchmark_state.algorithm.supporter.GetBestTrials(count=1)[0]
  metric_name = benchmark_state.experimenter.problem_statement(
  ).single_objective_metric_name
  best_metric = best_trial.final_measurement.metrics[metric_name].value
  return best_trial, best_metric
