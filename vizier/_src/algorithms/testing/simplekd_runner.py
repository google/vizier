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

"""SimpleKD convergence tests."""

from absl import logging
import attrs
from vizier import algorithms as vza
from vizier._src.benchmarks.experimenters.synthetic import simplekd
from vizier._src.benchmarks.runners import benchmark_runner
from vizier._src.benchmarks.runners import benchmark_state


class FailedSimpleKDConvergenceTestError(Exception):
  """Exception raised for simplekd convergence test fails."""


@attrs.define(kw_only=True)
class SimpleKDConvergenceTester:
  """The class tests designers convergence on SimpleKD problem."""

  # The SimpleKD problem type.
  best_category: simplekd.SimpleKDCategory

  # A designer factory to generate different instances of the designer.
  designer_factory: vza.DesignerFactory[vza.Designer]

  # The number of the suggested trials in each run.
  num_trials: int

  # The maximum objective value relative error to determine convergence.
  max_relative_error: float

  # The number of repeated runs used to obtain reliable results.
  num_repeats: int

  # The target number of converged runs.
  target_num_convergence: int

  # The number of designer trials suggested in each repeat.
  batch_size: int = 1

  # The number of float parameters.
  num_float_param: int = 1

  # The number of discrete parameters.
  num_discrete_param: int = 1

  # The number of integer parameters.
  num_int_param: int = 1

  # Whether to use random seeds when instantiating the designer.
  is_deterministic: bool = True

  def assert_convergence(self) -> None:
    """Run the convergence test."""
    exptr = simplekd.SimpleKDExperimenter(
        self.best_category,
        num_float_param=self.num_float_param,
        num_discrete_param=self.num_discrete_param,
        num_int_param=self.num_int_param,
        output_relative_error=True,
    )
    runner = benchmark_runner.BenchmarkRunner(
        benchmark_subroutines=[
            benchmark_runner.GenerateAndEvaluate(batch_size=self.batch_size),
        ],
        num_repeats=self.num_trials // self.batch_size,
    )
    num_success = 0
    for seed in range(self.num_repeats):
      state = benchmark_state.BenchmarkState(
          exptr,
          benchmark_state.PolicySuggester.from_designer_factory(
              problem=exptr.problem_statement(),
              designer_factory=self.designer_factory,
              seed=seed if self.is_deterministic else None,
          ),
      )
      runner.run(state)
      best_trial = state.algorithm.supporter.GetBestTrials(count=1)[0]
      metric_name = exptr.problem_statement().metric_information.item().name
      best_err = best_trial.final_measurement_or_die.metrics[metric_name].value
      if best_err <= self.max_relative_error:
        num_success += 1
    if num_success < self.target_num_convergence:
      raise FailedSimpleKDConvergenceTestError(
          'The simplekd convergence test failed. The number of converged runs'
          f' is {num_success}, though expected at least'
          f' {self.target_num_convergence} converged runs.'
      )
    else:
      logging.info(
          'The simplekd convergence test passed. The number of converged runs'
          ' is %s, and the expected number of converged runs is %s.',
          num_success,
          self.target_num_convergence,
      )
