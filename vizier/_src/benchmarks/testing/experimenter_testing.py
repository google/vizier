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

"""Assertions for testing experimenters."""

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.benchmarks.experimenters import experimenter as experimenter_lib
from vizier._src.benchmarks.runners import benchmark_runner
from vizier._src.benchmarks.runners import benchmark_state


def assert_evaluates_random_suggestions(
    test,
    experimenter: experimenter_lib.Experimenter,
) -> None:
  """Asserts that random suggestions from the search space are valid."""
  runner = benchmark_runner.BenchmarkRunner(
      [benchmark_runner.GenerateAndEvaluate(10)], num_repeats=1
  )

  state = benchmark_state.BenchmarkState(
      experimenter=experimenter,
      algorithm=benchmark_state.PolicySuggester.from_designer_factory(
          experimenter.problem_statement(), random.RandomDesigner.from_problem
      ),
  )

  runner.run(state)

  test.assertLen(
      state.algorithm.supporter.GetTrials(
          status_matches=vz.TrialStatus.COMPLETED
      ),
      10,
  )
