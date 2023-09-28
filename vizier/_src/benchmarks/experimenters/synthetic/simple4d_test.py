# Copyright 2023 Google LLC.
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

from absl.testing import parameterized
from vizier._src.algorithms.designers import grid
from vizier._src.benchmarks.experimenters.synthetic import simple4d
from vizier._src.benchmarks.runners import benchmark_runner
from vizier._src.benchmarks.runners import benchmark_state
from absl.testing import absltest


class Simple4DTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(best_category='corner'),
      dict(best_category='center'),
      dict(best_category='mixed'),
  )
  def test_sweep(self, best_category: simple4d.Simple4DCategory) -> None:
    experimenter = simple4d.Simple4D(best_category)
    runner = benchmark_runner.BenchmarkRunner(
        benchmark_subroutines=[
            benchmark_runner.GenerateSuggestions(),
            benchmark_runner.EvaluateActiveTrials(),
        ],
        num_repeats=300,
    )
    state = benchmark_state.BenchmarkState(
        experimenter,
        benchmark_state.PolicySuggester.from_designer_factory(
            experimenter.problem_statement(),
            grid.GridSearchDesigner.from_problem,
        ),
    )
    runner.run(state)


if __name__ == '__main__':
  absltest.main()
