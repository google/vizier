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

"""Convergence test for Eagle Strategy."""

from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.designers.eagle_strategy import testing
from vizier._src.algorithms.testing import comparator_runner
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest
from absl.testing import parameterized


class EagleStrategyConvergenceTest(parameterized.TestCase):
  """Convergence test for Eagle Strategy designer.

  Note that all optimization problems are MINIMIZATION.
  """

  @parameterized.parameters(
      testing.create_continuous_exptr(bbob.Gallagher101Me),
      testing.create_continuous_log_scale_exptr(bbob.Rastrigin),
      testing.create_categorical_exptr(),
  )
  def test_convergence(self, exptr):

    def _random_designer_factory(problem, seed):
      return random.RandomDesigner(problem.search_space, seed=seed)

    def _eagle_designer_factory(problem, seed):
      return eagle_strategy.EagleStrategyDesigner(problem, seed=seed)

    random_benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=_random_designer_factory, experimenter=exptr)

    eagle_benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=_eagle_designer_factory,
        experimenter=exptr,
    )
    evaluations = 1000
    # Random designer batch size is large to expedite run time.
    comparator_runner.SimpleRegretComparisonTester(
        baseline_num_trials=2 * evaluations,
        candidate_num_trials=evaluations,
        baseline_suggestion_batch_size=2 * evaluations,
        candidate_suggestion_batch_size=5,
        baseline_num_repeats=5,
        candidate_num_repeats=1,
        alpha=0.05,
        goal=vz.ObjectiveMetricGoal.MINIMIZE,
    ).assert_benchmark_state_better_simple_regret(
        baseline_benchmark_state_factory=random_benchmark_state_factory,
        candidate_benchmark_state_factory=eagle_benchmark_state_factory,
    )


if __name__ == '__main__':
  absltest.main()
