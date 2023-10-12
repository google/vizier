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

"""Simple4d convergence tests for Eagle designer."""

from absl.testing import parameterized
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.testing import simple4d_runner
from vizier._src.benchmarks.experimenters.synthetic import simple4d
from absl.testing import absltest


class Simple4DEagleDesignerTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(best_category='corner'),
      dict(best_category='center'),
      dict(best_category='mixed'),
  )
  def test_convergence(self, best_category: simple4d.Simple4DCategory) -> None:
    def _eagle_designer_factory(problem, seed):
      return eagle_strategy.EagleStrategyDesigner(problem, seed=seed)

    simple4d_runner.Simple4DConvergenceTester(
        best_category=best_category,
        designer_factory=_eagle_designer_factory,
        num_trials=5000,
        max_relative_error=0.05,
        num_repeats=20,
        target_num_convergence=10,
    ).assert_convergence()


if __name__ == '__main__':
  absltest.main()
