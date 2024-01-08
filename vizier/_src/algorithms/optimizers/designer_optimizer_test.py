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

"""Tests for designer_optimizer."""

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.evolution import nsga2
from vizier._src.algorithms.optimizers import designer_optimizer
from vizier._src.algorithms.testing import optimizer_test_utils

from absl.testing import absltest


class DesignerOptimizerTest(absltest.TestCase):

  def test_smoke(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('a', 0, 1)

    designer_factory = quasi_random.QuasiRandomDesigner.from_problem
    optimizer_test_utils.assert_passes_on_random_single_metric_function(
        self,
        problem.search_space,
        designer_optimizer.DesignerAsOptimizer(designer_factory),
        np_random_seed=1)

  def test_bi_objective(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('a', 0, 1)

    designer_factory = nsga2.NSGA2Designer
    optimizer_test_utils.assert_passes_on_random_multi_metric_function(
        self,
        problem.search_space,
        designer_optimizer.DesignerAsOptimizer(designer_factory),
        np_random_seed=1,
    )


if __name__ == '__main__':
  absltest.main()
