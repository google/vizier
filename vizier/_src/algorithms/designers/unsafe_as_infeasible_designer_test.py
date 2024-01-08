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

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import scalarization
from vizier._src.algorithms.designers import scalarizing_designer
from vizier._src.algorithms.designers import unsafe_as_infeasible_designer
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.testing import test_runners
from vizier.testing import test_studies
from absl.testing import absltest


class UnsafeAsInfeasibleDesignerTest(absltest.TestCase):

  def test_unsafe_eagle(self):
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.extend([
        vz.MetricInformation(
            name='metric1', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        ),
        vz.MetricInformation(
            name='metric_safe',
            goal=vz.ObjectiveMetricGoal.MAXIMIZE,
            safety_threshold=0.2,
        ),
    ])

    def eagle_designer_factory(ps, seed):
      return eagle_strategy.EagleStrategyDesigner(
          problem_statement=ps, seed=seed
      )

    safe_eagle = unsafe_as_infeasible_designer.UnsafeAsInfeasibleDesigner(
        problem, eagle_designer_factory
    )

    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=6,
            batch_size=5,
            verbose=1,
            validate_parameters=True,
        ).run_designer(safe_eagle),
        30,
    )

  def test_unsafe_multi_objective_eagle(self):
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.extend([
        vz.MetricInformation(
            name='metric1', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        ),
        vz.MetricInformation(
            name='metric2', goal=vz.ObjectiveMetricGoal.MINIMIZE
        ),
        vz.MetricInformation(
            name='metric_safe',
            goal=vz.ObjectiveMetricGoal.MAXIMIZE,
            safety_threshold=0.2,
        ),
    ])

    def eagle_designer_factory(ps, seed):
      return eagle_strategy.EagleStrategyDesigner(
          problem_statement=ps, seed=seed
      )

    def scalarized_eagle_factory(ps, seed):
      return scalarizing_designer.create_gaussian_scalarizing_designer(
          ps,
          eagle_designer_factory,
          scalarization.HyperVolumeScalarization,
          num_ensemble=3,
          seed=seed,
      )

    safe_eagle = unsafe_as_infeasible_designer.UnsafeAsInfeasibleDesigner(
        problem, scalarized_eagle_factory
    )

    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=6,
            batch_size=5,
            verbose=1,
            validate_parameters=True,
        ).run_designer(safe_eagle),
        30,
    )


if __name__ == '__main__':
  absltest.main()
