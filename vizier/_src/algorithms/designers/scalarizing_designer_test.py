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

from jax import numpy as jnp
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import scalarization
from vizier._src.algorithms.designers import scalarizing_designer
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.testing import test_runners
from vizier.testing import test_studies

from absl.testing import absltest


class ScalarizingDesignerTest(absltest.TestCase):

  def test_scalarizing_eagle(self):
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.extend([
        vz.MetricInformation(
            name='metric1', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        ),
        vz.MetricInformation(
            name='metric2', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        ),
    ])

    def eagle_designer_factory(ps, seed):
      return eagle_strategy.EagleStrategyDesigner(
          problem_statement=ps, seed=seed
      )

    scalarized_designer = scalarizing_designer.ScalarizingDesigner(
        problem,
        eagle_designer_factory,
        scalarizer=scalarization.HyperVolumeScalarization(
            weights=jnp.ones(len(problem.metric_information))
        ),
    )
    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=3,
            batch_size=5,
            verbose=1,
            validate_parameters=True,
        ).run_designer(scalarized_designer),
        15,
    )

  def test_ensemble_scalarizing_eagle(self):
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.extend([
        vz.MetricInformation(
            name='metric1', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        ),
        vz.MetricInformation(
            name='metric2', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        ),
    ])

    def eagle_designer_factory(ps, seed):
      return eagle_strategy.EagleStrategyDesigner(
          problem_statement=ps, seed=seed
      )

    scalarized_designer = (
        scalarizing_designer.create_gaussian_scalarizing_designer(
            problem,
            eagle_designer_factory,
            scalarization.HyperVolumeScalarization,
            num_ensemble=10,
        )
    )
    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=3,
            batch_size=5,
            verbose=1,
            validate_parameters=True,
        ).run_designer(scalarized_designer),
        15,
    )

  def test_missing_metrics(self):
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.extend([
        vz.MetricInformation(
            name='metric1', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        ),
        vz.MetricInformation(
            name='metric2', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        ),
    ])

    def eagle_designer_factory(ps, seed):
      return eagle_strategy.EagleStrategyDesigner(
          problem_statement=ps, seed=seed
      )

    scalarized_designer = scalarizing_designer.ScalarizingDesigner(
        problem,
        eagle_designer_factory,
        scalarizer=scalarization.HyperVolumeScalarization(
            weights=jnp.ones(len(problem.metric_information))
        ),
    )

    suggestions = scalarized_designer.suggest(1)
    trial = list(suggestions)[0].to_trial(1)
    trial.complete(vz.Measurement(metrics={'metric1': 0.4}))
    scalarized_designer.update(
        vza.CompletedTrials(trials=[trial]),
        all_active=vza.ActiveTrials(trials=[]),
    )
    self.assertTrue(
        jnp.isnan(trial.final_measurement_or_die.metrics['scalarized'].value)
    )


if __name__ == '__main__':
  absltest.main()
