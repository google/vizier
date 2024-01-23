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

"""Tests for scheduled_gp_bandit."""

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import scheduled_gp_bandit
from vizier._src.algorithms.testing import test_runners
from vizier.testing import test_studies
from absl.testing import absltest


class ScheduledGpBanditTest(absltest.TestCase):

  def test_schedule_designer(self):
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.append(
        vz.MetricInformation(
            name="metric", goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    designer = scheduled_gp_bandit.VizierScheduledGPBandit(
        problem,
        max_num_trials=5,
        init_ucb_coef=4.0,
        final_ucb_coef=1.0,
        decay_rate=1.2,
    )
    self.assertEqual(designer._compute_ucb_coefficient(), 4.0)
    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=5,
            batch_size=1,
            verbose=1,
            validate_parameters=True,
            seed=1,
        ).run_designer(designer),
        5,
    )
    self.assertEqual(designer._compute_ucb_coefficient(), 1.0)


if __name__ == "__main__":
  absltest.main()
