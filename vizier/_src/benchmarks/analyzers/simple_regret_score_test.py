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

"""Tests for simple_regret_score."""

import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.analyzers import simple_regret_score

from absl.testing import absltest
from absl.testing import parameterized


class SimpleRegretScoreTest(parameterized.TestCase):

  # @parameterized.product(
  #       create_problem=[
  #           create_continuous_problem,
  #           create_categorical_problem,
  #           create_mix_problem,
  #       ],
  #       n_features=list(range(10, 20)),
  #   )

  @parameterized.parameters(
      {
          'candidate_mean_values': [0.9],
          'goal': vz.ObjectiveMetricGoal.MAXIMIZE,
          'should_pass': False
      },
      {
          'candidate_mean_values': [1.2],
          'goal': vz.ObjectiveMetricGoal.MAXIMIZE,
          'should_pass': True
      },
      {
          'candidate_mean_values': 0.9 * np.ones(200),
          'goal': vz.ObjectiveMetricGoal.MAXIMIZE,
          'should_pass': False
      },
      {
          'candidate_mean_values': 1.2 * np.ones(200),
          'goal': vz.ObjectiveMetricGoal.MAXIMIZE,
          'should_pass': True
      },
      {
          'candidate_mean_values': [0.9],
          'goal': vz.ObjectiveMetricGoal.MINIMIZE,
          'should_pass': True
      },
      {
          'candidate_mean_values': [1.2],
          'goal': vz.ObjectiveMetricGoal.MINIMIZE,
          'should_pass': False
      },
      {
          'candidate_mean_values': 0.9 * np.ones(200),
          'goal': vz.ObjectiveMetricGoal.MINIMIZE,
          'should_pass': True
      },
      {
          'candidate_mean_values': 1.2 * np.ones(200),
          'goal': vz.ObjectiveMetricGoal.MINIMIZE,
          'should_pass': False
      },
  )
  def test_mean_score(self, candidate_mean_values, goal, should_pass):
    baseline_mean_values = np.ones(500) + 0.1 * np.random.normal(size=(500,))
    p_value = simple_regret_score.t_test_mean_score(baseline_mean_values,
                                                    candidate_mean_values, goal)
    if should_pass:
      self.assertLess(p_value, 0.05)
    else:
      self.assertGreater(p_value, 0.95)

  def test_score_decreases_with_samples_two_sample(self):
    goal = vz.ObjectiveMetricGoal.MAXIMIZE
    baseline_simple_regrets = np.ones(500) + 0.1 * np.random.normal(size=(500,))
    candidate_simple_regrets = 1.2 * np.ones(200)
    p_value500 = simple_regret_score.t_test_mean_score(
        baseline_simple_regrets, candidate_simple_regrets, goal)

    baseline_simple_regrets = np.ones(20) + 0.1 * np.random.normal(size=(20,))
    candidate_simple_regrets = 1.2 * np.ones(200)
    p_value20 = simple_regret_score.t_test_mean_score(baseline_simple_regrets,
                                                      candidate_simple_regrets,
                                                      goal)
    # test that the p-value decreases with the number of samples
    self.assertLess(p_value500, p_value20)


if __name__ == '__main__':
  absltest.main()
