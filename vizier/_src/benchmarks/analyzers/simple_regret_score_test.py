# Copyright 2022 Google LLC.
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

"""Tests for simple_regret_score."""

import numpy as np
from vizier._src.benchmarks.analyzers import simple_regret_score

from absl.testing import absltest


class SimpleRegretScoreTest(absltest.TestCase):

  def test_one_sample_fail(self):
    baseline_simple_regrets = np.ones(500) + 0.1 * np.random.normal(size=(500,))
    candidate_simple_regrets = [0.9]
    p_value = simple_regret_score.t_test_less_mean_score(
        baseline_simple_regrets, candidate_simple_regrets)
    # test that the p-value is high, which means the test would fail.
    self.assertGreater(p_value, 0.95)

  def test_one_sample_pass(self):
    baseline_simple_regrets = np.ones(500) + 0.1 * np.random.normal(size=(500,))
    candidate_simple_regrets = [1.2]
    p_value = simple_regret_score.t_test_less_mean_score(
        baseline_simple_regrets, candidate_simple_regrets)
    # test that the p-value is low, which means the test would fail.
    self.assertLess(p_value, 0.05)

  def test_two_sample_fail(self):
    baseline_simple_regrets = np.ones(500) + 0.1 * np.random.normal(size=(500,))
    candidate_simple_regrets = 0.99 * np.ones(200)
    p_value = simple_regret_score.t_test_less_mean_score(
        baseline_simple_regrets, candidate_simple_regrets)
    # test that the p-value is high, which means the test would fail.
    self.assertGreater(p_value, 0.95)

  def test_two_sample_pass(self):
    baseline_simple_regrets = np.ones(500) + 0.1 * np.random.normal(size=(500,))
    candidate_simple_regrets = 1.2 * np.ones(200)
    p_value = simple_regret_score.t_test_less_mean_score(
        baseline_simple_regrets, candidate_simple_regrets)
    # test that the p-value is high, which means the test would fail.
    self.assertLess(p_value, 0.05)

  def test_score_decreases_with_samples_one_sample(self):
    baseline_simple_regrets = np.ones(500) + 0.1 * np.random.normal(size=(500,))
    candidate_simple_regrets = [1.2]
    p_value500 = simple_regret_score.t_test_less_mean_score(
        baseline_simple_regrets, candidate_simple_regrets)

    baseline_simple_regrets = np.ones(20) + 0.1 * np.random.normal(size=(20,))
    candidate_simple_regrets = [1.2]
    p_value20 = simple_regret_score.t_test_less_mean_score(
        baseline_simple_regrets, candidate_simple_regrets)

    # test that the p-value decreases with the number of samples
    self.assertLess(p_value500, p_value20)

  def test_score_decreases_with_samples_two_sample(self):
    baseline_simple_regrets = np.ones(500) + 0.1 * np.random.normal(size=(500,))
    candidate_simple_regrets = 1.2 * np.ones(200)
    p_value500 = simple_regret_score.t_test_less_mean_score(
        baseline_simple_regrets, candidate_simple_regrets)

    baseline_simple_regrets = np.ones(20) + 0.1 * np.random.normal(size=(20,))
    candidate_simple_regrets = 1.2 * np.ones(200)
    p_value20 = simple_regret_score.t_test_less_mean_score(
        baseline_simple_regrets, candidate_simple_regrets)

    # test that the p-value decreases with the number of samples
    self.assertLess(p_value500, p_value20)


if __name__ == '__main__':
  absltest.main()
