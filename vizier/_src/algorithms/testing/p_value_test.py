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

"""Tests for p-value."""

import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.testing import p_value

from absl.testing import absltest
from absl.testing import parameterized


class PValueTest(parameterized.TestCase):

  @parameterized.parameters((0.01, 5), (0.05, 7))
  def test_continuous_p_value(self, observed_squared_dist, n_features):
    """Test that the conitnuous p-value formula agrees with empirical results.
    """
    low = 0
    high = 1
    evaluations = 30
    repeats = 600_000

    def empirical_continuous_p_value(observed_squared_dist: float) -> float:
      rng = np.random.default_rng(0)
      sample_within_count = 0
      center = rng.uniform(0.4, 0.6, size=(1, n_features))
      random_samples = rng.uniform(
          low, high, size=(repeats, evaluations, n_features))
      squared_dists = np.sum(np.square(random_samples - center), axis=2)
      min_squared_dist = np.min(squared_dists, axis=1)
      sample_within_count = np.sum(min_squared_dist <= observed_squared_dist)
      empirical_p_value = float(sample_within_count / repeats)
      return empirical_p_value

    # Compute the p-value from the formula, by simulating results with matching
    # 'observed_squared_distance'.
    opt_features = np.zeros(n_features)
    best_features = opt_features + np.sqrt(observed_squared_dist / n_features)
    continuous_p_value = p_value.compute_array_continuous_p_value(
        n_features, evaluations, best_features, opt_features)
    # Check the theortical and empirical results agrees on at least 3 places.
    empirical_p_value = empirical_continuous_p_value(observed_squared_dist)
    self.assertLessEqual(abs(continuous_p_value - empirical_p_value), 1e-3)

  @parameterized.parameters(0, 1, 2)
  def test_categorical_p_value(self, wrongs):
    """Test that the categorical p-value formula agrees with empirical results.
    """
    n_features = 8
    dim = 4
    evaluations = 20
    repeats = 300_000

    def empirical_categorical_p_value(wrongs: int) -> float:
      rng = np.random.default_rng(0)
      # Set the optimum features
      optimum_feautres = np.array([[1] + [0] * (dim - 1)] *
                                  n_features).reshape(-1)
      # Randomize categorical samples
      random_samples = np.eye(dim)[rng.choice(
          dim, size=(repeats, evaluations,
                     n_features))].reshape(repeats, evaluations,
                                           dim * n_features)
      # Evaluate peroformance
      abs_dists = np.sum(np.abs(random_samples - optimum_feautres), axis=2)
      min_wrongs = np.min(abs_dists, axis=1) / 2
      sample_within_count = np.sum(min_wrongs <= wrongs)
      empirical_p_value = float(sample_within_count / repeats)
      return empirical_p_value

    # Compute the p-value from the formula, by simulating results with matching
    # 'wrong_count' features.
    optimum_feautres = np.array([[1] + [0] * (dim)] * n_features).reshape(-1)
    # Create 'best_features' with 'wrong' incorrect parameters.
    best_features = np.array([[1] + [0] * (dim)] * (n_features - wrongs) +
                             [[0, 1] + [0] * (dim - 1)] * (wrongs)).reshape(-1)
    categorical_p_value = p_value.compute_array_categorical_p_value(
        n_features, dim, best_features, optimum_feautres, evaluations)
    # Check the theortical and empirical results agrees on at least 3 places.
    empirical_p_value = empirical_categorical_p_value(wrongs)
    self.assertLessEqual(abs(categorical_p_value - empirical_p_value), 1e-3)

  def test_trial_p_value(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    problem.search_space.root.add_categorical_param('c1', ['a', 'b', 'c'])
    problem.search_space.root.add_categorical_param('c2', ['x', 'y', 'z'])
    optimum_trial = vz.Trial({'f1': 0.05, 'f2': 0.05, 'c1': 'a', 'c2': 'x'})
    best_trial = vz.Trial({'f1': 0.0, 'f2': 0.0, 'c1': 'b', 'c2': 'x'})
    p_value_continuous, p_value_categorical = p_value.compute_trial_p_values(
        problem.search_space, 10, best_trial, optimum_trial)
    self.assertAlmostEqual(p_value_continuous, 0.14642887479936317)
    self.assertAlmostEqual(p_value_categorical, 0.9996992713401783)

  def test_scaled_trial_p_value(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', -5.0, 5.0)
    problem.search_space.root.add_float_param('f2', -5.0, 5.0)
    problem.search_space.root.add_categorical_param('c1', ['a', 'b', 'c'])
    problem.search_space.root.add_categorical_param('c2', ['x', 'y', 'z'])
    optimum_trial = vz.Trial({'f1': 0.5, 'f2': 0.5, 'c1': 'a', 'c2': 'x'})
    best_trial = vz.Trial({'f1': 0.0, 'f2': 0.0, 'c1': 'b', 'c2': 'x'})
    p_value_continuous, p_value_categorical = p_value.compute_trial_p_values(
        problem.search_space, 10, best_trial, optimum_trial)
    self.assertAlmostEqual(p_value_continuous, 0.14642887479936317)
    self.assertAlmostEqual(p_value_categorical, 0.9996992713401783)


if __name__ == '__main__':
  absltest.main()
