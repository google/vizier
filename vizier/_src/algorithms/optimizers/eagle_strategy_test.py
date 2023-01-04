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

"""Tests for eagle strategy."""

from absl.testing import parameterized
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import eagle_param_handler
from vizier._src.algorithms.optimizers import eagle_strategy
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier.pyvizier import converters

from absl.testing import absltest


class VectorizedEagleStrategyContinuousTest(parameterized.TestCase):

  def setUp(self):
    super(VectorizedEagleStrategyContinuousTest, self).setUp()
    self.config = eagle_strategy.EagleStrategyConfig(
        visibility=1, gravity=1, pool_size=4
    )
    problem = vz.ProblemStatement()
    root = problem.search_space.select_root()
    root.add_float_param('x1', 0.0, 1.0)
    root.add_float_param('x2', 0.0, 1.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    self.eagle = eagle_strategy.VectorizedEagleStrategy(
        converter=converter, config=self.config, batch_size=2, seed=1
    )
    self.eagle._iterations = 2

  def test_compute_features_diffs_and_dists(self):
    self.eagle._features = np.array([[1, 2], [3, 4], [7, 7], [8, 8]])
    features_diffs, dists = self.eagle._compute_features_diffs_and_dists()
    expected_features_diffs = np.array(
        [[[0, 0], [2, 2], [6, 5], [7, 6]], [[-2, -2], [0, 0], [4, 3], [5, 4]]]
    )
    np.testing.assert_array_equal(
        features_diffs,
        expected_features_diffs,
        err_msg='feature differences mismatch',
    )

    expected_dists = np.array([[0, 8, 61, 85], [8, 0, 25, 41]])
    np.testing.assert_array_almost_equal(
        dists, expected_dists, err_msg='feature distance mismatch'
    )

  def test_compute_scaled_directions(self):
    self.eagle._rewards = np.array([2, 3, 4, 1])
    g = self.config.gravity
    ng = -self.config.negative_gravity
    expected_scaled_directions = np.array([
        [g, g, g, ng],
        [ng, g, g, ng],
    ])
    scaled_directions = self.eagle._compute_scaled_directions()
    np.testing.assert_array_equal(scaled_directions, expected_scaled_directions)

  def test_compute_scaled_directions_with_removed_flies(self):
    self.eagle._rewards = np.array([-np.inf, 3, -np.inf, 1])
    g = self.config.gravity
    ng = -self.config.negative_gravity
    # Note that -np.inf - (-np.inf) is not >=0.
    expected_scaled_directions = np.array([
        [ng, g, ng, g],
        [ng, g, ng, ng],
    ])
    scaled_directions = self.eagle._compute_scaled_directions()
    np.testing.assert_array_equal(scaled_directions, expected_scaled_directions)

  @parameterized.parameters('random', 'mean')
  def test_compute_features_changes(self, norm_type):
    self.eagle._rewards = np.array([-np.inf, 3, -np.inf, 1])
    features_diffs = np.array(
        [[[0, 0], [2, 2], [6, 5], [7, 6]], [[-2, -2], [0, 0], [4, 3], [5, 4]]]
    )
    dists = np.array([[0, 8, 61, 85], [8, 0, 25, 41]])
    scaled_directions = np.array([
        [-1, 1, -1, 1],
        [-1, 1, -1, -1],
    ])
    if norm_type == 'mean':
      features_changes = self.eagle._compute_features_changes(
          features_diffs, dists, scaled_directions
      )
      # scaled_pulls array:
      # [[-1.00000000e+000  4.24835426e-018 -3.46883002e-133  2.65977679e-185]
      # [-4.24835426e-018  1.00000000e+000 -5.16642063e-055 -9.32462145e-090]
      c0 = (
          np.array([2, 2]) * 4.24835426e-018 / 2.0
          + np.array([7, 6]) * 2.65977679e-185 / 2.0
      )
      c1 = np.array([5, 4]) * (-9.32462145e-090) / 1.0
      expected_features_changes = np.vstack([c0, c1])
      self.assertEqual(features_changes.shape, (2, 2))
      np.testing.assert_array_almost_equal(
          expected_features_changes, features_changes
      )

    if norm_type == 'random':
      for _ in range(1000):
        # Repeat the check multiple time to increase confidence.
        features_changes = self.eagle._compute_features_changes(
            features_diffs, dists, scaled_directions
        )
        # scaled_pulls:
        # [[-1.00000000e+000  4.24835426e-018 -3.46883002e-133  2.65977679e-185]
        # [-4.24835426e-018  1.00000000e+000 -5.16642063e-055 -9.32462145e-090]]
        c0 = (
            np.array([2, 2]) * 4.24835426e-018 / 2.0
            + np.array([7, 6]) * 2.65977679e-185 / 2.0
        )
        c1 = np.array([5, 4]) * (-9.32462145e-090) / 1.0
        self.assertEqual(features_changes.shape, (2, 2))

        low_bound_00 = 7 * 2.65977679e-185
        high_bound_00 = 2 * 4.24835426e-018
        low_bound_01 = 6 * 2.65977679e-185
        high_bound_01 = 2 * 4.24835426e-018

        self.assertBetween(features_changes[0][0], low_bound_00, high_bound_00)
        self.assertBetween(features_changes[0][1], low_bound_01, high_bound_01)
        np.testing.assert_array_almost_equal(features_changes[1], c1)

        expected_features_changes = np.vstack([c0, c1])

        np.testing.assert_array_almost_equal(
            expected_features_changes, features_changes
        )

  def test_create_features(self):
    self.assertEqual(self.eagle._create_features().shape, (2, 2))

  def test_create_perturbations(self):
    perturbations = self.eagle._create_perturbations()
    self.assertEqual(perturbations.shape, (2, 2))

  def test_update_pool_features_and_rewards(self):
    self.eagle._features = np.array(
        [[1, 2], [3, 4], [7, 7], [8, 8]], dtype=np.float64
    )
    self.eagle._rewards = np.array([2, 3, 4, 1], dtype=np.float64)
    self.eagle._perturbations = np.array([1, 1, 1, 1], dtype=np.float64)

    self.eagle._last_suggested_features = np.array(
        [[9, 9], [10, 10]], dtype=np.float64
    )
    batch_rewards = np.array([5, 0.5], dtype=np.float64)

    self.eagle._update_pool_features_and_rewards(batch_rewards)
    np.testing.assert_array_equal(
        self.eagle._features,
        np.array([[9, 9], [3, 4], [7, 7], [8, 8]], dtype=np.float64),
        err_msg='Features are not equal.',
    )

    np.testing.assert_array_equal(
        self.eagle._rewards,
        np.array([5, 3, 4, 1], dtype=np.float64),
        err_msg='rewards are not equal.',
    )

    pc = self.config.penalize_factor
    np.testing.assert_array_equal(
        self.eagle._perturbations,
        np.array([1, pc, 1, 1], dtype=np.float64),
        err_msg='Perturbations are not equal.',
    )

  def test_update_best_reward(self):
    # Test replacing the best reward.
    self.eagle._rewards = np.array([2, 3, 4, 1], dtype=np.float64)
    batch_rewards = np.array([5, 0.5], dtype=np.float64)
    self.eagle._update_best_reward(batch_rewards)
    self.assertEqual(self.eagle._best_reward, 5.0)
    # Test not replacing the best reward.
    batch_rewards = np.array([2, 4], dtype=np.float64)
    self.eagle._update_best_reward(batch_rewards)
    self.assertEqual(self.eagle._best_reward, 5.0)

  def test_trim_pool(self):
    pc = self.config.perturbation
    self.eagle._features = np.array(
        [[1, 2], [3, 4], [7, 7], [8, 8]], dtype=np.float64
    )
    self.eagle._rewards = np.array([2, 3, 4, 1], dtype=np.float64)
    self.eagle._perturbations = np.array([pc, 0, 0, pc], dtype=np.float64)
    self.eagle._best_results = [
        vb.VectorizedStrategyResult(reward=4.0, features=np.array([1.0, 2.0]))
    ]
    self.eagle._trim_pool()

    np.testing.assert_array_almost_equal(
        self.eagle._features[[0, 2, 3], :],
        np.array([[1, 2], [7, 7], [8, 8]], dtype=np.float64),
        err_msg='Features are not equal.',
    )
    self.assertTrue(
        all(np.not_equal(self.eagle._features[1, :], np.array([3, 4]))),
        msg='Features are not equal.',
    )

    np.testing.assert_array_equal(
        self.eagle._rewards,
        np.array([2, -np.inf, 4, 1], dtype=np.float64),
        err_msg='rewards are not equal.',
    )
    # The best firefly is never removed.
    np.testing.assert_array_equal(
        self.eagle._perturbations,
        np.array([pc, pc, 0, pc], dtype=np.float64),
        err_msg='Perturbations are not equal.',
    )

  def test_create_strategy_from_factory(self):
    problem = vz.ProblemStatement()
    root = problem.search_space.select_root()
    root.add_float_param('x1', 0.0, 1.0)
    root.add_float_param('x2', 0.0, 1.0)
    root.add_float_param('x3', 0.0, 1.0)
    eagle_factory = eagle_strategy.VectorizedEagleStrategyFactory()
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    eagle = eagle_factory(converter)
    self.assertEqual(eagle._n_features, 3)

  def test_optimize_with_eagle(self):
    problem = vz.ProblemStatement()
    root = problem.search_space.select_root()
    root.add_float_param('x1', 0.0, 1.0)
    root.add_float_param('x2', 0.0, 1.0)
    root.add_float_param('x3', 0.0, 1.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    eagle_factory = eagle_strategy.VectorizedEagleStrategyFactory()
    optimizer = vb.VectorizedOptimizer(strategy_factory=eagle_factory)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    optimizer.optimize(converter, score_fn=lambda x: -np.sum(x, 1), count=1)


class EagleParamHandlerTest(parameterized.TestCase):

  def setUp(self):
    super(EagleParamHandlerTest, self).setUp()
    problem = vz.ProblemStatement()
    root = problem.search_space.select_root()
    root.add_categorical_param('c1', ['a', 'b'])
    root.add_float_param('f1', 0.0, 5.0)
    root.add_categorical_param('c2', ['a', 'b', 'c'])
    root.add_discrete_param('d1', [2.0, 3.0, 5.0, 11.0])
    converter = converters.TrialToArrayConverter.from_study_config(
        problem, max_discrete_indices=0, pad_oovs=True
    )
    self.config = eagle_strategy.EagleStrategyConfig()
    self.param_handler = eagle_param_handler.EagleParamHandler(
        converter=converter,
        rng=np.random.default_rng(1),
        categorical_perturbation_factor=self.config.categorical_perturbation_factor,
        pure_categorical_perturbation_factor=self.config.pure_categorical_perturbation_factor,
    )

  def test_init(self):
    self.assertEqual(self.param_handler.n_features, 9)
    self.assertFalse(self.param_handler.all_features_categorical)
    self.assertTrue(self.param_handler.has_categorical)
    self.assertEqual(self.param_handler.n_categorical, 2)

  def test_categorical_params_mask(self):
    expected_categorical_params_mask = np.array(
        [[1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1, 0]]
    )
    np.testing.assert_array_equal(
        self.param_handler._categorical_params_mask,
        expected_categorical_params_mask,
    )

  def test_categorical_mask(self):
    expected_categorical_mask = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(
        self.param_handler._categorical_mask, expected_categorical_mask
    )

  def test_tiebreak_mask(self):
    eps = self.param_handler._epsilon
    expected_tiebreak_mask = np.array(
        [-eps * (i + 1) for i in range(9)], dtype=float
    )
    np.testing.assert_array_equal(
        self.param_handler._tiebreak_array, expected_tiebreak_mask
    )

  def test_categorical_oov_mask(self):
    expected_oov_mask = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=float)
    np.testing.assert_array_equal(
        self.param_handler._oov_mask, expected_oov_mask
    )

  def test_perturbation_factors(self):
    cp = self.config.categorical_perturbation_factor
    expected_perturbation_factors = np.array(
        [cp, cp, cp, 1, cp, cp, cp, cp, 1], dtype=float
    )
    np.testing.assert_array_equal(
        self.param_handler.perturbation_factors, expected_perturbation_factors
    )

  def test_sample_categorical_features(self):
    # features shouldn't have values in oov_mask, and have the structure of:
    # [c1,c1,c1,f1,c2,c2,c2,c2,d1]
    features = np.array([
        [2.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.1, 0.0, 9.0],
        [3.0, 0.0, 0.0, 3.5, 5.0, 0.0, 0.0, 0.0, 8.0],
    ])
    expected_sampled_features = np.array([
        [1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 1.0, 0.0, 9.0],
        [1.0, 0.0, 0.0, 3.5, 1.0, 0.0, 0.0, 0.0, 8.0],
    ])
    sampled_features = self.param_handler.sample_categorical(features)
    np.testing.assert_array_equal(sampled_features, expected_sampled_features)

  def test_prior_trials(self):
    config = eagle_strategy.EagleStrategyConfig(
        visibility=1, gravity=1, pool_size=4, prior_trials_pool_pct=1.0
    )
    problem = vz.ProblemStatement()
    root = problem.search_space.select_root()
    root.add_float_param('x1', 0.0, 1.0)
    root.add_float_param('x2', 0.0, 1.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)

    prior_features = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    prior_rewards = np.array([1, 2, 3, 4])
    eagle = eagle_strategy.VectorizedEagleStrategy(
        converter=converter,
        config=config,
        batch_size=2,
        seed=1,
        prior_features=prior_features,
        prior_rewards=prior_rewards,
    )
    np.testing.assert_equal(eagle.prior_features, prior_features)

  @parameterized.parameters(2, 10)
  def test_prior_trials_with_too_few_or_many_trials(self, n_prior_trials):
    # Test proper populating of the pool in case the number of prior trials is
    # greater or lesser than the pool size.
    config = eagle_strategy.EagleStrategyConfig(
        visibility=1, gravity=1, pool_size=4, prior_trials_pool_pct=1.0
    )
    problem = vz.ProblemStatement()
    root = problem.search_space.select_root()
    root.add_float_param('x1', 0.0, 1.0)
    root.add_float_param('x2', 0.0, 1.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)

    prior_features = np.random.randn(n_prior_trials, 2)
    prior_rewards = np.random.randn(n_prior_trials)

    eagle = eagle_strategy.VectorizedEagleStrategy(
        converter=converter,
        config=config,
        batch_size=2,
        seed=1,
        prior_features=prior_features,
        prior_rewards=prior_rewards,
    )
    self.assertEqual(eagle.prior_features.shape, (n_prior_trials, 2))


if __name__ == '__main__':
  absltest.main()
