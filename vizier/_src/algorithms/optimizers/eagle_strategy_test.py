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
import jax
from jax import numpy as jnp
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import eagle_param_handler
from vizier._src.algorithms.optimizers import eagle_strategy
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier.pyvizier import converters

from absl.testing import absltest


def _create_features_simple(
    features, rewards, features_batch, rewards_batch, config, n_features
):
  """A version of `_create_features` that materializes large intermediates."""
  features_diffs = features - features_batch[:, jnp.newaxis, :]
  dists = jnp.sum(jnp.square(features_diffs), axis=-1)
  directions = rewards - rewards_batch[:, jnp.newaxis]
  scaled_directions = jnp.where(
      directions >= 0.0, config.gravity, -config.negative_gravity
  )

  # Normalize the distance by the number of features.
  force = jnp.exp(-config.visibility * dists / n_features * 10.0)
  scaled_force = scaled_directions * force
  # Handle removed fireflies without updated rewards.
  finite_ind = jnp.isfinite(rewards).astype(scaled_force.dtype)

  # Ignore fireflies that were removed from the pool.
  scaled_force = scaled_force * finite_ind

  # Separate forces to pull and push so to normalize them separately.
  scaled_pulls = jnp.maximum(scaled_force, 0.0)
  scaled_push = jnp.minimum(scaled_force, 0.0)
  features_changes = jnp.sum(
      features_diffs * (scaled_pulls + scaled_push)[..., jnp.newaxis], axis=1
  )
  return features_batch + features_changes


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
        converter=converter, config=self.config, batch_size=2
    )
    self.eagle._iterations = 2

  def test_create_features(self):
    features = jnp.array([[1, 2], [3, 4], [7, 7], [8, 8]])
    rewards = jnp.array([2, 3, 4, 1])
    seed = jax.random.PRNGKey(0)
    features_batch = features[: self.eagle.batch_size]
    rewards_batch = rewards[: self.eagle.batch_size]
    self.assertEqual(
        self.eagle._create_features(
            features,
            rewards,
            features_batch,
            rewards_batch,
            seed=seed,
        ).shape,
        (2, 2),
    )

    expected = _create_features_simple(
        features,
        rewards,
        features_batch,
        rewards_batch,
        self.config.replace(
            mutate_normalization_type=(
                eagle_strategy.MutateNormalizationType.UNNORMALIZED
            )
        ),
        self.eagle._n_features,
    )
    actual = self.eagle._create_features(
        features,
        rewards,
        features_batch,
        rewards_batch,
        seed=seed,
    )
    np.testing.assert_array_equal(expected, actual)

  def test_create_random_perturbations(self):
    seed = jax.random.PRNGKey(0)
    perturbations_batch = jnp.ones(self.eagle.batch_size)
    perturbations = self.eagle._create_random_perturbations(
        perturbations_batch, seed=seed
    )
    self.assertEqual(perturbations.shape, (2, 2))

  def test_update_pool_features_and_rewards(self):
    features = jnp.array([[1, 2], [3, 4], [7, 7], [8, 8]], dtype=jnp.float64)
    rewards = jnp.array([2, 3, 4, 1], dtype=jnp.float64)
    perturbations = jnp.array([1, 1, 1, 1], dtype=jnp.float64)

    batch_features = jnp.array([[9, 9], [10, 10]], dtype=jnp.float64)
    batch_rewards = jnp.array([5, 0.5], dtype=jnp.float64)

    new_features, new_rewards, new_perturbations = (
        self.eagle._update_pool_features_and_rewards(
            batch_features,
            batch_rewards,
            features[: self.eagle.batch_size],
            rewards[: self.eagle.batch_size],
            perturbations[: self.eagle.batch_size],
        )
    )
    np.testing.assert_array_equal(
        new_features,
        np.array([[9, 9], [3, 4]], dtype=np.float64),
        err_msg='Features are not equal.',
    )

    np.testing.assert_array_equal(
        new_rewards,
        np.array([5, 3], dtype=np.float64),
        err_msg='rewards are not equal.',
    )

    pc = self.config.penalize_factor
    np.testing.assert_array_almost_equal(
        new_perturbations,
        np.array([1, pc], dtype=np.float64),
        err_msg='Perturbations are not equal.',
    )

  def test_update_best_reward(self):
    # Test replacing the best reward.
    features = jnp.array([[1, 2], [3, 4], [7, 7], [8, 8]], dtype=jnp.float64)
    rewards = jnp.array([2, 3, 4, 1], dtype=jnp.float64)
    state = eagle_strategy.VectorizedEagleStrategyState(
        iterations=jnp.array(0),
        features=features,
        rewards=rewards,
        best_reward=jnp.max(rewards),
        perturbations=jnp.ones_like(rewards),
    )
    batch_features = jnp.array([[9, 9], [10, 10]], dtype=jnp.float64)
    batch_rewards = jnp.array([5, 0.5], dtype=jnp.float64)
    seed = jax.random.PRNGKey(0)
    new_state = self.eagle.update(
        state, batch_features, batch_rewards, seed=seed
    )
    self.assertEqual(new_state.best_reward, 5.0)
    # Test not replacing the best reward.
    batch_rewards = jnp.array([2, 4], dtype=jnp.float64)
    new_new_state = self.eagle.update(
        new_state, batch_features, batch_rewards, seed=seed
    )
    self.assertEqual(new_new_state.best_reward, 5.0)

  @parameterized.parameters(
      {'batch_size': 5, 'expected_batch_size': 5, 'max_pool_size': 50},
      {'batch_size': None, 'expected_batch_size': 50, 'max_pool_size': 50},
      {'batch_size': 5, 'expected_batch_size': 5, 'max_pool_size': 10},
      {'batch_size': None, 'expected_batch_size': 10, 'max_pool_size': 10},
  )
  def test_batch_size_and_pool_size(
      self, batch_size, expected_batch_size, max_pool_size
  ):
    problem = vz.ProblemStatement()
    root = problem.search_space.root
    for i in range(100):
      root.add_float_param(f'x{i}', 0.0, 1.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    config = eagle_strategy.EagleStrategyConfig(max_pool_size=max_pool_size)
    eagle = eagle_strategy.VectorizedEagleStrategy(
        converter=converter, config=config, batch_size=batch_size
    )
    self.assertEqual(eagle.pool_size, max_pool_size)
    self.assertEqual(eagle.batch_size, expected_batch_size)

  def test_trim_pool(self):
    pc = self.config.perturbation
    features_batch = jnp.array([[1, 2], [3, 4]], dtype=jnp.float64)
    rewards_batch = jnp.array([2, 3], dtype=jnp.float64)
    perturbations = jnp.array([pc, 0], dtype=jnp.float64)
    seed = jax.random.PRNGKey(0)
    new_features, new_rewards, new_perturbations = self.eagle._trim_pool(
        features_batch,
        rewards_batch,
        perturbations,
        best_reward=jnp.array(4.0),
        seed=seed,
    )

    np.testing.assert_array_almost_equal(
        new_features[0],
        features_batch[0],
        err_msg='Features are not equal.',
    )
    self.assertTrue(
        all(np.not_equal(new_features[1], features_batch[1])),
        msg='Features are not equal.',
    )

    np.testing.assert_array_equal(
        new_rewards,
        np.array([2, -np.inf], dtype=np.float64),
        err_msg='rewards are not equal.',
    )
    # The best firefly is never removed.
    np.testing.assert_array_almost_equal(
        new_perturbations,
        np.array([pc, pc], dtype=np.float64),
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
    optimizer.optimize(converter, score_fn=lambda x: -jnp.sum(x, 1), count=1)


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
    np.testing.assert_array_almost_equal(
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
    features = jnp.array([
        [2.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.1, 0.0, 9.0],
        [3.0, 0.0, 0.0, 3.5, 5.0, 0.0, 0.0, 0.0, 8.0],
    ])
    expected_sampled_features = np.array([
        [1.0, 0.0, 0.0, 1.5, 0.0, 0.0, 1.0, 0.0, 9.0],
        [1.0, 0.0, 0.0, 3.5, 1.0, 0.0, 0.0, 0.0, 8.0],
    ])
    sampled_features = self.param_handler.sample_categorical(
        features, seed=jax.random.PRNGKey(0)
    )
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

    prior_features = jnp.array([[1, -1], [2, 1], [3, 2], [4, 5]])
    prior_rewards = jnp.array([1, 2, 3, 4])
    eagle = eagle_strategy.VectorizedEagleStrategy(
        converter=converter,
        config=config,
        batch_size=2,
    )
    init_state = eagle.init_state(
        jax.random.PRNGKey(0), prior_features, prior_rewards
    )
    np.testing.assert_array_equal(
        init_state.features, jnp.flip(prior_features, axis=0)
    )

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
    )
    init_state = eagle.init_state(
        jax.random.PRNGKey(0), prior_features, prior_rewards
    )
    self.assertEqual(init_state.features.shape, (config.pool_size, 2))


if __name__ == '__main__':
  absltest.main()
