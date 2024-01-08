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

"""Tests for eagle strategy."""

from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import eagle_strategy
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import types
from vizier.pyvizier import converters
from vizier.pyvizier.converters import padding

from absl.testing import absltest

tfd = tfp.distributions


def _create_logits_vector_simple(
    categorical_features,
    categorical_features_batch,
    scale,
    categorical_sizes,
    max_categorical_size,
    config,
):
  n_batch = categorical_features_batch.shape[0]
  n_feat = categorical_features.shape[0]
  logits = np.zeros((n_batch, len(categorical_sizes), max_categorical_size))
  for i, s in enumerate(categorical_sizes):
    one_hot_features = np.zeros([n_feat, s])
    one_hot_features[np.arange(n_feat), categorical_features[:, i]] = 1

    one_hot_batch = np.zeros([n_batch, s])
    one_hot_batch[np.arange(n_batch), categorical_features_batch[:, i]] = 1

    features_change = np.matmul(
        scale, one_hot_features
    ) - one_hot_batch * np.sum(scale, axis=-1, keepdims=True)

    diff_category_logit = np.log(
        (1.0 - config.prob_same_category_without_perturbation) / (s - 1)
    )
    logits_i = np.zeros((n_batch, max_categorical_size)) + diff_category_logit
    logits_i[:, s:] = -np.inf
    logits_i[np.arange(n_batch), categorical_features_batch[:, i]] = np.log(
        config.prob_same_category_without_perturbation
    )
    logits_i[:, :s] = logits_i[:, :s] + features_change
    logits[:, i, :] = logits_i
  return logits


def _create_features_simple(
    features,
    rewards,
    features_batch,
    rewards_batch,
    config,
    n_features,
    categorical_sizes,
    max_categorical_size,
    seed,
):
  """A version of `_create_features` that materializes large intermediates."""
  # Only works with no parallel batch dimension.
  continuous_features_diffs = (
      features.continuous - features_batch.continuous[:, jnp.newaxis, :]
  )
  categorical_features_diffs = (
      features.categorical != features_batch.categorical[:, jnp.newaxis, :]
  )
  features_diffs = vb.VectorizedOptimizerInput(
      continuous=continuous_features_diffs,
      categorical=categorical_features_diffs,
  )
  dists = jax.tree_util.tree_map(
      lambda x: jnp.sum(jnp.square(x), axis=-1), features_diffs
  )
  directions = rewards - rewards_batch[:, jnp.newaxis]
  scaled_directions = jnp.where(
      directions >= 0.0, config.gravity, -config.negative_gravity
  )

  # Handle removed fireflies without updated rewards.
  finite_ind = jnp.isfinite(rewards).astype(directions.dtype)

  # Ignore fireflies that were removed from the pool.
  scale = jax.tree_util.tree_map(
      lambda x: finite_ind  # pylint: disable=g-long-lambda
      * scaled_directions
      * jnp.exp(-config.visibility * x / n_features * 10.0),
      dists,
  )

  # Separate forces to pull and push so to normalize them separately.
  new_continuous_features = features_batch.continuous + jnp.sum(
      features_diffs.continuous * scale.continuous[..., jnp.newaxis], axis=1
  )
  categorical_features_logits = _create_logits_vector_simple(
      features.categorical,
      features_batch.categorical,
      scale.categorical,
      categorical_sizes,
      max_categorical_size,
      config,
  )
  new_categorical_features = tfd.Categorical(
      logits=categorical_features_logits
  ).sample(seed=seed)

  return vb.VectorizedOptimizerInput(
      new_continuous_features, new_categorical_features
  )


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
    root.add_categorical_param('c1', ['a', 'b'])
    root.add_categorical_param('c2', ['a', 'b', 'c'])
    self.converter = converters.TrialToModelInputConverter.from_problem(problem)
    self.eagle = eagle_strategy.VectorizedEagleStrategyFactory(
        eagle_config=self.config
    )(converter=self.converter, suggestion_batch_size=2)

  def test_create_features_and_logits(self):
    features_continuous = jnp.array(
        [[[1.0, 2.0]], [[3.0, 4.0]], [[7.0, 7.0]], [[8.0, 8.0]]]
    )
    features_categorical = jnp.array([[1, 2], [0, 0], [0, 1], [1, 1]])[
        :, jnp.newaxis, :
    ]
    rewards = jnp.array([2, 3, 4, 1])
    seed = jax.random.PRNGKey(0)
    features_continuous_batch = features_continuous[: self.eagle.batch_size]
    features_categorical_batch = features_categorical[: self.eagle.batch_size]
    features = vb.VectorizedOptimizerInput(
        continuous=features_continuous, categorical=features_categorical
    )
    features_batch = vb.VectorizedOptimizerInput(
        continuous=features_continuous_batch,
        categorical=features_categorical_batch,
    )
    rewards_batch = rewards[: self.eagle.batch_size]
    created_features = self.eagle._create_features(
        features,
        rewards,
        features_batch,
        rewards_batch,
        vb.VectorizedOptimizerInput(
            jnp.zeros_like(features_continuous_batch),
            jnp.zeros(features_categorical_batch.shape + (3,)),
        ),
        seed=seed,
    )
    self.assertEqual(created_features.continuous.shape, (2, 1, 2))
    self.assertEqual(created_features.categorical.shape, (2, 1, 2))

    features_2d = vb.VectorizedOptimizerInput(
        features.continuous[:, 0, :], features.categorical[:, 0, :]
    )
    features_batch_2d = vb.VectorizedOptimizerInput(
        features_batch.continuous[:, 0, :], features_batch.categorical[:, 0, :]
    )
    expected = _create_features_simple(
        features_2d,
        rewards,
        features_batch_2d,
        rewards_batch,
        self.config.replace(
            mutate_normalization_type=(
                eagle_strategy.MutateNormalizationType.UNNORMALIZED
            )
        ),
        (
            self.eagle.n_feature_dimensions.continuous
            + self.eagle.n_feature_dimensions.categorical
        ),
        self.eagle.categorical_sizes,
        self.eagle.max_categorical_size,
        seed,
    )
    actual = self.eagle._create_features(
        features,
        rewards,
        features_batch,
        rewards_batch,
        vb.VectorizedOptimizerInput(
            jnp.zeros_like(features_continuous_batch),
            jnp.zeros(features_categorical_batch.shape + (3,)),
        ),
        seed=seed,
    )
    np.testing.assert_array_equal(
        expected.continuous, actual.continuous[:, 0, :]
    )
    np.testing.assert_array_equal(
        expected.categorical, actual.categorical[:, 0, :]
    )

    scale = np.random.normal(size=[self.eagle.batch_size, 4])
    expected_logits = _create_logits_vector_simple(
        features_2d.categorical,
        features_batch_2d.categorical,
        scale,
        self.eagle.categorical_sizes,
        self.eagle.max_categorical_size,
        self.config,
    )
    actual_logits = self.eagle._create_categorical_feature_logits(
        features.categorical, features_batch.categorical, scale
    )
    np.testing.assert_allclose(
        expected_logits, actual_logits[:, 0, :, :], rtol=1e-5
    )

  @parameterized.parameters(1, 5)
  def test_create_random_perturbations(self, n_parallel):
    seed = jax.random.PRNGKey(0)
    perturbations_batch = jnp.ones(self.eagle.batch_size)
    perturbations = self.eagle._create_random_perturbations(
        perturbations_batch,
        n_parallel=n_parallel,
        seed=seed,
    )
    self.assertEqual(perturbations.continuous.shape, (2, n_parallel, 2))
    self.assertEqual(perturbations.categorical.shape, (2, n_parallel, 2, 3))

  def test_update_pool_features_and_rewards(self):
    features = vb.VectorizedOptimizerInput(
        continuous=jnp.array(
            [[[1, 2]], [[3, 4]], [[7, 7]], [[8, 8]]], dtype=jnp.float64
        ),
        categorical=jnp.array(
            [[[1, 2]], [[3, 0]], [[0, 1]], [[2, 1]]], dtype=jnp.int32
        ),
    )
    rewards = jnp.array([2, 3, 4, 1], dtype=jnp.float64)
    perturbations = jnp.array([1, 1, 1, 1], dtype=jnp.float64)

    batch_features = vb.VectorizedOptimizerInput(
        continuous=jnp.array([[[9, 9]], [[10, 10]]], dtype=jnp.float64),
        categorical=jnp.array([[[0, 0]], [[1, 1]]], dtype=jnp.int32),
    )
    batch_rewards = jnp.array([5, 0.5], dtype=jnp.float64)

    new_features, new_rewards, new_perturbations = (
        self.eagle._update_pool_features_and_rewards(
            batch_features,
            batch_rewards,
            jax.tree_util.tree_map(
                lambda f: f[: self.eagle.batch_size], features
            ),
            rewards[: self.eagle.batch_size],
            perturbations[: self.eagle.batch_size],
        )
    )
    np.testing.assert_array_equal(
        new_features.continuous,
        np.array([[[9, 9]], [[3, 4]]], dtype=np.float64),
        err_msg='Features are not equal.',
    )
    np.testing.assert_array_equal(
        new_features.categorical,
        np.array([[[0, 0]], [[3, 0]]], dtype=np.int32),
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
    features = vb.VectorizedOptimizerInput(
        continuous=jnp.array(
            [[[1, 2]], [[3, 4]], [[7, 7]], [[8, 8]]], dtype=jnp.float64
        ),
        categorical=jnp.array(
            [[[1, 2]], [[3, 0]], [[0, 1]], [[2, 1]]], dtype=jnp.int32
        ),
    )
    rewards = jnp.array([2, 3, 4, 1], dtype=jnp.float64)
    state = eagle_strategy.VectorizedEagleStrategyState(
        iterations=jnp.array(0),
        features=features,
        rewards=rewards,
        best_reward=jnp.max(rewards),
        perturbations=jnp.ones_like(rewards),
    )
    batch_features = vb.VectorizedOptimizerInput(
        continuous=jnp.array([[[9, 9]], [[10, 10]]], dtype=jnp.float64),
        categorical=jnp.array([[[0, 0]], [[1, 1]]], dtype=jnp.int32),
    )
    batch_rewards = jnp.array([5, 0.5], dtype=jnp.float64)
    seed = jax.random.PRNGKey(0)
    new_state = self.eagle.update(seed, state, batch_features, batch_rewards)
    self.assertEqual(new_state.best_reward, 5.0)
    # Test not replacing the best reward.
    batch_rewards = jnp.array([2, 4], dtype=jnp.float64)
    new_new_state = self.eagle.update(
        seed, new_state, batch_features, batch_rewards
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
    converter = converters.TrialToModelInputConverter.from_problem(problem)
    config = eagle_strategy.EagleStrategyConfig(max_pool_size=max_pool_size)
    eagle = eagle_strategy.VectorizedEagleStrategyFactory(eagle_config=config)(
        converter=converter, suggestion_batch_size=batch_size
    )
    self.assertEqual(eagle.pool_size, max_pool_size)
    self.assertEqual(eagle.batch_size, expected_batch_size)

  def test_trim_pool(self):
    pc = self.config.perturbation
    features_batch = vb.VectorizedOptimizerInput(
        continuous=jnp.array([[[1, 2]], [[3, 4]]], dtype=jnp.float64),
        categorical=jnp.array([[[1, 2]], [[3, 0]]], dtype=jnp.int32),
    )
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
        new_features.continuous[0],
        features_batch.continuous[0],
        err_msg='Continuous features are not equal.',
    )
    np.testing.assert_array_almost_equal(
        new_features.categorical[0],
        features_batch.categorical[0],
        err_msg='Categorical features are not equal.',
    )
    self.assertTrue(
        np.all(
            np.not_equal(
                new_features.continuous[1], features_batch.continuous[1]
            )
        ),
        msg='Continuous features are not equal.',
    )
    self.assertTrue(
        np.all(
            np.not_equal(
                new_features.categorical[1], features_batch.categorical[1]
            )
        ),
        msg='Categorical features are not equal.',
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
    converter = converters.TrialToModelInputConverter.from_problem(problem)
    eagle = eagle_factory(converter)
    self.assertEqual(eagle.n_feature_dimensions.continuous, 3)
    self.assertEqual(eagle.n_feature_dimensions.categorical, 0)

  def test_optimize_with_eagle(self):

    eagle_factory = eagle_strategy.VectorizedEagleStrategyFactory()
    optimizer = vb.VectorizedOptimizerFactory(strategy_factory=eagle_factory)(
        self.converter
    )
    optimizer(
        score_fn=lambda x, _: -jnp.sum(x.continuous.padded_array, 1), count=1
    )

  def test_optimize_with_eagle_continuous_only(self):
    problem = vz.ProblemStatement()
    root = problem.search_space.select_root()
    root.add_float_param('x1', 0.0, 1.0)
    root.add_float_param('x2', 0.0, 1.0)
    root.add_float_param('x3', 0.0, 1.0)
    converter = converters.TrialToModelInputConverter.from_problem(problem)
    eagle_factory = eagle_strategy.VectorizedEagleStrategyFactory()
    optimizer = vb.VectorizedOptimizerFactory(strategy_factory=eagle_factory)(
        converter
    )
    n_parallel = 5
    results = optimizer(
        score_fn=lambda x, _: -jnp.sum(x.continuous.padded_array, axis=(1, 2)),
        count=1,
        n_parallel=n_parallel,
    )
    self.assertSequenceEqual(
        results.features.continuous.shape, (1, n_parallel, 3)
    )

  def test_optimize_with_eagle_padding(self):
    problem = vz.ProblemStatement()
    root = problem.search_space.select_root()
    root.add_float_param('x1', 0.0, 1.0)
    root.add_float_param('x2', 0.0, 1.0)
    root.add_float_param('x3', 0.0, 1.0)
    converter = converters.TrialToModelInputConverter.from_problem(
        problem,
        padding_schedule=padding.PaddingSchedule(
            num_trials=padding.PaddingType.POWERS_OF_2,
            num_features=padding.PaddingType.POWERS_OF_2,
        ),
    )
    eagle_factory = eagle_strategy.VectorizedEagleStrategyFactory()
    optimizer = vb.VectorizedOptimizerFactory(strategy_factory=eagle_factory)(
        converter
    )
    n_parallel = 2
    results = optimizer(
        score_fn=lambda x, _: -jnp.sum(x.continuous.padded_array, axis=(1, 2)),
        count=1,
        n_parallel=n_parallel,
    )
    self.assertSequenceEqual(
        results.features.continuous.shape, (1, n_parallel, 4)
    )

  @parameterized.parameters(
      {'num_prior_trials': 0},
      {'num_prior_trials': 3},
      {'num_prior_trials': 53},
  )
  def test_optimize_with_eagle_trials_padding(self, num_prior_trials):
    eagle_factory = eagle_strategy.VectorizedEagleStrategyFactory()
    problem = vz.ProblemStatement()
    root = problem.search_space.select_root()
    root.add_float_param('x1', 0.0, 1.0)
    root.add_float_param('x2', 0.0, 1.0)
    root.add_float_param('x3', 0.0, 1.0)
    root.add_categorical_param('c1', ['a', 'b', 'c'])
    root.add_categorical_param('c2', ['d', 'e', 'f', 'g'])

    rng = np.random.default_rng(seed=1)
    prior_features_continuous = rng.uniform(size=(num_prior_trials, 3))
    prior_features_categorical = rng.integers(
        low=0,
        high=[[3, 4]],
        size=(num_prior_trials, 2),
    )

    def score_fn(x, _):
      return jnp.sum(
          x.continuous.replace_fill_value(0).padded_array, axis=(1,)
      ) + jnp.sum(x.categorical.replace_fill_value(0).padded_array, axis=(1,))

    count = 3

    converter = converters.TrialToModelInputConverter.from_problem(
        problem,
    )
    optimizer = vb.VectorizedOptimizerFactory(strategy_factory=eagle_factory)(
        converter
    )
    prior_features = types.ModelInput(
        continuous=types.PaddedArray.as_padded(prior_features_continuous),
        categorical=types.PaddedArray.as_padded(prior_features_categorical),
    )
    results = optimizer(
        score_fn=score_fn,
        count=count,
        prior_features=None if num_prior_trials == 0 else prior_features,
    )

    padding_schedule = padding.PaddingSchedule(
        num_trials=padding.PaddingType.MULTIPLES_OF_10,
    )
    padding_converter = converters.TrialToModelInputConverter.from_problem(
        problem,
        padding_schedule=padding_schedule,
    )
    padding_optimizer = vb.VectorizedOptimizerFactory(
        strategy_factory=eagle_factory
    )(padding_converter)
    padded_prior_features = types.ModelInput(
        continuous=padding_schedule.pad_features(prior_features_continuous),
        categorical=padding_schedule.pad_features(prior_features_categorical),
    )
    padding_results = padding_optimizer(
        score_fn=score_fn,
        count=count,
        prior_features=None if num_prior_trials == 0 else padded_prior_features,
    )

    self.assertSequenceEqual(results.features.continuous.shape, (count, 1, 3))
    self.assertSequenceEqual(results.features.categorical.shape, (count, 1, 2))
    np.testing.assert_array_almost_equal(
        results.features.continuous,
        padding_results.features.continuous,
        decimal=5,
    )
    np.testing.assert_array_almost_equal(
        results.features.categorical,
        padding_results.features.categorical,
        decimal=5,
    )

  def test_factory(self):
    self.assertEqual(self.eagle.n_feature_dimensions.continuous, 2)
    self.assertEqual(self.eagle.n_feature_dimensions.categorical, 2)
    self.assertLen(self.eagle.categorical_sizes, 2)

  def test_sample_categorical_features(self):
    # features shouldn't have values in oov_mask, and have the structure of:
    # [c1,c1,c1,f1,c2,c2,c2,c2,d1]
    num_parallel = 3
    sampled_features = self.eagle._sample_random_features(
        20, n_parallel=num_parallel, seed=jax.random.PRNGKey(0)
    )
    self.assertTrue(
        np.all(
            (sampled_features.continuous >= 0.0)
            & (sampled_features.continuous <= 1.0)
        )
    )
    self.assertTrue(
        np.all(
            sampled_features.categorical
            < np.array(self.eagle.categorical_sizes)
        )
    )

  def test_prior_trials(self):
    config = eagle_strategy.EagleStrategyConfig(
        visibility=1, gravity=1, pool_size=4, prior_trials_pool_pct=1.0
    )
    problem = vz.ProblemStatement()
    root = problem.search_space.select_root()
    root.add_float_param('x1', 0.0, 1.0)
    root.add_float_param('x2', 0.0, 1.0)
    root.add_categorical_param('c1', ['a', 'b', 'c'])
    converter = converters.TrialToModelInputConverter.from_problem(problem)

    prior_features_continuous = jnp.array(
        [[[1, -1]], [[2, 1]], [[3, 2]], [[4, 5]]], dtype=jnp.float32
    )
    prior_features = vb.VectorizedOptimizerInput(
        continuous=prior_features_continuous,
        categorical=jnp.array([[0], [2], [1], [1]])[:, jnp.newaxis],
    )
    prior_rewards = jnp.array([1, 2, 3, 4])
    eagle = eagle_strategy.VectorizedEagleStrategyFactory(
        eagle_config=config,
    )(converter=converter, suggestion_batch_size=2)
    init_state = eagle.init_state(
        jax.random.PRNGKey(0),
        prior_features=prior_features,
        prior_rewards=prior_rewards,
    )
    jax.tree_util.tree_map(
        lambda x, y: np.testing.assert_array_equal(y, jnp.flip(x, axis=0)),
        init_state.features,
        prior_features,
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
    root.add_categorical_param('c1', ['a', 'b', 'c'])
    converter = converters.TrialToModelInputConverter.from_problem(problem)

    n_parallel = 3
    prior_features = vb.VectorizedOptimizerInput(
        continuous=jnp.asarray(np.random.randn(n_prior_trials, n_parallel, 2)),
        categorical=jnp.asarray(
            np.random.randint(3, size=(n_prior_trials, n_parallel, 1))
        ),
    )
    prior_rewards = jnp.asarray(np.random.randn(n_prior_trials))

    eagle = eagle_strategy.VectorizedEagleStrategyFactory(
        eagle_config=config,
    )(converter=converter, suggestion_batch_size=2)
    init_state = eagle.init_state(
        jax.random.PRNGKey(0),
        n_parallel=n_parallel,
        prior_features=prior_features,
        prior_rewards=prior_rewards,
    )
    self.assertEqual(
        init_state.features.continuous.shape, (config.pool_size, n_parallel, 2)
    )
    self.assertEqual(
        init_state.features.categorical.shape, (config.pool_size, n_parallel, 1)
    )


if __name__ == '__main__':
  absltest.main()
