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

"""Eagle Strategy Optimizer.

Implements a variation of Eagle Strategy without the Levy random walk, aka
Firefly Algorithm (FA) for the purpose of optimizing a given objective function.

Reference: Yang XS. (2009) Firefly Algorithms for Multimodal Optimization.
In: Stochastic Algorithms: Foundations and Applications (SAGA) 2009.
DOI: https://doi.org/10.1007/978-3-642-04944-6_14

Firefly Algorithm Summary
=========================
FA is a genetic algorithm that maintains a pool of fireflies. Each
firefly emits a light whose intensity is non-decreasing in (or simply equal
to) the objective value. Each iteration, a firefly chases after a brighter
firefly, but the brightness it perceives decreases in distance. This allows
multiple "clusters" to form, as opposed to all fireflies collapsing to a
single point. Not included in the original algorithm, we added "repulsion" which
in addition to the "attraction" forces, meaning fireflies move towards the
bright spots as well as away from the dark spots. We also support non-decimal
parameter types (categorical, discrete, integer), and treat them uniquely when
computing distance, adding pertrubation, and mutating fireflies.

For more details, see the linked paper.

OSS Vizier Implementation Summary
=================================
The fireflies are stored in three JAX arrays: features, rewards, perturbations.
Each iteration we mutate 'batch_size' fireflies to generate new features. The
new features are evaluated on the objective function to obtain their associated
rewards and update the pool where improvement was obtained, and decrease the
perturbation factor otherwise.

If the firefly's perturbation reaches the `perturbation_lower_bound` threshold
it's removed and replaced with new random features.

For performance consideration, the 'pool size' is a multiplier of the
'batch size', and so each iteration the pool is sliced to obtain the current
fireflies to be mutated.

Passing prior trials is supported for knowledge-transfering from previous runs
of the optimizer. Prior trials are used to populate the pool with fireflies
which are closer to the optimum.

Note that the strategy assumes that the search space is not conditional.

Example
=======
# Construct the optimizer with eagle strategy.
optimizer = VectorizedOptimizerFactory(
    VectorizedEagleStrategyFactory(),
)
# Run the optimization.
trials = optimizer.optimize(problem_statement, objective_function)
"""
# pylint: disable=g-long-lambda

import enum
import logging
import math
from typing import Optional, Tuple

import attr
from flax import struct
import jax
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import types
from vizier.pyvizier import converters

tfd = tfp.distributions


@enum.unique
class MutateNormalizationType(enum.IntEnum):
  """The force normalization mode. Use IntEnum for JIT compatibility."""

  MEAN = 0
  RANDOM = 1
  UNNORMALIZED = 2


@struct.dataclass
class EagleStrategyConfig:
  """Eagle Strategy optimizer config.

  Attributes:
    visibility: The sensetivity to distance between flies when computing pulls.
    gravity: The maximum amount of attraction pull.
    negative_gravity: The maximum amount of repulsion pull.
    perturbation: The default amount of noise for perturbation.
    categorical_perturbation_factor: A factor to apply on categorical params.
    pure_categorical_perturbation_factor: A factor on purely categorical space.
    prob_same_category_without_perturbation: Baseline probability of selecting
      the same category.
    perturbation_lower_bound: The threshold below flies are removed from pool.
    penalize_factor: The perturbation decrease for unsuccessful flies.
    pool_size_exponent: An exponent for computing pool size based on search
      space size.
    pool_size: An optional way to set the pool size. If not 0, this takes
      precedent over the programatic pool size computation.
    max_pool_size: Ceiling on pool size.
    mutate_normalization_type: The type of force mutation normalizatoin used for
      damping down the force intensity to stay within reasonable bounds.
    normalization_scale: A mutation force scale-factor to control intensity.
    prior_trials_pool_pct: The percentage of the pool populated by prior trials.
  """

  # Visibility
  visibility: float = 0.45
  # Gravity
  gravity: float = 1.5
  negative_gravity: float = 0.008
  # Perturbation
  perturbation: float = 0.16
  categorical_perturbation_factor: float = 1.0
  pure_categorical_perturbation_factor: float = 30
  prob_same_category_without_perturbation: float = 0.98
  # Penalty
  perturbation_lower_bound: float = 7e-5
  penalize_factor: float = 7e-1
  # Pool size
  pool_size_exponent: float = struct.field(pytree_node=False, default=1.2)
  pool_size: int = struct.field(pytree_node=False, default=0)
  max_pool_size: int = struct.field(pytree_node=False, default=100)
  # Force normalization mode
  mutate_normalization_type: MutateNormalizationType = struct.field(
      pytree_node=False,
      default=MutateNormalizationType.MEAN,
  )
  # Multiplier factor when using normalized modes
  normalization_scale: float = 0.5
  # The percentage of the firefly pool to be populated with prior trials
  prior_trials_pool_pct: float = struct.field(pytree_node=False, default=0.96)


@attr.define(frozen=True)
class VectorizedEagleStrategyFactory(vb.VectorizedStrategyFactory):
  """Eagle strategy factory."""

  eagle_config: EagleStrategyConfig = attr.field(factory=EagleStrategyConfig)

  def __call__(
      self,
      converter: converters.TrialToModelInputConverter,
      suggestion_batch_size: Optional[int] = None,
  ) -> "VectorizedEagleStrategy":
    """Create a new vectorized eagle strategy.

    In order to create the strategy a converter has to be passed, which the
    strategy will then store a pointer to. The strategy uses the converter to
    get information about the original Vizier Parameters and their relation to
    the array indices (which index belong to which Vizier Parameter). This is
    useful for example for sampling CATEGORICAL features.

    Arguments:
      converter: TrialToArrayConverter that matches the converter used in
        computing the objective / acuisition function.
      suggestion_batch_size: The batch_size of the returned suggestion array.

    Returns:
      A new instance of VectorizedEagleStrategy.
    """
    valid_types = [
        converters.NumpyArraySpecType.DISCRETE,
        converters.NumpyArraySpecType.CONTINUOUS,
    ]
    if any(
        spec.type not in valid_types
        for spec in (
            list(converter.output_specs.continuous)
            + list(converter.output_specs.categorical)
        )
    ):
      raise ValueError("Only DISCRETE/CONTINUOUS parameters are supported!")

    empty_features = converter.to_features([])
    n_feature_dimensions_with_padding = types.ContinuousAndCategorical[int](
        continuous=empty_features.continuous.shape[-1],
        categorical=empty_features.categorical.shape[-1],
    )
    n_feature_dimensions = types.ContinuousAndCategorical(
        continuous=len(converter.output_specs.continuous),
        categorical=len(converter.output_specs.categorical),
    )

    categorical_sizes = []
    for spec in converter.output_specs.categorical:
      categorical_sizes.append(spec.bounds[1])
    if categorical_sizes:
      max_categorical_size = max(categorical_sizes)
    else:
      max_categorical_size = 0
    extra_categories = (
        n_feature_dimensions_with_padding.categorical
        - n_feature_dimensions.categorical
    )
    categorical_sizes = categorical_sizes + [0] * extra_categories

    n_features = (
        n_feature_dimensions.continuous + n_feature_dimensions.categorical
    )
    pool_size = self.eagle_config.pool_size
    if pool_size == 0:
      pool_size = 10 + int(
          0.5 * n_features + n_features**self.eagle_config.pool_size_exponent
      )
      pool_size = min(pool_size, self.eagle_config.max_pool_size)
      if suggestion_batch_size is not None:
        # If the batch_size was set, ensure pool_size is multiply of batch_size.
        pool_size = int(
            math.ceil(pool_size / suggestion_batch_size) * suggestion_batch_size
        )
    logging.info("Pool size: %d", pool_size)
    if suggestion_batch_size is None:
      # This configuration updates all the fireflies in each iteration.
      suggestion_batch_size = pool_size
    # Use priors to populate Eagle state
    # pytype: disable=wrong-arg-types  # jnp-type
    return VectorizedEagleStrategy(
        n_feature_dimensions=n_feature_dimensions,
        n_feature_dimensions_with_padding=n_feature_dimensions_with_padding,
        batch_size=suggestion_batch_size,
        config=self.eagle_config,
        pool_size=pool_size,
        categorical_sizes=jnp.array(categorical_sizes),
        max_categorical_size=max_categorical_size,
        dtype=converter._impl.dtype,
    )
    # pytype: enable=wrong-arg-types


@struct.dataclass
class VectorizedEagleStrategyState:
  """Container for Eagle strategy state."""

  iterations: jax.Array  # Scalar integer.
  features: vb.VectorizedOptimizerInput  # (pool_size, n_parallel, n_features).
  rewards: jax.Array  # Shape (pool_size,).
  best_reward: jax.Array  # Scalar float.
  perturbations: jax.Array  # Shape (pool_size,).


def _compute_features_dist(
    x_batch: vb.VectorizedOptimizerInput, x_pool: vb.VectorizedOptimizerInput
) -> jax.Array:
  """Computes distance between features (or parallel feature batches)."""
  dist = jnp.zeros([], dtype=x_batch.continuous.dtype)
  if x_batch.continuous.size > 0:
    x_batch_cont = jnp.reshape(
        x_batch.continuous, (x_batch.continuous.shape[0], -1)
    )
    x_pool_cont = jnp.reshape(
        x_pool.continuous, (x_pool.continuous.shape[0], -1)
    )
    continuous_dists = (
        jnp.sum(x_batch_cont**2, axis=-1, keepdims=True)
        + jnp.sum(x_pool_cont**2, axis=-1)
        - 2.0 * jnp.matmul(x_batch_cont, x_pool_cont.T)
    )  # shape (batch_size, pool_size)
    dist = dist + continuous_dists

  if x_batch.categorical.size > 0:
    x_batch_cat = jnp.reshape(
        x_batch.categorical, (x_batch.categorical.shape[0], -1)
    )
    x_pool_cat = jnp.reshape(
        x_pool.categorical, (x_pool.categorical.shape[0], -1)
    )
    categorical_diffs = (x_batch_cat[..., jnp.newaxis, :] != x_pool_cat).astype(
        x_batch.continuous.dtype
    )
    dist = dist + jnp.sum(categorical_diffs, axis=-1)
  return dist


def _mask_flip(
    prior_features: vb.VectorizedOptimizerInput, prior_rewards: types.Array
) -> Tuple[vb.VectorizedOptimizerInput, types.Array]:
  """Flips the ordering of the elements in `prior_rewards` and `prior_features`.

  Args:
    prior_features: Prior features to be flipped.
    prior_rewards: Prior rewards to be flipped.

  Returns:
    A tuple of flipped prior features and prior rewards such that all elements
    corresponding to -inf entries in `prior_rewards` are at the end, while all
    other elements have the opposite order. For example, if `prior_rewards` is
    [1, -jnp.inf, 3, -jnp.inf ,2],  `flipped_prior_rewards` will be
    [2, 3, 1, -jnp.inf, -jnp.inf].
  """
  mask = jnp.invert(jnp.isneginf(prior_rewards))
  indices = jnp.flip(
      jnp.argsort(jnp.where(mask, jnp.arange(prior_rewards.shape[0]), -1))
  )
  flipped_prior_features = jax.tree_util.tree_map(
      lambda x: x[indices], prior_features
  )
  flipped_prior_rewards = prior_rewards[indices]
  return flipped_prior_features, flipped_prior_rewards


@struct.dataclass
class VectorizedEagleStrategy(
    vb.VectorizedStrategy[VectorizedEagleStrategyState]
):
  """Eagle strategy implementation for maximization problem based on Numpy.

  Attributes:
    config: The Eagle strategy configuration.
    n_features: The number of features.
    batch_size: The number of suggestions generated at each suggestion call.
    pool_size: The total number of flies in the pool.
  """

  n_feature_dimensions: types.ContinuousAndCategorical[int]
  categorical_sizes: jax.Array
  n_feature_dimensions_with_padding: types.ContinuousAndCategorical[int] = (
      struct.field(pytree_node=False)
  )
  max_categorical_size: int = struct.field(pytree_node=False)
  pool_size: int = struct.field(pytree_node=False)
  dtype: jnp.dtype = struct.field(pytree_node=False)
  batch_size: Optional[int] = struct.field(pytree_node=False, default=None)
  config: EagleStrategyConfig = struct.field(
      default_factory=EagleStrategyConfig
  )

  def init_state(
      self,
      seed: jax.Array,
      n_parallel: int = 1,
      *,
      prior_features: Optional[vb.VectorizedOptimizerInput] = None,
      prior_rewards: Optional[types.Array] = None,
  ) -> VectorizedEagleStrategyState:
    """Initializes the state."""
    if prior_features is not None and prior_rewards is not None:
      if prior_features.continuous.shape[1] != n_parallel:
        raise ValueError(
            "`prior_features.continuous` dimension 1 "
            f"({prior_features.continuous.shape[1]}) "
            f"doesn't match n_parallel ({n_parallel})!"
        )
      if prior_features.categorical.shape[1] != n_parallel:
        raise ValueError(
            "`prior_features.categorical` dimension 1 "
            f"({prior_features.categorical.shape[1]}) "
            f"doesn't match n_parallel ({n_parallel})!"
        )
      init_features = self._populate_pool_with_prior_trials(
          seed, prior_features, prior_rewards
      )
    else:
      init_features = self._sample_random_features(
          self.pool_size, n_parallel=n_parallel, seed=seed
      )
    # pytype: disable=wrong-arg-types  # jnp-type
    return VectorizedEagleStrategyState(
        iterations=jnp.array(0),
        features=init_features,
        rewards=jnp.ones(self.pool_size) * -jnp.inf,
        best_reward=-jnp.inf,
        perturbations=jnp.ones(self.pool_size) * self.config.perturbation,
    )
    # pytype: enable=wrong-arg-types

  def _sample_random_features(
      self, num_samples: int, n_parallel: int, seed: jax.Array
  ) -> vb.VectorizedOptimizerInput:
    cont_seed, cat_seed = jax.random.split(seed)

    if self.max_categorical_size > 0:
      sizes = jnp.array(self.categorical_sizes)[:, jnp.newaxis]
      logits = jnp.where(
          jnp.arange(self.max_categorical_size) < sizes, 0.0, -jnp.inf
      )
      random_categorical_features = (
          tfd.Categorical(logits=logits)
          .sample((num_samples, n_parallel), seed=cat_seed)
          .astype(types.INT_DTYPE)
      )
    else:
      random_categorical_features = jnp.zeros(
          [num_samples, n_parallel, 0], types.INT_DTYPE
      )
    return types.ContinuousAndCategoricalArray(
        continuous=jax.random.uniform(
            cont_seed,
            shape=(
                num_samples,
                n_parallel,
                self.n_feature_dimensions_with_padding.continuous,
            ),
        ),
        categorical=random_categorical_features,
    )

  def _populate_pool_with_prior_trials(
      self,
      seed: jax.Array,
      prior_features: types.ContinuousAndCategoricalArray,
      prior_rewards: types.Array,
  ) -> types.ContinuousAndCategoricalArray:
    """Populate the pool with prior trials.

    Args:
      seed: Random seed.
      prior_features: (n_prior_features, n_parallel, features_count)
      prior_rewards: (n_prior_features, )

    Returns:
      initial_features

    A portion of the pool is first populated with random features based on
    'prior_trials_pool_pct', then the rest of the flies are populated by
    sequentially iterate over the prior trials, finding the cloest firefly in
    the pool and replace it if the reward is better.
    """
    if prior_features is None or prior_rewards is None:
      raise ValueError("One of prior features / prior rewards wasn't provided!")

    if prior_features.continuous is not None:
      continuous_obs, _, continuous_dim = prior_features.continuous.shape
      if continuous_obs != prior_rewards.shape[0]:
        raise ValueError(
            f"prior continuous features shape ({continuous_obs}) doesn't match"
            f" prior rewards shape ({prior_rewards.shape[0]})!"
        )
      expected_dim = self.n_feature_dimensions_with_padding.continuous
      if continuous_dim != expected_dim:
        raise ValueError(
            f"prior continuous features shape ({continuous_dim}) doesn't match "
            f"n_features {expected_dim}!"
        )
    if prior_features.categorical is not None:
      categorical_obs, _, categorical_dim = prior_features.categorical.shape
      if categorical_obs != prior_rewards.shape[0]:
        raise ValueError(
            f"prior categorical features shape ({categorical_obs}) doesn't "
            f"match prior rewards shape ({prior_rewards.shape[0]})!"
        )
      expected_dim = self.n_feature_dimensions_with_padding.categorical
      if categorical_dim != expected_dim:
        raise ValueError(
            f"prior categorical features shape ({categorical_dim}) doesn't"
            f" match n_features {expected_dim}!"
        )
    if len(prior_rewards.shape) > 1:
      raise ValueError("prior rewards is expected to be 1D array!")

    # Reverse the order of prior trials to assign more weight to recent trials.
    flipped_prior_features, flipped_prior_rewards = _mask_flip(
        prior_features, prior_rewards
    )

    n_parallel = flipped_prior_features.continuous.shape[1]

    pool_random_features = self._sample_random_features(
        self.pool_size, n_parallel, seed
    )
    n_random_flies = int(
        self.pool_size * (1 - self.config.prior_trials_pool_pct)
    )
    # The pool is configured to have at least this many random trials.
    init_features = jax.tree_util.tree_map(
        lambda x: x[:n_random_flies, :, :], pool_random_features
    )
    pool_left_space = self.pool_size - n_random_flies
    # When the number of prior trials is smaller than configured, we fill in
    # random trials.
    random_features = jax.tree_util.tree_map(
        lambda x: x[n_random_flies:, :, :], pool_random_features
    )

    # Starts with the most recent `pool_left_space` prior trials as the chosen
    # set of trials, and loops through the remaining prior trials. At each
    # iteration, if the remaining trial has a better reward than its closest
    # neighbor in the chosen set, it replaces the closest neighbor. Note that
    # `pool_left_space` can be larger than the number of prior trials, in which
    # case the chosen set contains all prior trials.
    features = jax.tree_util.tree_map(
        lambda x: x[:pool_left_space], flipped_prior_features
    )
    rewards = flipped_prior_rewards[:pool_left_space]

    def _loop_body(i, args):
      features, rewards = args
      ind = jnp.argmin(
          _compute_features_dist(
              jax.tree_util.tree_map(
                  lambda x: x[i][jnp.newaxis],
                  flipped_prior_features,
              ),
              features,
          ),
          axis=-1,
      )[0]
      return jax.lax.cond(
          rewards[ind] < flipped_prior_rewards[i],
          lambda: (
              jax.tree_util.tree_map(
                  lambda f, pf: f.at[ind].set(pf[i]),
                  features,
                  flipped_prior_features,
              ),
              rewards.at[ind].set(flipped_prior_rewards[i]),
          ),
          lambda: (features, rewards),
      )

    # TODO: Use a vectorized method to populate the pool and avoid
    # the for-loop.
    features, _ = jax.lax.fori_loop(
        lower=pool_left_space,
        upper=prior_rewards.shape[0],
        body_fun=_loop_body,
        init_val=(features, rewards),
    )

    num_chosen_trials = rewards.shape[0]
    # Replaces padded trials in the chosen trials with random trials.
    features = jax.tree_util.tree_map(
        lambda x, y: jnp.where(
            jnp.isneginf(rewards)[:, jnp.newaxis, jnp.newaxis],
            x[:num_chosen_trials, :, :],
            y,
        ),
        random_features,
        features,
    )
    # Then ensures the chosen set of trials has exactly `pool_left_space` trials
    # by filling in random trials.
    features = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y[num_chosen_trials:, :, :]]),
        features,
        random_features,
    )

    return jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y]), init_features, features
    )

  @property
  def suggestion_batch_size(self) -> int:
    """The number of suggestions returned at each call of 'suggest'."""
    return self.batch_size

  def suggest(
      self,
      seed: jax.Array,
      state: VectorizedEagleStrategyState,
      n_parallel: int = 1,
  ) -> vb.VectorizedOptimizerInput:
    """Suggest new mutated and perturbed features.

    After initializing, at each call `batch_size` fireflies are mutated to
    generate new features using pulls (attraction/repulsion) from all other
    fireflies in the pool.

    Args:
      seed: Random seed.
      state: Current strategy state.
      n_parallel: Number of points that the acquisition function maps to a
        single value. This arg may be greater than 1 if a parallel acquisition
        function (qEI, qUCB) is used; otherwise it should be 1.

    Returns:
      suggested batch features: (batch_size, n_parallel, n_features)
    """
    batch_id = state.iterations % (self.pool_size // self.batch_size)
    start = batch_id * self.batch_size
    features_batch = jax.tree_util.tree_map(
        lambda f: jax.lax.dynamic_slice_in_dim(f, start, self.batch_size),
        state.features,
    )
    rewards_batch = jax.lax.dynamic_slice_in_dim(
        state.rewards, start, self.batch_size
    )
    perturbations_batch = jax.lax.dynamic_slice_in_dim(
        state.perturbations, start, self.batch_size
    )
    features_seed, perturbations_seed = jax.random.split(seed)

    def _mutate_features(features_batch_):
      perturbations = self._create_random_perturbations(
          perturbations_batch, n_parallel, perturbations_seed
      )
      return self._create_features(
          state.features,
          state.rewards,
          features_batch_,
          rewards_batch,
          perturbations,
          features_seed,
      )

    # If the strategy is still initializing, return the random/prior features.
    new_features = jax.lax.cond(
        state.iterations < self.pool_size // self.batch_size,
        lambda x: x,
        _mutate_features,
        features_batch,
    )

    # TODO: The range of features is not always [0, 1].
    #   Specifically, for features that are single-point, it can be [0, 0]; we
    #   also want this code to be aware of the feature's bounds to enable
    #   contextual bandit operation.  Note that if a parameter's bound changes,
    #   we might also want to change the firefly noise or normalizations.
    return vb.VectorizedOptimizerInput(
        continuous=jnp.clip(new_features.continuous, 0.0, 1.0),
        categorical=new_features.categorical,
    )

  def _create_features(
      self,
      features: vb.VectorizedOptimizerInput,
      rewards: jax.Array,
      features_batch: vb.VectorizedOptimizerInput,
      rewards_batch: jax.Array,
      perturbations_batch: types.ContinuousAndCategoricalArray,
      seed: jax.Array,
  ) -> vb.VectorizedOptimizerInput:
    """Create new batch of mutated and perturbed features.

    The pool fireflies forces (pull/push) are being normalized to ensure the
    combined force doesn't throw the firefly too far. Mathematically, the
    normalization guarantees that the combined normalized force is within the
    simplex constructed by the unnormalized forces and therefore within bounds.

    Args:
      features: (pool_size, n_parallel, n_features)
      rewards: (pool_size,)
      features_batch: (batch_size, n_parallel, n_features)
      rewards_batch: (batch_size,)
      perturbations_batch: (batch_size,)
      seed: Random seed.

    Returns:
      batch features: (batch_size, n_parallel, n_features)
    """
    # Compute the pairwise squared distances between the features batch and the
    # pool. We use a less numerically precise squared distance formulation to
    # avoid materializing a possibly large intermediate of shape
    # (batch_size, pool_size, n_features).
    dists = _compute_features_dist(features_batch, features)

    # Compute the scaled direction for applying pull between two flies.
    # scaled_directions[i,j] := direction of force applied by fly 'j' on fly
    # 'i'. Note that to compute 'directions' we might perform subtract with
    # removed flies with having value of -np.inf. Moreover, we might even
    # subtract between two removed flies which will result in np.nan. Both cases
    # are handled when computing the actual feautre changes applying a relevant
    # mask.
    directions = rewards - rewards_batch[:, jnp.newaxis]
    scaled_directions = jnp.where(
        directions >= 0.0, self.config.gravity, -self.config.negative_gravity
    )  # shape (batch_size, pool_size)

    # Normalize the distance by the number of features.
    # Get the number of non-padded features.
    n_feature_dimensions = sum(
        jax.tree_util.tree_leaves(self.n_feature_dimensions)
    )
    force = jnp.exp(
        -self.config.visibility * dists / n_feature_dimensions * 10.0
    )
    scaled_force = scaled_directions * force
    # Handle removed fireflies without updated rewards.
    finite_ind = jnp.isfinite(rewards).astype(scaled_force.dtype)

    # Ignore fireflies that were removed from the pool.
    scaled_force = scaled_force * finite_ind

    # Separate forces to pull and push so to normalize them separately.
    scaled_pulls = jnp.maximum(scaled_force, 0.0)
    scaled_push = jnp.minimum(scaled_force, 0.0)

    seed, categorical_seed = jax.random.split(seed)
    if self.config.mutate_normalization_type == MutateNormalizationType.MEAN:
      # Divide the push and pull forces by the number of flies participating.
      # Also multiply by normalization_scale.
      # pytype: disable=wrong-arg-types  # jnp-type
      norm_scaled_pulls = self.config.normalization_scale * jnp.nan_to_num(
          scaled_pulls / jnp.sum(scaled_pulls > 0.0, axis=1, keepdims=True), 0
      )
      norm_scaled_push = self.config.normalization_scale * jnp.nan_to_num(
          scaled_push / jnp.sum(scaled_push < 0.0, axis=1, keepdims=True), 0
      )
      # pytype: enable=wrong-arg-types
    elif self.config.mutate_normalization_type == (
        MutateNormalizationType.RANDOM
    ):
      # Create random matrices and normalize each row, s.t. the sum is 1.
      pull_seed, push_seed = jax.random.split(seed)
      scaled_pulls_pos = scaled_pulls > 0
      pull_rand_matrix = (
          jax.random.uniform(pull_seed, shape=scaled_pulls.shape)
          * scaled_pulls_pos
      )
      pull_weight_matrix = pull_rand_matrix / jnp.sum(
          pull_rand_matrix, axis=1, keepdims=True
      )
      push_rand_matrix = (
          jax.random.uniform(push_seed, shape=scaled_pulls.shape)
          * scaled_pulls_pos
      )
      push_weight_matrix = push_rand_matrix / jnp.sum(
          push_rand_matrix, axis=1, keepdims=True
      )
      # Normalize pulls/pulls by the weight matrices.
      # Also multiply by normalization_scale.
      norm_scaled_pulls = (
          self.config.normalization_scale * scaled_pulls * pull_weight_matrix
      )
      norm_scaled_push = (
          self.config.normalization_scale * scaled_push * push_weight_matrix
      )
    elif self.config.mutate_normalization_type == (
        MutateNormalizationType.UNNORMALIZED
    ):
      # Doesn't normalize the forces. Use this option with caution.
      norm_scaled_pulls = scaled_pulls
      norm_scaled_push = scaled_push

    # Sums normalized forces (pull/push) of all fireflies. This is equivalent to
    # features_dist[i, j] := distance between fly 'j' and fly 'i'
    # but avoids materializing the large pairwise distance matrix.
    scale = norm_scaled_pulls + norm_scaled_push
    flat_features = jnp.reshape(
        features.continuous, (features.continuous.shape[0], -1)
    )
    flat_features_batch = jnp.reshape(
        features_batch.continuous, (features_batch.continuous.shape[0], -1)
    )

    # TODO: Consider computing per batch member.
    features_changes_continuous = jnp.matmul(
        scale, flat_features
    ) - flat_features_batch * jnp.sum(scale, axis=-1, keepdims=True)

    features_continuous = (
        features_batch.continuous
        + jnp.reshape(
            features_changes_continuous, features_batch.continuous.shape
        )
        + perturbations_batch.continuous
    )
    if self.max_categorical_size > 0:
      features_categorical_logits = (
          self._create_categorical_feature_logits(
              features.categorical, features_batch.categorical, scale
          )
          + perturbations_batch.categorical
      )
      features_categorical = tfd.Categorical(
          logits=features_categorical_logits
      ).sample(seed=categorical_seed)
    else:
      features_categorical = jnp.zeros(
          features_batch.continuous.shape[:2] + (0,), dtype=types.INT_DTYPE
      )
    return vb.VectorizedOptimizerInput(
        continuous=features_continuous, categorical=features_categorical
    )

  def _create_logits_vector(
      self,
      features_one_category: jax.Array,  # [pool_size]
      feature_batch_member_one_category: jax.Array,  # scalar integer
      scale_batch_member: jax.Array,  # [pool_size]
      feature_size: jax.Array,
  ):  # scalar
    categories = jnp.arange(self.max_categorical_size)
    logit_same_category = jnp.log(
        self.config.prob_same_category_without_perturbation
    )
    logit_different_category = jnp.log(
        (1.0 - self.config.prob_same_category_without_perturbation)
        / (feature_size - 1.0)
    )
    logits = (
        jnp.sum(
            jnp.where(
                categories[:, jnp.newaxis] == features_one_category,
                scale_batch_member,
                0.0,
            ),
            axis=-1,
        )
        + logit_different_category
    )
    logits = jnp.where(categories < feature_size, logits, -jnp.inf)
    return logits.at[feature_batch_member_one_category].add(
        -jnp.sum(scale_batch_member)
        + logit_same_category
        - logit_different_category
    )  # [num_categories]

  def _create_logits_one_feature(
      self,
      features_one_category: jax.Array,  # [pool_size, num_parallel]
      features_batch_one_category: jax.Array,  # [batch_size, num_parallel]
      scale: jax.Array,  # [batch_size, pool_size]
      feature_size: jax.Array,  # scalar
  ):
    return jax.vmap(  # map over batch
        jax.vmap(self._create_logits_vector, in_axes=(-1, -1, None, None)),
        in_axes=(None, 0, 0, None),
    )(
        features_one_category, features_batch_one_category, scale, feature_size
    )  # [batch_size, num_parallel, max_num_categories]

  def _create_categorical_feature_logits(
      self,
      features: jax.Array,  # [pool_size, num_parallel, num_features]
      features_batch: jax.Array,  # [batch_size, num_parallel, num_features]
      scale: jax.Array,  # [batch_size, pool_size]
  ):
    return jax.vmap(
        self._create_logits_one_feature, in_axes=(-1, -1, None, 0), out_axes=2
    )(
        features, features_batch, scale, jnp.array(self.categorical_sizes)
    )  # [batch_size, num_parallel, num_features, num_categories]

  def _create_random_perturbations(
      self,
      perturbations_batch: jax.Array,
      n_parallel: int,
      seed: jax.Array,
  ) -> types.ContinuousAndCategoricalArray:
    """Create random perturbations for the newly created batch.

    Args:
      perturbations_batch: (batch_size,)
      n_parallel: Number of points that the acquisition function maps to a
        single value. This arg may be greater than 1 if a parallel acquisition
        function (qEI, qUCB) is used; otherwise it should be 1.
      seed: Random seed.

    Returns:
      perturbations: (batch_size, n_parallel, n_features)
    """
    cont_seed, cat_seed = jax.random.split(seed)
    # Generate normalized noise for each batch.
    batch_noise_continuous = jax.random.laplace(
        cont_seed,
        shape=(
            self.batch_size,
            n_parallel,
            self.n_feature_dimensions_with_padding.continuous,
        ),
    )
    if self.n_feature_dimensions_with_padding.continuous > 0:
      batch_noise_continuous /= jnp.max(
          jnp.abs(batch_noise_continuous), axis=1, keepdims=True
      )

    if self.n_feature_dimensions_with_padding.continuous == 0:
      categorical_perturbation = (
          self.config.pure_categorical_perturbation_factor
      )
    else:
      categorical_perturbation = self.config.categorical_perturbation_factor
    batch_noise_categorical = (
        jax.random.laplace(
            cat_seed,
            shape=(
                self.batch_size,
                n_parallel,
                self.n_feature_dimensions_with_padding.categorical,
                self.max_categorical_size,
            ),
        )
        * categorical_perturbation
    )
    return types.ContinuousAndCategoricalArray(
        continuous=(
            batch_noise_continuous
            * perturbations_batch[:, jnp.newaxis, jnp.newaxis]
        ),
        categorical=(
            batch_noise_categorical
            * perturbations_batch[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        ),
    )

  def update(
      self,
      seed: jax.Array,
      state: VectorizedEagleStrategyState,
      batch_features: vb.VectorizedOptimizerInput,
      batch_rewards: types.Array,
  ) -> VectorizedEagleStrategyState:
    """Update the firefly pool based on the new batch of results.

    Arguments:
      seed: Random seed.
      state: Current state.
      batch_features: (batch_size, n_parallel, n_features)
      batch_rewards: (batch_size, )

    Returns:
      new_state: Updated state.
    """
    new_best_reward = jnp.maximum(state.best_reward, jnp.max(batch_rewards))
    batch_id = state.iterations % (self.pool_size // self.batch_size)
    batch_start_ind = batch_id * self.batch_size
    batch_perturbations = jax.lax.dynamic_slice_in_dim(
        state.perturbations, batch_start_ind, self.batch_size
    )

    def _update(batch_features, batch_rewards, batch_perturbations):
      # Pass the new batch rewards and the associated last suggested features.
      new_batch_features, new_batch_rewards, new_batch_perturbations = (
          self._update_pool_features_and_rewards(
              batch_features,
              batch_rewards,
              jax.tree_util.tree_map(
                  lambda f: jax.lax.dynamic_slice_in_dim(
                      f, batch_start_ind, self.batch_size
                  ),
                  state.features,
              ),
              jax.lax.dynamic_slice_in_dim(
                  state.rewards, batch_start_ind, self.batch_size
              ),
              batch_perturbations,
          )
      )
      return self._trim_pool(
          new_batch_features,
          new_batch_rewards,
          new_batch_perturbations,
          new_best_reward,
          seed,
      )

    # If the strategy is still initializing, return the random/prior values.
    (new_batch_features, new_batch_rewards, new_batch_perturbations) = (
        jax.lax.cond(
            state.iterations < self.pool_size // self.batch_size,
            lambda *args: args,
            _update,
            batch_features,
            batch_rewards,
            batch_perturbations,
        )
    )

    return VectorizedEagleStrategyState(
        iterations=state.iterations + 1,
        features=jax.tree_util.tree_map(
            lambda sf, nbf: jax.lax.dynamic_update_slice_in_dim(
                sf, nbf, batch_start_ind, axis=0
            ),
            state.features,
            new_batch_features,
        ),
        rewards=jax.lax.dynamic_update_slice_in_dim(
            state.rewards, new_batch_rewards, batch_start_ind, axis=0
        ),
        best_reward=new_best_reward,
        perturbations=jax.lax.dynamic_update_slice_in_dim(
            state.perturbations,
            new_batch_perturbations,
            batch_start_ind,
            axis=0,
        ),
    )

  def _update_pool_features_and_rewards(
      self,
      batch_features: vb.VectorizedOptimizerInput,
      batch_rewards: jax.Array,
      prev_batch_features: vb.VectorizedOptimizerInput,
      prev_batch_rewards: jax.Array,
      perturbations: jax.Array,
  ) -> Tuple[vb.VectorizedOptimizerInput, jax.Array, jax.Array]:
    """Update the features and rewards for flies with improved rewards.

    Arguments:
      batch_features: (batch_size, n_parallel, n_features), new proposed
        features batch.
      batch_rewards: (batch_size,), rewards for new proposed features batch.
      prev_batch_features: (batch_size, n_parallel, n_features), previous
        features batch.
      prev_batch_rewards: (batch_size,), rewards for previous features batch.
      perturbations: (batch_size,)

    Returns:
      sliced features, sliced rewards, sliced perturbations
    """
    # Find indices of flies that their generated features made an improvement.
    improve_indx = batch_rewards > prev_batch_rewards
    # Update successful flies' with the associated last features and rewards.
    new_batch_features = jax.tree_util.tree_map(
        lambda bf, pbf: jnp.where(
            improve_indx[..., jnp.newaxis, jnp.newaxis], bf, pbf
        ),
        batch_features,
        prev_batch_features,
    )
    new_batch_rewards = jnp.where(
        improve_indx, batch_rewards, prev_batch_rewards
    )
    # Penalize unsuccessful flies.
    new_batch_perturbations = jnp.where(
        improve_indx, perturbations, perturbations * self.config.penalize_factor
    )
    return new_batch_features, new_batch_rewards, new_batch_perturbations

  def _trim_pool(
      self,
      batch_features: vb.VectorizedOptimizerInput,
      batch_rewards: jax.Array,
      batch_perturbations: jax.Array,
      best_reward: jax.Array,
      seed: jax.Array,
  ) -> Tuple[vb.VectorizedOptimizerInput, jax.Array, jax.Array]:
    """Trim the pool by replacing unsuccessful fireflies with new random ones.

    A firefly is considered unsuccessful if its current perturbation is below
    'perturbation_lower_bound' and it's not the best fly seen thus far.
    Random features are created to replace the existing ones, and rewards
    are set to -np.inf to indicate that we don't have values for those feaures
    yet and we shouldn't use them during suggest.

    Args:
      batch_features: (batch_size, n_parallel, n_features)
      batch_rewards: (batch_size,)
      batch_perturbations: (batch_size,)
      best_reward: Best reward seen so far.
      seed: Random seed.

    Returns:
      updated feature, reward, and perturbation batches.
    """
    indx = batch_perturbations < self.config.perturbation_lower_bound
    # Ensure the best firefly is never removed. For optimization purposes,
    # this logic is inside the if statement to be peformed only if needed.
    indx = indx & (batch_rewards != best_reward)

    # Replace fireflies with random features and evaluate rewards.
    random_features = self._sample_random_features(
        self.batch_size,
        n_parallel=batch_features.continuous.shape[1],
        seed=seed,
    )
    new_batch_features = jax.tree_util.tree_map(
        lambda rf, bf: jnp.where(indx[..., jnp.newaxis, jnp.newaxis], rf, bf),
        random_features,
        batch_features,
    )
    new_batch_perturbations = jnp.where(
        indx, self.config.perturbation, batch_perturbations
    )
    # Setting rewards to -inf to filter out those fireflies during suggest.
    new_batch_rewards = jnp.where(indx, -jnp.inf, batch_rewards)
    return new_batch_features, new_batch_rewards, new_batch_perturbations
