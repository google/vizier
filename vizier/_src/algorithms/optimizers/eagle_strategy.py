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
optimizer = VectorizedOptimizer(
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
import chex
import jax
from jax import numpy as jnp
from vizier._src.algorithms.optimizers import eagle_param_handler
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import types
from vizier.pyvizier import converters


@enum.unique
class MutateNormalizationType(enum.IntEnum):
  """The force normalization mode. Use IntEnum for JIT compatibility."""

  MEAN = 0
  RANDOM = 1
  UNNORMALIZED = 2


@chex.dataclass(frozen=True)
class EagleStrategyConfig:
  """Eagle Strategy optimizer config.

  Attributes:
    visibility: The sensetivity to distance between flies when computing pulls.
    gravity: The maximum amount of attraction pull.
    negative_gravity: The maximum amount of repulsion pull.
    perturbation: The default amount of noise for perturbation.
    categorical_perturbation_factor: A factor to apply on categorical params.
    pure_categorical_perturbation_factor: A factor on purely categorical space.
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
  categorical_perturbation_factor: float = 25
  pure_categorical_perturbation_factor: float = 30
  # Penalty
  perturbation_lower_bound: float = 7e-5
  penalize_factor: float = 7e-1
  # Pool size
  pool_size_exponent: float = 1.2
  pool_size: int = 0
  max_pool_size: int = 100
  # Force normalization mode
  mutate_normalization_type: MutateNormalizationType = (
      MutateNormalizationType.MEAN
  )
  # Multiplier factor when using normalized modes
  normalization_scale: float = 0.5
  # The percentage of the firefly pool to be populated with prior trials
  prior_trials_pool_pct: float = 0.96


@attr.define(frozen=True)
class VectorizedEagleStrategyFactory(vb.VectorizedStrategyFactory):
  """Eagle strategy factory."""

  eagle_config: EagleStrategyConfig = attr.field(factory=EagleStrategyConfig)

  def __call__(
      self,
      converter: converters.TrialToArrayConverter,
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
    return VectorizedEagleStrategy(
        converter=converter,
        config=self.eagle_config,
        batch_size=suggestion_batch_size,
    )


@chex.dataclass(frozen=True)
class VectorizedEagleStrategyState:
  """Container for Eagle strategy state."""

  iterations: jax.Array  # Scalar integer.
  features: jax.Array  # Shape (pool_size, n_features).
  rewards: jax.Array  # Shape (pool_size,).
  best_reward: jax.Array  # Scalar float.
  perturbations: jax.Array  # Shape (pool_size,).


@attr.define
class VectorizedEagleStrategy(vb.VectorizedStrategy):
  """Eagle strategy implementation for maximization problem based on Numpy.

  Attributes:
    converter: The converter used for the optimization problem.
    config: The Eagle strategy configuration.
    n_features: The number of features.
    batch_size: The number of suggestions generated at each suggestion call.
    pool_size: The total number of flies in the pool.
  """

  converter: converters.TrialToArrayConverter = attr.field(
      init=True, repr=False
  )
  config: EagleStrategyConfig = attr.field(init=True, repr=False)
  batch_size: Optional[int] = attr.field(init=True, default=None)
  pool_size: int = attr.field(init=False)
  # Attributes related to computations.
  _n_features: int = attr.field(init=False)
  _perturbation_factors: jax.Array = attr.field(init=False, repr=False)

  def __attrs_post_init__(self):
    self._initialize()
    logging.info("Eagle class attributes:\n%s", self)
    logging.info("Eagle configuration:\n%s", self.config)

  def __hash__(self):
    # Make this class hashable so it can be a static arg to a JIT-ed function.
    return hash(id(self))

  def _compute_pool_size(self) -> int:
    """Compute the pool size, and ensures it's a multiple of the batch_size."""
    n_params = len(self.converter.output_specs)
    pool_size = 10 + int(
        0.5 * n_params + n_params**self.config.pool_size_exponent
    )
    pool_size = min(pool_size, self.config.max_pool_size)
    if self.batch_size is not None:
      # If the batch_size was set, ensure pool_size is multiply of batch_size.
      return int(math.ceil(pool_size / self.batch_size) * self.batch_size)
    else:
      return pool_size

  def _initialize(self) -> None:
    """Initialize the designer state."""
    self._param_handler = eagle_param_handler.EagleParamHandler(
        converter=self.converter,
        categorical_perturbation_factor=self.config.categorical_perturbation_factor,
        pure_categorical_perturbation_factor=self.config.pure_categorical_perturbation_factor,
    )
    self._n_features = self._param_handler.n_features
    if self.config.pool_size > 0:
      # This allow to override the pool size computation.
      self.pool_size = self.config.pool_size
    else:
      self.pool_size = self._compute_pool_size()
    logging.info("Pool size: %d", self.pool_size)
    if self.batch_size is None:
      # This configuration updates all the fireflies in each iteration.
      self.batch_size = self.pool_size
    self._perturbation_factors = self._param_handler.perturbation_factors
    # Use priors to populate Eagle state

  def init_state(
      self,
      seed: chex.PRNGKey,
      prior_features: Optional[chex.Array] = None,
      prior_rewards: Optional[chex.Array] = None,
  ) -> VectorizedEagleStrategyState:
    """Initializes the state."""
    if prior_features is not None or prior_rewards is not None:
      init_features = self._populate_pool_with_prior_trials(
          seed, prior_features, prior_rewards
      )
    else:
      init_features = self._param_handler.random_features(self.pool_size, seed)
    return VectorizedEagleStrategyState(
        iterations=jnp.array(0),
        features=init_features,
        rewards=jnp.ones(self.pool_size) * -jnp.inf,
        best_reward=-jnp.inf,
        perturbations=jnp.ones(self.pool_size) * self.config.perturbation,
    )

  def _populate_pool_with_prior_trials(
      self,
      seed: chex.PRNGKey,
      prior_features: chex.Array,
      prior_rewards: chex.Array,
  ) -> jax.Array:
    """Populate the pool with prior trials.

    Args:
      seed: Random seed.
      prior_features: (n_prior_features, features_count)
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
    if prior_features.shape[0] != prior_rewards.shape[0]:
      raise ValueError(
          f"prior features shape ({prior_features.shape[0]}) doesn't match"
          f" prior  rewards shape ({prior_rewards.shape[0]})!"
      )
    if prior_features.shape[1] != self._n_features:
      raise ValueError(
          f"prior features shape ({prior_features.shape[1]}) doesn't match "
          f"n_features ({self._n_features})!"
      )
    if len(prior_rewards.shape) > 1:
      raise ValueError("prior rewards is expected to be 1D array!")

    # Reverse the order of prior trials to assign more weight to recent trials.
    flipped_prior_features = jnp.flip(prior_features, axis=0)
    flipped_prior_rewards = jnp.flip(prior_rewards, axis=0)

    # Fill pool with random features.
    n_random_flies = int(
        self.pool_size * (1 - self.config.prior_trials_pool_pct)
    )
    seed1, seed2 = jax.random.split(seed)
    init_features = self._param_handler.random_features(n_random_flies, seed1)
    pool_left_space = self.pool_size - n_random_flies

    if prior_features.shape[0] < pool_left_space:
      # Less prior trials than left space. Take all prior trials for the pool.
      init_features = jnp.concatenate([init_features, flipped_prior_features])
      # Randomize the rest of the pool fireflies.
      random_features = self._param_handler.random_features(
          self.pool_size - len(init_features), seed2
      )
      return jnp.concatenate([init_features, random_features])
    else:
      # More prior trials than left space. Iteratively populate the pool.
      tmp_features = flipped_prior_features[:pool_left_space]
      tmp_rewards = flipped_prior_rewards[:pool_left_space]

      def _loop_body(i, args):
        features, rewards = args
        ind = jnp.argmin(
            jnp.sum(jnp.square(flipped_prior_features[i] - features), axis=-1)
        )
        return jax.lax.cond(
            rewards[ind] < flipped_prior_rewards[i],
            lambda: (
                features.at[ind].set(flipped_prior_features[i]),
                rewards.at[ind].set(flipped_prior_rewards[i]),
            ),
            lambda: (features, rewards),
        )

      # TODO: Use a vectorized method to populate the pool and avoid
      # the for-loop.
      tmp_features, _ = jax.lax.fori_loop(
          lower=pool_left_space,
          upper=prior_features.shape[0],
          body_fun=_loop_body,
          init_val=(tmp_features, tmp_rewards),
      )
      return jnp.concatenate([init_features, tmp_features])

  @property
  def suggestion_batch_size(self) -> int:
    """The number of suggestions returned at each call of 'suggest'."""
    return self.batch_size

  def suggest(
      self, state: VectorizedEagleStrategyState, seed: chex.PRNGKey
  ) -> jax.Array:
    """Suggest new mutated and perturbed features.

    After initializing, at each call `batch_size` fireflies are mutated to
    generate new features using pulls (attraction/repulsion) from all other
    fireflies in the pool.

    Args:
      state: Current strategy state.
      seed: Random seed.

    Returns:
      suggested batch features: (batch_size, n_features)
    """
    batch_id = state.iterations % (self.pool_size // self.batch_size)
    start = batch_id * self.batch_size
    features_batch = jax.lax.dynamic_slice_in_dim(
        state.features, start, self.batch_size
    )
    rewards_batch = jax.lax.dynamic_slice_in_dim(
        state.rewards, start, self.batch_size
    )
    perturbations_batch = jax.lax.dynamic_slice_in_dim(
        state.perturbations, start, self.batch_size
    )
    features_seed, perturbations_seed, cat_seed = jax.random.split(seed, num=3)

    def _mutate_features(features_batch_):
      mutated_features = self._create_features(
          state.features,
          state.rewards,
          features_batch_,
          rewards_batch,
          features_seed,
      )
      perturbations = self._create_random_perturbations(
          perturbations_batch, perturbations_seed
      )
      return mutated_features + perturbations

    # If the strategy is still initializing, return the random/prior features.
    new_features = jax.lax.cond(
        state.iterations < self.pool_size // self.batch_size,
        lambda x: x,
        _mutate_features,
        features_batch,
    )

    new_features = self._param_handler.sample_categorical(
        new_features, cat_seed
    )
    # TODO: The range of features is not always [0, 1].
    #   Specifically, for features that are single-point, it can be [0, 0]; we
    #   also want this code to be aware of the feature's bounds to enable
    #   contextual bandit operation.  Note that if a parameter's bound changes,
    #   we might also want to change the firefly noise or normalizations.
    return jnp.clip(new_features, 0.0, 1.0)

  def _create_features(
      self,
      features: jax.Array,
      rewards: jax.Array,
      features_batch: jax.Array,
      rewards_batch: jax.Array,
      seed: chex.PRNGKey,
  ) -> jax.Array:
    """Create new batch of mutated and perturbed features.

    The pool fireflies forces (pull/push) are being normalized to ensure the
    combined force doesn't throw the firefly too far. Mathematically, the
    normalization guarantees that the combined normalized force is within the
    simplex constructed by the unnormalized forces and therefore within bounds.

    Args:
      features: (pool_size, n_features)
      rewards: (pool_size,)
      features_batch: (batch_size, n_features)
      rewards_batch: (batch_size,)
      seed: Random seed.

    Returns:
      batch features: (batch_size, n_features)
    """
    # Compute the pairwise squared distances between the features batch and the
    # pool. We use a less numerically precise squared distance formulation to
    # avoid materializing a possibly large intermediate of shape
    # (batch_size, pool_size, n_features).
    dists = (
        jnp.sum(features_batch**2, axis=-1, keepdims=True)
        + jnp.sum(features**2, axis=-1)
        - 2.0 * jnp.matmul(features_batch, features.T)
    )  # shape (batch_size, pool_size)

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
    force = jnp.exp(-self.config.visibility * dists / self._n_features * 10.0)
    scaled_force = scaled_directions * force
    # Handle removed fireflies without updated rewards.
    finite_ind = jnp.isfinite(rewards).astype(scaled_force.dtype)

    # Ignore fireflies that were removed from the pool.
    scaled_force = scaled_force * finite_ind

    # Separate forces to pull and push so to normalize them separately.
    scaled_pulls = jnp.maximum(scaled_force, 0.0)
    scaled_push = jnp.minimum(scaled_force, 0.0)

    if self.config.mutate_normalization_type == MutateNormalizationType.MEAN:
      # Divide the push and pull forces by the number of flies participating.
      # Also multiply by normalization_scale.
      norm_scaled_pulls = self.config.normalization_scale * jnp.nan_to_num(
          scaled_pulls / jnp.sum(scaled_pulls > 0.0, axis=1, keepdims=True), 0
      )
      norm_scaled_push = self.config.normalization_scale * jnp.nan_to_num(
          scaled_push / jnp.sum(scaled_push < 0.0, axis=1, keepdims=True), 0
      )
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
    features_changes = jnp.matmul(scale, features) - features_batch * jnp.sum(
        scale, axis=-1, keepdims=True
    )
    return features_batch + features_changes

  def _create_random_perturbations(
      self, perturbations_batch: jax.Array, seed: chex.PRNGKey
  ) -> jax.Array:
    """Create random perturbations for the newly created batch.

    Args:
      perturbations_batch: (batch_size,)
      seed: Random seed.

    Returns:
      perturbations: (batch_size, n_features)
    """
    # Generate normalized noise for each batch.
    batch_noise = jax.random.laplace(
        seed, shape=(self.batch_size, self._n_features)
    )
    batch_noise /= jnp.max(jnp.abs(batch_noise), axis=1, keepdims=True)
    return (
        batch_noise
        * perturbations_batch[:, jnp.newaxis]
        * self._perturbation_factors
    )

  def update(
      self,
      state: VectorizedEagleStrategyState,
      batch_features: types.Array,
      batch_rewards: types.Array,
      seed: chex.PRNGKey,
  ) -> VectorizedEagleStrategyState:
    """Update the firefly pool based on the new batch of results.

    Arguments:
      state: Current state.
      batch_features: (batch_size, n_features)
      batch_rewards: (batch_size, )
      seed: Random seed.

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
              jax.lax.dynamic_slice_in_dim(
                  state.features, batch_start_ind, self.batch_size
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
        features=jax.lax.dynamic_update_slice_in_dim(
            state.features, new_batch_features, batch_start_ind, axis=0
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
      batch_features: jax.Array,
      batch_rewards: jax.Array,
      prev_batch_features: jax.Array,
      prev_batch_rewards: jax.Array,
      perturbations: jax.Array,
  ) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Update the features and rewards for flies with improved rewards.

    Arguments:
      batch_features: (batch_size, n_features), new proposed features batch.
      batch_rewards: (batch_size,), rewards for new proposed features batch.
      prev_batch_features: (batch_size, n_features), previous features batch.
      prev_batch_rewards: (batch_size,), rewards for previous features batch.
      perturbations: (batch_size,)

    Returns:
      sliced features, sliced rewards, sliced perturbations
    """
    # Find indices of flies that their generated features made an improvement.
    improve_indx = batch_rewards > prev_batch_rewards
    # Update successful flies' with the associated last features and rewards.
    new_batch_features = jnp.where(
        improve_indx[..., jnp.newaxis], batch_features, prev_batch_features
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
      batch_features: jax.Array,
      batch_rewards: jax.Array,
      batch_perturbations: jax.Array,
      best_reward: jax.Array,
      seed: chex.PRNGKey,
  ) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Trim the pool by replacing unsuccessful fireflies with new random ones.

    A firefly is considered unsuccessful if its current perturbation is below
    'perturbation_lower_bound' and it's not the best fly seen thus far.
    Random features are created to replace the existing ones, and rewards
    are set to -np.inf to indicate that we don't have values for those feaures
    yet and we shouldn't use them during suggest.

    Args:
      batch_features: (batch_size, n_features)
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
    random_features = self._param_handler.random_features(self.batch_size, seed)
    new_batch_features = jnp.where(
        indx[..., jnp.newaxis], random_features, batch_features
    )
    new_batch_perturbations = jnp.where(
        indx, self.config.perturbation, batch_perturbations
    )
    # Setting rewards to -inf to filter out those fireflies during suggest.
    new_batch_rewards = jnp.where(indx, -jnp.inf, batch_rewards)
    return new_batch_features, new_batch_rewards, new_batch_perturbations
