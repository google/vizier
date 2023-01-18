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
The firefly are stored in three Numpy arrays: features, rewards, perturbations.
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
optimizer.optimize(problem_statement, objective_function)

# Access the best features and reward.
best_reward, best_parameters = optimizer.best_results
"""

import logging
from typing import Optional, Literal

import attr
import numpy as np
from vizier._src.algorithms.optimizers import eagle_param_handler
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier.pyvizier import converters


@attr.define
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
  # Force normalization mode
  mutate_normalization_type: str = "mean"
  # Multiplier factor when using normalized modes.
  normalization_scale: float = 0.5
  # The percentage of the firefly pool to be populated with prior trials.
  prior_trials_pool_pct: float = 0.96


@attr.define(frozen=True)
class VectorizedEagleStrategyFactory(vb.VectorizedStrategyFactory):
  """Eagle strategy factory."""

  eagle_config: EagleStrategyConfig = attr.field(factory=EagleStrategyConfig)

  def __call__(
      self,
      converter: converters.TrialToArrayConverter,
      suggestion_batch_size: int = 5,
      seed: Optional[int] = None,
      prior_features: Optional[np.ndarray] = None,
      prior_rewards: Optional[np.ndarray] = None,
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
      seed: The seed to input into the random generator.
      prior_features: The prior features for populating the pool.
        (n_prior_candidates, n_features)
      prior_rewards:  The prior rewards for populating the pool.
        (n_prior_candidates,)

    Returns:
      A new instance of VectorizedEagleStrategy.
    """
    return VectorizedEagleStrategy(
        converter=converter,
        config=self.eagle_config,
        batch_size=suggestion_batch_size,
        seed=seed,
        prior_features=prior_features,
        prior_rewards=prior_rewards,
    )


# TODO: Create jit-compatible JAX version.
@attr.define
class VectorizedEagleStrategy(vb.VectorizedStrategy):
  """Eagle strategy implementation for maximization problem based on Numpy.

  Attributes:
    converter: The converter used for the optimization problem.
    config: The Eagle strategy configuration.
    n_features: The number of features.
    batch_size: The number of suggestions generated at each suggestion call.
    prior_features: The prior features to be used for seeding the pool. When the
      optimizer is used to optimize a designer's acquisition function, the prior
      features are the previous designer suggestions provided in the ordered
      they were suggested. Shape: (n_prior_trials, n_features).
    prior_rewards: The associated prior rewards of the prior features.
      Shape: (n_prior_trials,)
    seed: The seed to generate random values.
    pool_size: The total number of flies in the pool.
    _features: Array with dimensions (suggestion_count, feature_count) storing
      the firefly features.
    _rewards: Array with dimensions (suggestion_count,) that for each firefly
      stores the current (best) associated objective function result.
    _perturbations: Array with dimensions (suggestion_count,) storing the
      firefly current perturbations.
    _batch_id: The current batch index which is in [0, pool_size/batch_size - 1]
    _batch_slice: The slice of the indices in the current batch.
    _iterations: The total number of batches suggested.
    _num_removed_flies: The total number of removed flies (for debugging).
    _best_results: An Heap storing the best results across all flies.
    _last_suggested_features: The last features the strategy has suggested.
  """

  converter: converters.TrialToArrayConverter
  config: EagleStrategyConfig = attr.field(init=True, repr=False)
  batch_size: int = attr.field(init=True)
  seed: Optional[int] = attr.field(init=True)
  pool_size: int = attr.field(init=False)
  prior_features: Optional[np.ndarray] = attr.field(init=True, default=None)
  prior_rewards: Optional[np.ndarray] = attr.field(init=True, default=None)
  # Attributes related to the strategy's state.
  _features: np.ndarray = attr.field(init=False, repr=False)
  _rewards: np.ndarray = attr.field(init=False, repr=False)
  _perturbations: np.ndarray = attr.field(init=False, repr=False)
  _batch_id: int = attr.field(init=False, repr=False)
  _batch_slice: slice = attr.field(init=False, repr=False)
  _iterations: int = attr.field(init=False)
  _last_suggested_features: np.ndarray = attr.field(init=False, repr=False)
  _best_reward: float = attr.field(init=False)
  # Attributes related to computations.
  _num_removed_flies: int = attr.field(init=False, default=0)
  _n_features: int = attr.field(init=False)
  _perturbation_factors: np.ndarray = attr.field(init=False, repr=False)

  def __attrs_post_init__(self):
    self._initialize()
    logging.info("Eagle class attributes:\n%s", self)
    logging.info("Eagle configuration:\n%s", self.config)

  def _compute_pool_size(self):
    """Compute the pool size, and ensures it's a multiply of the batch_size."""
    raw_pool_size = 10 + int(
        0.5 * self._n_features
        + self._n_features**self.config.pool_size_exponent
    )
    return int(np.ceil(raw_pool_size / self.batch_size) * self.batch_size)

  def _initialize(self):
    """Initialize the designer state."""
    self._rng = np.random.default_rng(self.seed)
    self._param_handler = eagle_param_handler.EagleParamHandler(
        converter=self.converter,
        rng=self._rng,
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
    if self.batch_size == -1:
      # This configuration updates all the fireflies in each iteration.
      self.batch_size = self.pool_size
    self._batch_id = 0
    self._iterations = 0
    self._batch_slice = np.s_[0 : self.batch_size]
    self._perturbations = (
        np.ones(
            self.pool_size,
        )
        * self.config.perturbation
    )
    self._last_suggested_features = None
    self._perturbation_factors = self._param_handler.perturbation_factors
    # Use priors to populate Eagle state
    if self.prior_features is not None or self.prior_rewards is not None:
      self._populate_pool_with_prior_trials()
    else:
      self._features = self._param_handler.random_features(
          self.pool_size, self._n_features
      )
    # Rewards are not populated from seed, as they were potentially generated
    # from a different objective function. New rewards will be obtained.
    self._rewards = np.ones(
        self.pool_size,
    ) * (-np.inf)
    self._best_reward = -np.inf
    # Ignore subtracting np.inf - np.inf. See '_compute_scaled_directions' for
    # explanation on why we ignore warning in this case.
    np.seterr(invalid="ignore")

  def _populate_pool_with_prior_trials(self) -> None:
    """Populate the pool with prior trials.

    A portion of the pool is first populated with random features based on
    'prior_trials_pool_pct', then the rest of the flies are populated by
    sequentially iterate over the prior trials, finding the cloest firefly in
    the pool and replace it if the reward is better.
    """
    if self.prior_features is None or self.prior_rewards is None:
      raise ValueError("One of prior features / prior rewards wasn't provided!")
    if self.prior_features.shape[0] != self.prior_rewards.shape[0]:
      raise ValueError(
          f"prior features shape ({self.prior_features.shape[0]}) doesn't match"
          f" prior  rewards shape ({self.prior_rewards.shape[0]})!"
      )
    if self.prior_features.shape[1] != self._n_features:
      raise ValueError(
          "prior features shape doesn't match n_features{self._n_features}!"
      )
    if len(self.prior_rewards.shape) > 1:
      raise ValueError("prior rewards is expected to be 1D array!")

    # Reverse the order of prior trials to assign more weight to recent trials.
    self.prior_features = np.flip(self.prior_features, axis=-1)
    self.prior_rewards = np.flip(self.prior_rewards, axis=-1)

    self._features = np.zeros((0, self._n_features))
    # Fill pool with random features.
    n_random_flies = int(
        self.pool_size * (1 - self.config.prior_trials_pool_pct)
    )
    self._features = self._param_handler.random_features(
        n_random_flies, self._n_features
    )
    pool_left_space = self.pool_size - n_random_flies

    if self.prior_features.shape[0] < pool_left_space:
      # Less prior trials than left space. Take all prior trials for the pool.
      self._features = np.concatenate([self._features, self.prior_features])
      # Randomize the rest of the pool fireflies.
      random_features = self._param_handler.random_features(
          self.pool_size - len(self._features), self._n_features
      )
      self._features = np.concatenate([self._features, random_features])
    else:
      # More prior trials than left space. Iteratively populate the pool.
      tmp_features = self.prior_features[:pool_left_space]
      tmp_rewards = self.prior_rewards[:pool_left_space]
      for i in range(pool_left_space, self.prior_features.shape[0]):
        ind = np.argmin(
            np.sum(np.square(self.prior_features[i] - tmp_features), axis=-1)
        )
        if tmp_rewards[ind] < self.prior_rewards[i]:
          # Only take the prior trials features. Rewards obtain during update.
          tmp_features[ind] = self.prior_features[i]
          tmp_rewards[ind] = self.prior_rewards[i]
      self._features = np.concatenate([self._features, tmp_features])

  @property
  def suggestion_batch_size(self) -> int:
    """The number of suggestions returned at each call of 'suggest'."""
    return self.batch_size

  def suggest(self) -> np.ndarray:
    """Suggest new mutated and perturbed features.

    After initializing, at each call `batch_size` fireflies are mutated to
    generate new features using pulls (attraction/repulsion) from all other
    fireflies in the pool.

    Returns:
      suggested batch features: (batch_size, n_features)
    """
    if self._iterations < self.pool_size // self.batch_size:
      # The strategy is still initializing. Return the random/prior features.
      new_features = self._features[self._batch_slice]
    else:
      mutated_features = self._create_features()
      perturbations = self._create_perturbations()
      new_features = mutated_features + perturbations

    new_features = self._param_handler.sample_categorical(new_features)
    suggested_features = np.clip(new_features, 0, 1)
    # Save the suggested features to be used in update.
    self._last_suggested_features = suggested_features
    return suggested_features

  def _increment_batch(self):
    """Increment the batch of fireflies features are generate from."""
    self._batch_id = (self._batch_id + 1) % (self.pool_size // self.batch_size)
    start_batch = self._batch_id * self.batch_size
    end_batch = (self._batch_id + 1) * self.batch_size
    self._batch_slice = np.s_[start_batch:end_batch]

  def _create_features(self) -> np.ndarray:
    """Create new batch of mutated and perturbed features.

    Returns:
      batch features: (batch_size, n_features)
    """
    features_diffs, dists = self._compute_features_diffs_and_dists()
    scaled_directions = self._compute_scaled_directions()
    features_changes = self._compute_features_changes(
        features_diffs, dists, scaled_directions
    )
    return self._features[self._batch_slice] + features_changes

  def _compute_features_diffs_and_dists(self) -> tuple[np.ndarray, np.ndarray]:
    """Compute the features difference and distances.

    The computation is done between the 'batch_size' fireflies and all
    other fireflies in the pool.

    features_diff[i, j, :] := features[j, :] - features[i, :]
    features_dist[i, j, :] := distance between fly 'j' and fly 'i'

    Returns:
      feature differences: (batch_size, pool_size, n_features)
      features distances: (batch_size, pool_size)
    """
    shape = (self.batch_size,) + self._features.shape
    features_diffs = np.broadcast_to(self._features, shape) - np.expand_dims(
        self._features[self._batch_slice], 1
    )
    dists = np.sum(np.square(features_diffs), axis=-1)
    return features_diffs, dists

  def _compute_scaled_directions(self) -> np.ndarray:
    """Compute the scaled direction for applying pull between two flies.

    scaled_directions[i,j] := direction of force applied by fly 'j' on fly 'i'.

    Note that to compute 'directions' we might perform subtract with removed
    flies with having value of -np.inf. Moreover, we might even subtract between
    two removed flies which will result in np.nan. Both cases are handled when
    computing the actual feautre changes applying a relevant mask.

    Returns:
      scaled directions: (batch_size, pool_size)
    """
    shape = (self.batch_size,) + self._rewards.shape
    directions = np.broadcast_to(self._rewards, shape) - np.expand_dims(
        self._rewards[self._batch_slice], -1
    )

    scaled_directions = np.where(
        directions >= 0, self.config.gravity, -self.config.negative_gravity
    )
    return scaled_directions

  def _compute_features_changes(
      self,
      features_diffs: np.ndarray,
      dists: np.ndarray,
      scaled_directions: np.ndarray,
  ) -> np.ndarray:
    """Compute the firefly features changes due to mutation.

    The pool fireflies forces (pull/push) are being normalized to ensure the
    combined force doesn't throw the firefly too far. Mathematically, the
    normalization guarantees that the combined normalized force is within the
    simplex constructed by the unnormalized forces and therefore within bounds.

    Arguments:
      features_diffs: (batch_size, pool_size, n_features)
      dists: (batch_size, pool_size)
      scaled_directions: (batch_size, pool_size)

    Returns:
      feature changes: (batch_size, feature_n)
    """
    # Normalize the distance by the number of features.
    force = np.exp(-self.config.visibility * dists / self._n_features * 10)
    scaled_force = np.expand_dims(scaled_directions * force, -1)
    # Handle removed fireflies without updated rewards.
    inf_indx = np.isinf(self._rewards)
    if np.sum(inf_indx) == self.pool_size:
      logging.warning(
          (
              "All firefly were recently removed. This Shouldn't happen."
              "Pool Features:\n%sPool rewards:\n%s"
          ),
          self._features,
          self._rewards,
      )
      return np.zeros((self.batch_size, self._n_features))

    # Ignore fireflies that were removed from the pool.
    features_diffs = features_diffs[:, ~inf_indx, :]
    scaled_force = scaled_force[:, ~inf_indx, :]

    # Separate forces to pull and push so to normalize them separately.
    scaled_pulls = np.where(scaled_force > 0, scaled_force, 0)
    scaled_push = np.where(scaled_force < 0, scaled_force, 0)

    if self.config.mutate_normalization_type == "mean":
      # Divide the push and pull forces by the number of flies participating.
      # Also multiply by normalization_scale.
      norm_scaled_pulls = self.config.normalization_scale * np.nan_to_num(
          scaled_pulls / np.sum(scaled_pulls > 0, axis=1, keepdims=True), 0
      )
      norm_scaled_push = self.config.normalization_scale * np.nan_to_num(
          scaled_push / np.sum(scaled_push < 0, axis=1, keepdims=True), 0
      )
    elif self.config.mutate_normalization_type == "random":
      # Create random matrices and normalize each row, s.t. the sum is 1.
      pull_rand_matrix = self._rng.uniform(
          0, 1, size=scaled_pulls.shape
      ) * np.int_(scaled_pulls > 0)
      pull_weight_matrix = pull_rand_matrix / np.sum(
          pull_rand_matrix, axis=1, keepdims=True
      )
      push_rand_matrix = self._rng.uniform(
          0, 1, size=scaled_pulls.shape
      ) * np.int_(scaled_pulls > 0)
      push_weight_matrix = push_rand_matrix / np.sum(
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
    elif self.config.mutate_normalization_type == "unnormalized":
      # Doesn't normalize the forces. Use this option with caution.
      norm_scaled_pulls = scaled_pulls
      norm_scaled_push = scaled_push

    # Sums normalized forces (pull/push) of all fireflies.
    return np.sum(
        features_diffs * (norm_scaled_pulls + norm_scaled_push), axis=1
    )

  def _create_perturbations(self) -> np.ndarray:
    """Create random perturbations for the newly creatd batch.

    Returns:
      batched perturbations: (base_size, n_features)
    """
    # Generate normalized noise for each batch.
    batch_noise = self._rng.laplace(size=(self.batch_size, self._n_features))
    batch_noise /= np.max(np.abs(batch_noise), axis=1, keepdims=True)
    # Scale the noise by the each fly current perturbation.
    return (
        batch_noise
        * self._perturbations[self._batch_slice][:, np.newaxis]
        * self._perturbation_factors
    )

  def update(self, batch_rewards: np.ndarray) -> None:
    """Update the firefly pool based on the new batch of results.

    Arguments:
      batch_rewards: (batch_size, )
    """
    self._update_best_reward(batch_rewards)
    if self._iterations < self.pool_size // self.batch_size:
      # The strategy is still initializing. Assign rewards.
      self._features[self._batch_slice] = self._last_suggested_features
      self._rewards[self._batch_slice] = batch_rewards
    else:
      # Pass the new batch rewards and the associated last suggested features.
      self._update_pool_features_and_rewards(batch_rewards)
      self._trim_pool()
    self._increment_batch()
    self._iterations += 1

  def _update_best_reward(self, batch_rewards: np.ndarray) -> None:
    """Store the best result seen thus far to be used in pool trimming."""
    self._best_reward = np.max([self._best_reward, np.max(batch_rewards)])

  def _update_pool_features_and_rewards(
      self,
      batch_rewards: np.ndarray,
  ):
    """Update the features and rewards for flies with improved rewards.

    Arguments:
      batch_rewards: (batch_size, )
    """
    sliced_features = self._features[self._batch_slice]
    sliced_rewards = self._rewards[self._batch_slice]
    sliced_perturbations = self._perturbations[self._batch_slice]
    # Find indices of flies that their generated features made an improvement.
    improve_indx = batch_rewards > sliced_rewards
    # Update successful flies' with the associated last features and rewards.
    sliced_features[improve_indx] = self._last_suggested_features[improve_indx]
    sliced_rewards[improve_indx] = batch_rewards[improve_indx]
    # Penalize unsuccessful flies.
    sliced_perturbations[~improve_indx] *= self.config.penalize_factor

  def _trim_pool(self) -> None:
    """Trim the pool by replacing unsuccessful fireflies with new random ones.

    A firefly is considered unsuccessful if its current perturbation is below
    'perturbation_lower_bound' and it's not the best fly seen thus far.
    Random features are created to replace the existing ones, and rewards
    are set to -np.inf to indicate that we don't have values for those feaures
    yet and we shouldn't use them during suggest.
    """
    sliced_perturbations = self._perturbations[self._batch_slice]
    indx = sliced_perturbations < self.config.perturbation_lower_bound
    n_remove = np.sum(indx)
    if n_remove > 0:
      sliced_features = self._features[self._batch_slice]
      sliced_rewards = self._rewards[self._batch_slice]
      # Ensure the best firefly is never removed. For optimization purposes,
      # this logic is inside the if statement to be peformed only if needed.
      indx = indx & (sliced_rewards != self._best_reward)
      n_remove = np.sum(indx)
      if n_remove == 0:
        return
      # Replace fireflies with random features and evaluate rewards.
      sliced_features[indx] = self._param_handler.random_features(
          n_remove, self._n_features
      )
      sliced_perturbations[indx] = (
          np.ones(
              n_remove,
          )
          * self.config.perturbation
      )
      # Setting rewards to -inf to filter out those fireflies during suggest.
      sliced_rewards[indx] = np.ones(
          n_remove,
      ) * (-np.inf)
      self._num_removed_flies += n_remove
