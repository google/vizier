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
from typing import Tuple, Optional

import attr
import numpy as np
from vizier._src.algorithms.optimizers import eagle_param_handler
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier.pyvizier import converters


@attr.define
class EagleStrategyConfig:
  """Eagle Strategy Optimizer Config.

  Attributes:
    visibility: The sensetivity to distance between flies when computing pulls.
    gravity: The maximum amount of attraction pull.
    negative_gravity: The maximum amount of repulsion pull.
    perturbation: The default amount of noise for perturbation.
    perturbation_lower_bound: The threshold below flies are removed from pool.
    penalize_factor: The perturbation decrease for unsuccessful flies.
  """
  # Visibility
  visibility: float = 3.0
  # Gravity
  gravity: float = 1.0
  negative_gravity: float = 0.02
  # Perturbation
  perturbation: float = 0.01
  categorical_perturbation_factor: float = 25
  pure_categorical_perturbation_factor: float = 30
  # Penalty
  perturbation_lower_bound: float = 0.001
  penalize_factor: float = 0.9
  pool_size: int = 50


@attr.define(frozen=True)
class VectorizedEagleStrategyFactory(vb.VectorizedStrategyFactory):
  """Eagle strategy factory."""
  eagle_config: EagleStrategyConfig = attr.field(factory=EagleStrategyConfig)

  def __call__(
      self,
      converter: converters.TrialToArrayConverter,
      suggestion_batch_size: int = 5,
      seed: Optional[int] = None,
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

    Returns:
      A new instance of VectorizedEagleStrategy.
    """
    return VectorizedEagleStrategy(
        converter=converter,
        config=self.eagle_config,
        batch_size=suggestion_batch_size,
        seed=seed,
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

  def _initialize(self):
    """Initialize the designer state."""
    self._rng = np.random.default_rng(self.seed)
    self._param_handler = eagle_param_handler.EagleParamHandler(
        converter=self.converter,
        rng=self._rng,
        categorical_perturbation_factor=self.config
        .categorical_perturbation_factor,
        pure_categorical_perturbation_factor=self.config
        .pure_categorical_perturbation_factor)
    self._n_features = self._param_handler.n_features
    self._batch_id = 0
    self._iterations = 0
    self._batch_slice = np.s_[0:self.batch_size]
    self.pool_size = self.config.pool_size
    self._features = self._param_handler.random_features(
        self.pool_size, self._n_features)
    self._rewards = np.ones(self.pool_size,) * (-np.inf)
    self._perturbations = np.ones(self.pool_size,) * self.config.perturbation
    self._last_suggested_features = None
    self._perturbation_factors = self._param_handler.perturbation_factors
    self._best_reward = -np.inf

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
      # The strategy is still initializing. Return the random features.
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
    features_changes = self._compute_features_changes(features_diffs, dists,
                                                      scaled_directions)
    return self._features[self._batch_slice] + features_changes

  def _compute_features_diffs_and_dists(self) -> Tuple[np.ndarray, np.ndarray]:
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
        self._features[self._batch_slice], 1)
    dists = np.sum(np.square(features_diffs), axis=-1)
    return features_diffs, dists

  def _compute_scaled_directions(self) -> np.ndarray:
    """Compute the scaled direction for applying pull between two flies.

    scaled_directions[i,j] := direction of force applied by fly 'j' on fly 'i'.

    Note that to compute 'directions' we might subtract with removed flies with
    reward value of -np.inf, which will result in either np.inf/-np.inf.
    Moreover, we might even subtract between two removed flies which will result
    in np.nan. We handle all of those cases when we compute the actual feautre
    changes by masking the contribution of those cases.

    Returns:
      scaled directions: (batch_size, pool_size)
    """
    shape = (self.batch_size,) + self._rewards.shape
    directions = np.broadcast_to(self._rewards, shape) - np.expand_dims(
        self._rewards[self._batch_slice], -1)
    scaled_directions = np.where(directions >= 0, self.config.gravity,
                                 -self.config.negative_gravity)
    return scaled_directions

  def _compute_features_changes(
      self,
      features_diffs: np.ndarray,
      dists: np.ndarray,
      scaled_directions: np.ndarray,
  ) -> np.ndarray:
    """Compute the firefly features changes due to mutation.

    Arguments:
      features_diffs: (batch_size, pool_size, n_features)
      dists: (batch_size, pool_size)
      scaled_directions: (batch_size, pool_size)

    Returns:
      feature changes: (batch_size, feature_n)
    """
    # Normalize the distance by the number of features.
    pulls = np.exp(-self.config.visibility * dists / self._n_features * 10)
    scaled_pulls = np.expand_dims(scaled_directions * pulls, -1)
    # Handle removed fireflies without updated rewards.
    inf_indx = np.isinf(self._rewards)
    if np.sum(inf_indx) == self.pool_size:
      logging.warning(
          ("All firefly were recently removed. This Shouldn't happen."
           "Pool Features:\n%sPool rewards:\n%s"), self._features,
          self._rewards)

      return np.zeros((self.batch_size, self._n_features))
    # Sums contributions of all non-outdated fireflies with invalid directions.
    return np.sum(
        features_diffs[:, ~inf_indx] * scaled_pulls[:, ~inf_indx], axis=1)

  def _create_perturbations(self) -> np.ndarray:
    """Create random perturbations for the newly creatd batch.

    Returns:
      batched perturbations: (base_size, n_features)
    """
    # Generate normalized noise for each batch.
    batch_noise = self._rng.laplace(size=(self.batch_size, self._n_features))
    batch_noise /= np.max(np.abs(batch_noise), axis=1, keepdims=True)
    # Scale the noise by the each fly current perturbation.
    return batch_noise * self._perturbations[
        self._batch_slice][:, np.newaxis] * self._perturbation_factors

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
          n_remove, self._n_features)
      sliced_perturbations[indx] = np.ones(n_remove,) * self.config.perturbation
      # Setting rewards to -inf to filter out those fireflies during suggest.
      sliced_rewards[indx] = np.ones(n_remove,) * (-np.inf)
      self._num_removed_flies += n_remove
