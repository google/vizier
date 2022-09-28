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
The firefly are stored in three Numpy arrays: features, metrics, perturbations.
Each iteration we mutate 'batch_size' fireflies to generate new features. The
new features are evaluated on the objective function to obtain their associated
metrics and update the pool where improvement was obtained, and decrease the
perturbation factor otherwise.

If the firefly's perturbation reaches the `perturbation_lower_bound` threshold
it's removed and replaced with a new random features.

For performance consideration, the 'pool size' is a multiplier of the
'batch size', and so each iteration the pool is sliced to obtain the current
fireflies to be mutated.

Example
=======
# Construct the vectorizd eagle strategy.
eagle_strategy = eagle_optimizer.VectorizedEagleStrategy(
        config=eagle_optimizer.EagleStrategyConfig(),
        n_features=n_features,
        low_bound=0.0,
        high_bound=1.0,
        pool_size=50,
        batch_size=10)

# Construct the optimizer.
optimizer = vectorized_base.NumpyOptimizer(
    strategy=eagle_strategy,
    config=vectorized_base.VectorizedOptimizerConfiguration()
)
# Run the optimization.
optimizer.optimize(objective_function)

# Access the best features and reward.
best_reward, best_parameters = optimizer.best_results
"""

import logging
from typing import Optional, Tuple

import attr
import numpy as np
from vizier._src.algorithms.optimizers import vectorized_base


@attr.define
# TODO: Perform self-tuning with GP aquisition function.
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
  visibility: float = 3.0
  gravity: float = 1.0
  negative_gravity: float = 0.02
  perturbation: float = 0.01
  perturbation_lower_bound: float = 0.001
  penalize_factor: float = 0.9


# TODO: Create jit-compatible JAX version.
@attr.define
class VectorizedEagleStrategy(vectorized_base.VectorizedStrategy):
  """Eagle strategy implementation based on Numpy.

  This implementation only supports continuous features with a linear scale type
  in the range of (low_bound, high_bound). The features within Eagle Strategy
  are in [0,1] and before returning the suggestion they are linearly mapped to
  original search space bounds.

  Attributes:
    config: The Eagle strategy configuration.
    n_features: The number of features.
    low_bound: The low bound of features in the search space.
    high_bound: The high bound of features in the search space.
    pool_size: The total number of flies in the pool.
    batch_size: The number of suggestions generated at each suggestion call.
    _features: Array with dimensions (suggestion_count, feature_count) storing
      the firefly features.
    _metrics: Array with dimensions (suggestion_count,) stroing the fireflies'
      best objective function results.
    _perturbations: Array with dimensions (suggestion_count,) stroing the
      firefly current perturbations.
    _batch_id: The current batch index which is in [0, pool_size/batch_size - 1]
    _batch_slice: The slice of the indices in the current batch.
    _iterations: The total number of batches suggested.
    _num_removed_flies: The total number of removed flies (for debugging).
  """
  config: EagleStrategyConfig = attr.field(init=True, repr=False)
  n_features: int = attr.field(init=True)
  pool_size: int = attr.field(init=True)
  batch_size: int = attr.field(init=True)
  low_bound: float = attr.field(init=True, default=0.0)
  high_bound: float = attr.field(init=True, default=1.0)
  seed: int = attr.field(init=True, default=42)
  # The following attributes constitute the strategy state.
  _features: np.ndarray = attr.field(init=False, repr=False)
  _metrics: np.ndarray = attr.field(init=False, repr=False)
  _perturbations: np.ndarray = attr.field(init=False, repr=False)
  _batch_id: int = attr.field(init=False, repr=False)
  _batch_slice: slice = attr.field(init=False, repr=False)
  _iterations: int = attr.field(init=False)
  _num_removed_flies: int = attr.field(init=False, default=0)
  _best_features: np.ndarray = attr.field(init=False, repr=False)
  _best_metric: Optional[float] = attr.field(init=False, default=None)
  _last_suggested_features: np.ndarray = attr.field(init=False, repr=False)

  def __attrs_post_init__(self):
    self._initialize()

  def _initialize(self):
    """Initialize the designer state."""
    self._rng = np.random.default_rng(self.seed)
    self._batch_id = 0
    self._iterations = 0
    self._batch_slice = np.s_[0:self.batch_size]
    self._features = self._random_features((self.pool_size, self.n_features))
    self._metrics = np.ones(self.pool_size,) * (-np.inf)
    self._perturbations = np.ones(self.pool_size,) * self.config.perturbation
    self._best_metric = None
    self._last_suggested_features = None

  @property
  def best_results(self) -> Tuple[np.ndarray, float]:
    """Returns the best features and metric the strategy seen thus far."""
    if self._best_metric is None:
      raise Exception("The strategy hasn't run yet!")
    return self._convert(self._best_features), self._best_metric

  def _random_features(self, size: Tuple[int, int]) -> np.ndarray:
    """Create random features with uniform distribution."""
    return self._rng.uniform(low=0.0, high=1.0, size=size)

  def _convert(self, features: np.ndarray) -> np.ndarray:
    """Linearly scale features to map into objective function's bounds."""
    return self.low_bound + (self.high_bound - self.low_bound) * features

  @property
  def suggestion_count(self) -> int:
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
      suggested_features = self._features[self._batch_slice]
    else:
      mutated_features = self._create_features()
      perturbations = self._create_perturbations()
      suggested_features = np.clip(mutated_features + perturbations, 0, 1)
    # Save the suggested features to be used in update.
    self._last_suggested_features = suggested_features
    return self._convert(suggested_features)

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

    Returns:
      scaled directions: (batch_size, pool_size)
    """
    shape = (self.batch_size,) + self._metrics.shape
    directions = np.broadcast_to(self._metrics, shape) - np.expand_dims(
        self._metrics[self._batch_slice], -1)
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
    pulls = np.exp(-self.config.visibility * dists / self.n_features * 10)
    scaled_pulls = np.expand_dims(scaled_directions * pulls, -1)
    # Handle removed fireflies without updated metrics.
    inf_indx = np.isinf(self._metrics)
    if np.sum(inf_indx) == self.pool_size:
      logging.warning(
          ("All firefly were recently removed. This Shouldn't happen."
           "Pool Features:\n%sPool Metrics:\n%s"), self._features,
          self._metrics)

      return np.zeros((self.batch_size, self.n_features))
    # Sums contributions of all non-outdated fireflies with invalid directions.
    return np.sum(
        features_diffs[:, ~inf_indx] * scaled_pulls[:, ~inf_indx], axis=1)

  def _create_perturbations(self) -> np.ndarray:
    """Create random perturbations for the newly creatd batch.

    Returns:
      batched perturbations: (base_size, n_features)
    """
    # Generate normalized noise for each batch.
    batch_noise = self._rng.laplace(size=(self.batch_size, self.n_features))
    batch_noise /= np.max(np.abs(batch_noise), axis=1, keepdims=True)
    # Scale the noise by the each fly current perturbation.
    return batch_noise * self._perturbations[self._batch_slice][:, np.newaxis]

  def update(self, batch_metrics: np.ndarray) -> None:
    """Update the firefly pool based on the new batch of results.

    Arguments:
      batch_metrics: (batch_size, )
    """
    self._update_best_result(batch_metrics)
    if self._iterations < self.pool_size // self.batch_size:
      # The strategy is still initializing. Assign metrics.
      self._features[self._batch_slice] = self._last_suggested_features
      self._metrics[self._batch_slice] = batch_metrics
    else:
      # Pass the new batch metrics and the associated last suggested features.
      self._update_pool_features_and_metrics(batch_metrics)
      self._trim_pool()
    self._increment_batch()
    self._iterations += 1

  def _update_best_result(self, batch_metrics: np.ndarray) -> None:
    """Update best results the strategy seen thus far."""
    best_ind = np.argmax(batch_metrics)
    if not self._best_metric or batch_metrics[best_ind] > self._best_metric:
      self._best_metric = batch_metrics[best_ind]
      self._best_features = self._last_suggested_features[best_ind]

  def _update_pool_features_and_metrics(
      self,
      batch_metrics: np.ndarray,
  ):
    """Update the features and metrics for flies with improved metrics.

    Arguments:
      batch_metrics: (batch_size, )
    """
    sliced_features = self._features[self._batch_slice]
    sliced_metrics = self._metrics[self._batch_slice]
    sliced_perturbations = self._perturbations[self._batch_slice]
    # Find indices of flies that their generated features made an improvement.
    improve_indx = batch_metrics > sliced_metrics
    # Update successful flies' with the associated last features and metrics.
    sliced_features[improve_indx] = self._last_suggested_features[improve_indx]
    sliced_metrics[improve_indx] = batch_metrics[improve_indx]
    # Penalize unsuccessful flies.
    sliced_perturbations[~improve_indx] *= self.config.penalize_factor

  def _trim_pool(self) -> None:
    """Trim the pool by replacing unsuccessful fireflies with new random ones.

      A firefly is considered unsuccessful if its current perturbation is below
      'perturbation_lower_bound' and it's not the best fly seen thus far.
      Random features are created to replace the existing ones, and metrics
      are set to -np.inf to indicate that we don't have values for those feaures
      yet and we shouldn't use them during suggest.
    """
    sliced_perturbations = self._perturbations[self._batch_slice]
    indx = sliced_perturbations < self.config.perturbation_lower_bound
    n_remove = np.sum(indx)
    if n_remove > 0:
      sliced_features = self._features[self._batch_slice]
      sliced_metrics = self._metrics[self._batch_slice]
      # Ensure the best firefly is never removed. For optimization purposes,
      # this logic is inside the if statement to be peformed only if needed.
      indx = indx & (sliced_metrics != self._best_metric)
      n_remove = np.sum(indx)
      if n_remove == 0:
        return
      # Replace fireflies with random features and evaluate metrics.
      sliced_features[indx] = self._random_features((n_remove, self.n_features))
      sliced_perturbations[indx] = np.ones(n_remove,) * self.config.perturbation
      # Setting metrics to -inf to filter out those fireflies during suggest.
      sliced_metrics[indx] = np.ones(n_remove,) * (-np.inf)
      self._num_removed_flies += n_remove
