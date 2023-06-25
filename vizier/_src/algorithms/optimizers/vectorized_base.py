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

"""Base class for vectorized acquisition optimizers."""

import abc
import json
from typing import Callable, Generic, Optional, Protocol, TypeVar, Union

import attr
import equinox as eqx
from flax import struct
import jax
from jax import numpy as jnp
import numpy as np
from vizier import pyvizier as vz
from vizier._src.jax import types
from vizier.pyvizier import converters
from vizier.pyvizier.converters import feature_mapper
from vizier.utils import json_utils


_S = TypeVar('_S')  # A container of optimizer state that works as a Pytree.

ArrayConverter = Union[
    converters.TrialToArrayConverter, converters.PaddedTrialToArrayConverter
]


class VectorizedStrategyResults(eqx.Module):
  """Container for a vectorized strategy result."""

  features: types.Array  # (batch_size, n_parallel, n_features)
  rewards: types.Array  # (batch_size,)
  aux: dict[str, jax.Array] = eqx.field(default_factory=dict)


class VectorizedStrategy(abc.ABC, Generic[_S]):
  """Interface class to implement a pure vectorized strategy.

  The strategy is responsible for generating suggestions that will maximize the
  reward. The order of calls is important. It's expected to be used in
  'suggest','update', 'suggest', 'update', etc.
  """

  @abc.abstractmethod
  def init_state(
      self,
      seed: jax.random.KeyArray,
      n_parallel: int = 1,
      *,
      prior_features: Optional[types.Array] = None,
      prior_rewards: Optional[types.Array] = None,
  ) -> _S:
    """Initialize the state.

    Arguments:
      seed: Random seed for state initialization.
      n_parallel: Number of points that the acquisition function maps to a
        single value. This arg may be greater than 1 if a parallel acquisition
        function (qEI, qUCB) is used; otherwise it should be 1.
      prior_features: (n_prior_features, n_parallel, features_count)
      prior_rewards: (n_prior_features, )

    Returns:
      initial_state:
    """

  @abc.abstractmethod
  def suggest(
      self,
      seed: jax.random.KeyArray,
      state: _S,
      n_parallel: int = 1,
  ) -> jax.Array:
    """Generate new suggestions.

    Arguments:
      seed: Random seed.
      state: Optimizer state.
      n_parallel: Number of points that the acquisition function maps to a
        single value. This arg may be greater than 1 if a parallel acquisition
        function (qEI, qUCB) is used; otherwise it should be 1.

    Returns:
      suggested features: (batch_size, features_count)
    """

  @property
  @abc.abstractmethod
  def suggestion_batch_size(self) -> int:
    """The number of suggestions returned at every suggest call."""

  @abc.abstractmethod
  def update(
      self,
      seed: jax.random.KeyArray,
      state: _S,
      batch_features: types.Array,
      batch_rewards: types.Array,
  ) -> _S:
    """Update the strategy state with the results of the last suggestions.

    Arguments:
      seed: Random seed.
      state: Optimizer state.
      batch_features: (batch_size, n_parallel, features_count)
      batch_rewards: (batch_size, )
    """


class VectorizedStrategyFactory(Protocol):
  """Factory class to generate vectorized strategy.

  It's used in VectorizedOptimizer to create a new strategy every 'optimize'
  call.
  """

  def __call__(
      self,
      converter: ArrayConverter,
      *,
      suggestion_batch_size: int,
  ) -> VectorizedStrategy:
    """Create a new vectorized strategy.

    Arguments:
      converter: The trial to array converter.
      suggestion_batch_size: The number of trials to be evaluated at once.
    """
    ...


class ArrayScoreFunction(Protocol):
  """Protocol for scoring array of trials.

  This protocol is suitable for optimizing batch of candidates (each one with
  its own separate score).
  """

  def __call__(self, batched_array_trials: types.Array) -> types.Array:
    """Evaluates the array of batched trials.

    Arguments:
      batched_array_trials: Array of shape (batch_size, n_feature_dimensions).

    Returns:
      Array of shape (batch_size,).
    """


# TODO: Decide on consistent terminology for acquisition functions
# that evaluate sets of points.
class ParallelArrayScoreFunction(Protocol):
  """Protocol for scoring array of parallel trials.

  This protocol is suitable for optimizing in parallel multiple candidates
  (e.g. qUCB).
  """

  def __call__(self, parallel_array_trials: types.Array) -> types.Array:
    """Evaluates the array of batched trials.

    Arguments:
      parallel_array_trials: Array of shape (batch_size, n_parallel,
        n_feature_dimensions).

    Returns:
      Array of shape (batch_size).
    """


@struct.dataclass
class VectorizedOptimizer:
  """Vectorized strategy optimizer.

  The optimizer is stateless and will create a new vectorized strategy at the
  beginning of every 'optimize' call.

  The optimizer is responsible for running the iterative optimization process
  using the vectorized strategy, which consists of:
  1. Ask the strategy for suggestions.
  2. Evaluate the suggestions to get rewards.
  3. Tell the strategy about the rewards of its suggestion, so the strategy can
  update its internal state.

  The optimization process will terminate when the time limit or the total
  objective function evaluations limit has exceeded.

  Attributes:
    strategy: A factory for generating new strategy.
    n_feature_dimensions_with_padding: Number of feature dimensions including
      padding.
    n_feature_dimensions: Number of feature dimensions (less than or equal to
      `n_feature_dimensions_with_padding`).
    suggestion_batch_size: Number of suggested points returned at each call.
    max_evaluations: The maximum number of objective function evaluations.
    use_fori: Whether to use JAX's fori_loop in the suggest-evalute-update loop.
  """

  strategy: VectorizedStrategy
  n_feature_dimensions: int
  n_feature_dimensions_with_padding: int = struct.field(pytree_node=False)
  suggestion_batch_size: int = struct.field(pytree_node=False, default=25)
  max_evaluations: int = struct.field(pytree_node=False, default=75_000)
  use_fori: bool = struct.field(pytree_node=False, default=True)

  # TODO: Remove score_fn argument.
  # pylint: disable=g-bare-generic
  def __call__(
      self,
      score_fn: Union[ArrayScoreFunction, ParallelArrayScoreFunction],
      *,
      score_with_aux_fn: Optional[Callable] = None,
      count: int = 1,
      prior_features: Optional[types.Array] = None,
      n_parallel: Optional[int] = None,
      seed: Optional[int] = None,
  ) -> VectorizedStrategyResults:
    """Optimize the objective function.

    The ask-evaluate-tell optimization procedure that runs until the allocated
    time or evaluations count exceeds.

    The number of suggestions is determined by the strategy, which is the
    `suggestion_count` property.

    The converter should be the same one used to convert trials to arrays in the
    format that the objective function expects to receive. It's used to convert
    backward the strategy's best array candidates to trial candidates.
    In addition, the converter is passed to the strategy so it could be aware
    of the type associated with each array index, and which indices are part of
    the same CATEGORICAL parameter one-hot encoding.

    The optimization stops when either of 'max_evaluations' or 'max_duration' is
    reached.

    Arguments:
      score_fn: A callback that expects 2D Array with dimensions (batch_size,
        features_count) or a 3D array with dimensions (batch_size, n_parallel,
        features_count) and returns a 1D Array (batch_size,).
      score_with_aux_fn: A callback similar to score_fn but additionally returns
        an array tree.
      count: The number of suggestions to generate.
      prior_features: Completed trials to be used for knowledge transfer. When
        the optimizer is used to optimize a designer's acquisition function, the
        prior trials are the previous designer suggestions provided in the
        ordered they were suggested.
      n_parallel: This arg should be specified if a parallel acquisition
        function (e.g. qEI, qUCB) is used, which evaluates a set of points of
        this size to a single value. Pass `None` when optimizing acquisition
        functions that evaluate a single point. (Note that `num_parallel=1` and
        `num_parallel=None` will each return a single suggestion, though they
        are different in that `num_parallel=1` indicates that the acquisition
        function consumes the two rightmost dimensions of the input while
        `num_parallel=None` indicates that only the rightmost dimension is
        consumed.)
      seed: The seed to use in the random generator.

    Returns:
      An array containing the best trials found in the optimization of shape
      (count, n_parallel or 1, n_feature_dimensions).
    """
    seed = jax.random.PRNGKey(0) if seed is None else seed

    dimension_is_missing = (
        jnp.arange(self.n_feature_dimensions_with_padding)
        > self.n_feature_dimensions
    )

    if n_parallel is None:
      # Squeeze out the singleton dimension of `features` before passing to a
      # non-parallel acquisition function to avoid batch shape collisions.
      eval_score_fn = lambda x: score_fn(x[:, 0, :])
    else:
      eval_score_fn = score_fn

    # TODO: We should pass RNGKey to score_fn.
    prior_rewards = None
    parallel_dim = n_parallel or 1
    if prior_features is not None:
      num_prior_obs = prior_features.shape[0]
      num_prior_batches = num_prior_obs // parallel_dim
      prior_features = jnp.reshape(
          prior_features[: num_prior_batches * parallel_dim],
          [num_prior_batches, parallel_dim, -1],
      )
      prior_rewards = eval_score_fn(prior_features)

    def _optimization_one_step(_, args):
      state, best_results, seed = args
      suggest_seed, update_seed, new_seed = jax.random.split(seed, num=3)
      new_features = self.strategy.suggest(
          suggest_seed, state=state, n_parallel=parallel_dim
      )
      # Ensure masking out padded dimensions in new features.
      new_features = jnp.where(
          dimension_is_missing, jnp.zeros_like(new_features), new_features
      )
      # We assume `score_fn` is aware of padded dimensions.
      new_rewards = eval_score_fn(new_features)
      new_state = self.strategy.update(
          update_seed, state, new_features, new_rewards
      )
      new_best_results = self._update_best_results(
          best_results, count, new_features, new_rewards
      )
      return new_state, new_best_results, new_seed

    init_seed, loop_seed = jax.random.split(seed)
    init_best_results = VectorizedStrategyResults(
        rewards=-jnp.inf * jnp.ones([count]),
        features=jnp.zeros(
            [count, parallel_dim, self.n_feature_dimensions_with_padding]
        ),
    )
    init_args = (
        self.strategy.init_state(
            init_seed,
            n_parallel=parallel_dim,
            prior_features=prior_features,
            prior_rewards=prior_rewards,
        ),
        init_best_results,
        loop_seed,
    )
    if self.use_fori:
      _, best_results, _ = jax.lax.fori_loop(
          0,
          (self.max_evaluations - 1) // self.suggestion_batch_size + 1,
          _optimization_one_step,
          init_args,
      )
    else:
      args = init_args
      for i in range(
          (self.max_evaluations - 1) // self.suggestion_batch_size + 1
      ):
        args = _optimization_one_step(i, args)
      _, best_results, _ = args

    if score_with_aux_fn:
      if n_parallel is None:
        aux = score_with_aux_fn(best_results.features[:, 0, :])[1]
      else:
        aux = score_with_aux_fn(best_results.features)[1]

      return VectorizedStrategyResults(
          best_results.features,
          best_results.rewards,
          aux,
      )

    return best_results

  def _update_best_results(
      self,
      best_results: VectorizedStrategyResults,
      count: int,
      batch_features: jax.Array,
      batch_rewards: jax.Array,
  ) -> VectorizedStrategyResults:
    """Update the best results the optimizer seen thus far.

    The best results are kept in a heap to efficiently maintain the top 'count'
    results throughout the optimizer run.

    Arguments:
      best_results: A heap storing the best results seen thus far. Implemented
        as a list of maximum size of 'count'.
      count: The number of best results to store.
      batch_features: The current suggested features batch array with a
        dimension of (batch_size, feature_dim) or (batch_size, n_parallel,
        feature_dim).
      batch_rewards: The current reward batch array with a dimension of
        (batch_size,).

    Returns:
      trials:
    """
    all_rewards = jnp.concatenate([batch_rewards, best_results.rewards], axis=0)
    all_features = jnp.concatenate(
        [batch_features, best_results.features], axis=0
    )
    top_indices = jnp.argpartition(-all_rewards, count - 1)[:count]
    return VectorizedStrategyResults(
        rewards=all_rewards[top_indices],
        features=all_features[top_indices],
    )


# TODO: Should return suggestions not trials.
def best_candidates_to_trials(
    best_results: VectorizedStrategyResults,
    converter: ArrayConverter,
) -> list[vz.Trial]:
  """Returns the best candidate trials in the original search space."""
  trials = []
  sorted_ind = jnp.argsort(-best_results.rewards)
  features = best_results.features
  if isinstance(features, types.ContinuousAndCategorical):
    features = feature_mapper.ContinuousCategoricalFeatureMapper(
        converter
    ).unmap(features)
  for i in range(len(best_results.rewards)):
    # Create trials and convert the strategy features back to parameters.
    ind = sorted_ind[i]
    suggested_features = features[ind]
    reward = best_results.rewards[ind]

    # Loop over the number of candidates per batch (which will be one, unless a
    # parallel acquisition function is used).
    for j in range(suggested_features.shape[0]):
      trial = vz.Trial(
          parameters=converter.to_parameters(
              jnp.expand_dims(suggested_features[j], axis=0)
          )[0]
      )

      metadata = trial.metadata.ns('devinfo')
      metadata['acquisition_optimization'] = json.dumps(
          {'acquisition': best_results.rewards[ind]}
          | jax.tree_map(
              lambda x, ind=ind: np.asarray(x[ind]), best_results.aux
          ),
          cls=json_utils.NumpyEncoder,
      )

      trial.complete(vz.Measurement({'acquisition': reward}))
      trials.append(trial)
  return trials


# TODO: This function should return jax types.
def trials_to_sorted_array(
    prior_trials: list[vz.Trial],
    converter: ArrayConverter,
) -> Optional[types.Array]:
  """Sorts trials by the order they were created and converts to array."""
  if prior_trials:
    prior_trials = sorted(prior_trials, key=lambda x: x.creation_time)
    prior_features = converter.to_features(prior_trials)
    # TODO: Update this code to work more cleanly with
    # PaddedArrays.
    if isinstance(converter, converters.PaddedTrialToArrayConverter):
      # We need to mask out the `NaN` padded trials with zeroes.
      prior_features = np.array(prior_features.padded_array)
      prior_features[len(prior_trials) :, ...] = 0.0
  else:
    prior_features = None
  return prior_features


@attr.define
class VectorizedOptimizerFactory:
  """Vectorized strategy optimizer factory."""

  strategy_factory: VectorizedStrategyFactory
  max_evaluations: int = 75_000
  suggestion_batch_size: int = 25
  use_fori: bool = True

  def __call__(
      self,
      converter: ArrayConverter,
  ) -> VectorizedOptimizer:
    """Generates a new VectorizedOptimizer object."""
    strategy = self.strategy_factory(
        converter, suggestion_batch_size=self.suggestion_batch_size
    )
    n_feature_dimensions = sum(
        spec.num_dimensions for spec in converter.output_specs
    )
    n_feature_dimensions_with_padding = converter.to_features([]).shape[-1]
    return VectorizedOptimizer(
        strategy=strategy,
        n_feature_dimensions=n_feature_dimensions,
        n_feature_dimensions_with_padding=n_feature_dimensions_with_padding,
        suggestion_batch_size=self.suggestion_batch_size,
        max_evaluations=self.max_evaluations,
        use_fori=self.use_fori,
    )
