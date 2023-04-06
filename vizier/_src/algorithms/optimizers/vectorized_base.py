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
import datetime
import logging
from typing import Generic, Optional, Protocol, Sequence, TypeVar

import attr
import chex
import jax
from jax import numpy as jnp
from vizier import pyvizier as vz
from vizier._src.jax import types
from vizier.pyvizier import converters

_S = TypeVar('_S')  # A container of optimizer state that works as a Pytree.


@chex.dataclass(frozen=True)
class VectorizedStrategyResults:
  """Container for a vectorized strategy result."""

  features: types.Array
  rewards: types.Array


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
      prior_features: Optional[types.Array] = None,
      prior_rewards: Optional[types.Array] = None,
  ) -> _S:
    """Initialize the state.

    Arguments:
      seed: Random seed for state initialization.
      prior_features: (n_prior_features, features_count)
      prior_rewards: (n_prior_features, )

    Returns:
      initial_state:
    """

  @abc.abstractmethod
  def suggest(self, state: _S, seed: jax.random.KeyArray) -> jax.Array:
    """Generate new suggestions.

    Arguments:
      state: Optimizer state.
      seed: Random seed.

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
      state: _S,
      batch_features: types.Array,
      batch_rewards: types.Array,
      seed: jax.random.KeyArray,
  ) -> _S:
    """Update the strategy state with the results of the last suggestions.

    Arguments:
      state: Optimizer state.
      batch_features: (batch_size, features_count)
      batch_rewards: (batch_size, )
      seed: Random seed.
    """


class VectorizedStrategyFactory(Protocol):
  """Factory class to generate vectorized strategy.

  It's used in VectorizedOptimizer to create a new strategy every 'optimize'
  call.
  """

  def __call__(
      self,
      converter: converters.TrialToArrayConverter,
      *,
      suggestion_batch_size: int,
  ) -> VectorizedStrategy:
    """Create a new vectorized strategy.

    Arguments:
      converter: The trial to array converter.
      suggestion_batch_size: The number of trials to be evaluated at once.
    """
    ...


class BatchArrayScoreFunction(Protocol):
  """Protocol for scoring array of batched trials."""

  def __call__(self, batched_array_trials: types.Array) -> types.Array:
    """Evaluates the array of batched trials.

    Arguments:
      batched_array_trials: 2D Array of shape (batch_size, n_features).

    Returns:
      1D Array of shape (batch_size,).
    """


@attr.define(kw_only=True)
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
    strategy_factory: A factory for generating new strategy.
    max_evaluations: The maximum number of objective function evaluations.
    max_duration: The maximum duration of the optimization process.
    suggestion_batch_size: The batch size of the suggestion vector received at
      each 'suggest' call.
    jit_loop: If True, JIT compile the optimization loop. If False, the loop
      body (an optimization step) will still be JIT compiled, but the outer loop
      will not be. (This arg is for speed testing and may be removed later.)
  """

  strategy_factory: VectorizedStrategyFactory
  suggestion_batch_size: int = 25
  max_evaluations: int = 75_000
  jit_loop: bool = True

  def optimize(
      self,
      converter: converters.TrialToArrayConverter,
      score_fn: BatchArrayScoreFunction,
      *,
      count: int = 1,
      prior_trials: Optional[Sequence[vz.Trial]] = None,
      seed: Optional[int] = None,
  ) -> list[vz.Trial]:
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
      converter: The converter used to convert Trials to arrays.
      score_fn: A callback that expects 2D Array with dimensions (batch_size,
        features_count) and returns a 1D Array (batch_size,).
      count: The number of suggestions to generate.
      prior_trials: Completed trials to be used for knowledge transfer. When the
        optimizer is used to optimize a designer's acquisition function, the
        prior trials are the previous designer suggestions provided in the
        ordered they were suggested.
      seed: The seed to use in the random generator.

    Returns:
      The best trials found in the optimization.
    """
    seed = jax.random.PRNGKey(seed or 0)
    if prior_trials:
      # Sort the trials by the order they were created.
      prior_trials = sorted(prior_trials, key=lambda x: x.creation_time)
      prior_features = converter.to_features(prior_trials)
      prior_rewards = score_fn(prior_features).reshape(-1)
    else:
      prior_features, prior_rewards = None, None

    strategy = self.strategy_factory(
        converter=converter,
        suggestion_batch_size=self.suggestion_batch_size,
    )

    def _optimization_one_step(_, args):
      state, best_results, seed = args
      suggest_seed, update_seed, new_seed = jax.random.split(seed, num=3)
      new_features = strategy.suggest(state, suggest_seed)
      new_rewards = score_fn(new_features)
      new_state = strategy.update(state, new_features, new_rewards, update_seed)
      new_best_results = self._update_best_results(
          best_results, count, new_features, new_rewards
      )
      return new_state, new_best_results, new_seed

    def _optimize():
      init_seed, loop_seed = jax.random.split(seed)
      n_features = sum(spec.num_dimensions for spec in converter.output_specs)
      init_best_results = VectorizedStrategyResults(
          rewards=-jnp.inf * jnp.ones([count]),
          features=jnp.zeros([count, n_features]),
      )
      init_args = (
          strategy.init_state(
              init_seed,
              prior_features=prior_features,
              prior_rewards=prior_rewards,
          ),
          init_best_results,
          loop_seed,
      )
      return jax.lax.fori_loop(
          0,
          self.max_evaluations // self.suggestion_batch_size,
          _optimization_one_step,
          init_args,
      )

    if self.jit_loop:
      _optimize = jax.jit(_optimize)  # pylint: disable=invalid-name

    start_time = datetime.datetime.now()
    _, best_results, _ = _optimize()
    logging.info(
        (
            'Optimization completed. Duration: %s. Evaluations: %s. Best'
            ' Results: %s'
        ),
        datetime.datetime.now() - start_time,
        (
            (self.max_evaluations // self.suggestion_batch_size)
            * self.suggestion_batch_size
        ),
        best_results,
    )
    return self._best_candidates(best_results, converter)

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
        dimension of (batch_size, feature_dim).
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

  def _best_candidates(
      self,
      best_results: VectorizedStrategyResults,
      converter: converters.TrialToArrayConverter,
  ) -> list[vz.Trial]:
    """Returns the best candidate trials in the original search space."""
    trials = []
    sorted_ind = jnp.argsort(-best_results.rewards)
    for i in range(len(best_results.rewards)):
      # Create trials and convert the strategy features back to parameters.
      ind = sorted_ind[i]
      trial = vz.Trial(
          parameters=converter.to_parameters(
              jnp.expand_dims(best_results.features[ind], axis=0)
          )[0]
      )
      trial.complete(vz.Measurement({'acquisition': best_results.rewards[ind]}))
      trials.append(trial)
    return trials


@attr.define
class VectorizedOptimizerFactory:
  """Vectorized strategy optimizer factory."""

  strategy_factory: VectorizedStrategyFactory

  def __call__(
      self,
      suggestion_batch_size: int,
      max_evaluations: int,
  ) -> VectorizedOptimizer:
    """Generates a new VectorizedOptimizer object."""
    return VectorizedOptimizer(
        strategy_factory=self.strategy_factory,
        suggestion_batch_size=suggestion_batch_size,
        max_evaluations=max_evaluations,
    )
