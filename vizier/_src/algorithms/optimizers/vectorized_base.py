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

"""Base class for vectorized acquisition optimizers."""

import abc
import datetime
import heapq
import logging
from typing import Optional, Protocol, Union

import attr
import chex
import numpy as np
from vizier import pyvizier as vz
from vizier.pyvizier import converters

# Support for both JAX and Numpy arrays
Array = Union[np.ndarray, chex.Array]


@attr.define
class VectorizedStrategyResult:
  """Container for a vectorized strategy result."""
  features: Array
  reward: float

  def __lt__(self, other):
    return self.reward < other.reward


class VectorizedStrategy(abc.ABC):
  """Interface class to implement a pure vectorized strategy.

  The strategy is responsible for generating suggestions that will maximize the
  reward. The order of calls is important. It's expected to be used in
  'suggest','update', 'suggest', 'update', etc.
  """

  @abc.abstractmethod
  def suggest(self) -> Array:
    """Generate new suggestions.

    Returns:
      suggested features: (batch_size, features_count)
    """

  @property
  @abc.abstractmethod
  def suggestion_batch_size(self) -> int:
    """The number of suggestions returned at every suggest call."""

  @abc.abstractmethod
  def update(self, rewards: Array) -> None:
    """Update the strategy state with the results of the last suggestions.

    Arguments:
      rewards: (batch_size, )
    """


class VectorizedStrategyFactory(Protocol):
  """Factory class to generate vectorized strategy.

  It's used in VectorizedOptimizer to create a new strategy every 'optimize'
  call.
  """

  def __call__(self,
               converter: converters.TrialToArrayConverter,
               suggestion_batch_size: int,
               seed: Optional[int] = None) -> VectorizedStrategy:
    """Create a new vectorized strategy."""
    ...


class BatchArrayScoreFunction(Protocol):
  """Protocol for scoring array of batched trials."""

  def __call__(self, batched_array_trials: Array) -> Array:
    """Evaluates the array of batched trials.

    Args:
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
  """
  strategy_factory: VectorizedStrategyFactory
  suggestion_batch_size: int = 5
  max_evaluations: int = 15_000
  max_duration: Optional[datetime.timedelta] = None

  def optimize(
      self,
      converter: converters.TrialToArrayConverter,
      score_fn: BatchArrayScoreFunction,
      count: int = 1,
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
      seed: The seed to use in the random generator.

    Returns:
      The best trials found in the optimization.
    """
    strategy = self.strategy_factory(converter, self.suggestion_batch_size,
                                     seed)
    start_time = datetime.datetime.now()
    evaluated_count = 0
    best_results = []

    while not self._should_stop(start_time, evaluated_count):
      new_features = strategy.suggest()
      new_rewards = score_fn(new_features)
      strategy.update(new_rewards)
      self._update_best_results(best_results, count, new_features, new_rewards)
      evaluated_count += len(new_rewards)
    logging.info(
        'Optimization completed. Duration: %s. Evaluations: %s. Best Results: %s',
        datetime.datetime.now() - start_time, evaluated_count, best_results)

    return self._best_candidates(best_results, converter)

  def _update_best_results(
      self,
      best_results: list[VectorizedStrategyResult],
      count: int,
      batch_features: np.ndarray,
      batch_rewards: np.ndarray,
  ) -> None:
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
    """
    sorted_indices = sorted(
        range(len(batch_rewards)), key=lambda x: batch_rewards[x],
        reverse=True)[:count]

    for ind in sorted_indices:
      if len(best_results) < count:
        # 'best_results' is below capacity, add the best result from the batch.
        new_result = VectorizedStrategyResult(
            reward=batch_rewards[ind], features=batch_features[ind])
        heapq.heappush(best_results, new_result)

      elif batch_rewards[ind] > best_results[0].reward:
        # 'best_result' is at capacity and the best batch result is better than
        # the worst item in 'best_results'.
        new_result = VectorizedStrategyResult(
            reward=batch_rewards[ind], features=batch_features[ind])
        heapq.heapreplace(best_results, new_result)

  def _best_candidates(
      self,
      best_results: list[VectorizedStrategyResult],
      converter: converters.TrialToArrayConverter,
  ) -> list[vz.Trial]:
    """Returns the best candidate trials in the original search space."""
    trials = []
    for best_result in sorted(best_results, reverse=True):
      # Create trials and convert the strategy features back to parameters.
      trial = vz.Trial(
          parameters=converter.to_parameters(
              np.expand_dims(best_result.features, axis=0))[0])
      trial.complete(vz.Measurement({'acquisition': best_result.reward}))
      trials.append(trial)
    return trials

  def _should_stop(self, start_time: datetime.datetime,
                   evaluated_count: int) -> bool:
    """Determines if the optimizer has reached its optimization budget."""
    duration = datetime.datetime.now() - start_time
    if self.max_duration and duration >= self.max_duration:
      logging.info(
          'Optimization completed. Reached time limit. Duration: %s. Evaluations: %s',
          duration, evaluated_count)
      return True
    elif evaluated_count >= self.max_evaluations:
      logging.info(
          'Optimization completed. Reached evaluations limit. Duration: %s. Evaluations: %s',
          duration, evaluated_count)
      return True
    else:
      return False


@attr.define
class VectorizedOptimizerFactory:
  """Vectorized strategy optimizer factory."""

  strategy_factory: VectorizedStrategyFactory

  def __call__(
      self,
      suggestion_batch_size: int,
      max_evaluations: int,
      max_duration: Optional[datetime.timedelta] = None,
  ) -> VectorizedOptimizer:
    """Generates a new VectorizedOptimizer object."""
    return VectorizedOptimizer(
        strategy_factory=self.strategy_factory,
        suggestion_batch_size=suggestion_batch_size,
        max_evaluations=max_evaluations,
        max_duration=max_duration,
    )
