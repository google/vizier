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
      suggested features: (suggestion_count, features_count)
    """

  @property
  @abc.abstractmethod
  def suggestion_count(self) -> int:
    """The number of suggestions returned as every suggest call."""

  @property
  @abc.abstractmethod
  def best_results(self) -> list[VectorizedStrategyResult]:
    """The best features and rewards the strategy seen thus far.

    Note that the result is in the *scaled* space, which is [0,1]^n.

    Returns:
      The best search results *before* converted backward to parameters.
    """

  @abc.abstractmethod
  def update(self, rewards: Array) -> None:
    """Update the strategy state with the results of the last suggestions.

    Arguments:
      rewards: (suggestion_count, )
    """


class VectorizedStrategyFactory(Protocol):
  """Factory class to generate vectorized strategy.

  It's used in VectorizedOptimizer to create a new strategy every 'optimize'
  call.
  """

  def __call__(self, converter: converters.TrialToArrayConverter,
               count: int) -> VectorizedStrategy:
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
    strategy_factory: An factory to call for generating new strategy.
  """
  strategy_factory: VectorizedStrategyFactory

  def optimize(
      self,
      converter: converters.TrialToArrayConverter,
      score_fn: BatchArrayScoreFunction,
      *,
      count: int = 1,
      max_evaluations: int = 15_000,
      max_duration: Optional[datetime.timedelta] = None,
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
      max_evaluations: The maximum number of objective function evaluations.
      max_duration: The maximum duration of the optimization process.

    Returns:
      The best trials found in the optimization.
    """

    # Create a new strategy using the factory. The effective search space is
    # included in the converter.
    strategy = self.strategy_factory(converter, count)
    start_time = datetime.datetime.now()
    evaluated_count = 0

    while not self._should_stop(start_time, evaluated_count, max_evaluations,
                                max_duration):
      new_features = strategy.suggest()
      new_rewards = score_fn(new_features)
      strategy.update(new_rewards)
      evaluated_count += len(new_rewards)
    logging.info(
        'Optimization completed. Duration: %s. Evalutions: %s. Best Results: %s',
        datetime.datetime.now() - start_time, evaluated_count,
        strategy.best_results)

    return self._best_candidates(strategy, converter)

  def _best_candidates(
      self, strategy: VectorizedStrategy,
      converter: converters.TrialToArrayConverter) -> list[vz.Trial]:
    """Returns the best candidate trials in the original search space."""
    # Convert the array features to trials.
    best_results = strategy.best_results
    trials = []
    for best_result in best_results:
      # Create trials and convert the strategy features back to parameters.
      trial = vz.Trial(
          parameters=converter.to_parameters(
              np.expand_dims(best_result.features, axis=0))[0])
      trial.complete(vz.Measurement({'acquisition': best_result.reward}))
      trials.append(trial)
    return trials

  def _should_stop(self, start_time: datetime.datetime, evaluated_count: int,
                   max_evaluations: int,
                   max_duration: datetime.timedelta) -> bool:
    """Determines if the optimizer has reached its optimization budget."""
    duration = datetime.datetime.now() - start_time
    if max_duration and duration >= max_duration:
      logging.info(
          'Optimization completed. Reached time limit. Duration: %s. Evalutions: %s',
          duration, evaluated_count)
      return True
    elif evaluated_count >= max_evaluations:
      logging.info(
          'Optimization completed. Reched evaluations limit. Duration: %s. Evalutions: %s',
          duration, evaluated_count)
      return True
    else:
      return False
