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
from typing import Optional, Protocol, Union, List

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
  def best_features_results(self) -> List[VectorizedStrategyResult]:
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
    max_evaluations: The maximum number of objective function evaluations.
    max_duration: The maximum duration of the optimization process.
  """
  strategy_factory: VectorizedStrategyFactory
  max_evaluations: int = 15_000
  max_duration: Optional[datetime.timedelta] = None
  _start_time: datetime.datetime = attr.field(init=False)
  _evaluated_count: int = attr.field(init=False, default=0)
  _strategy: VectorizedStrategy = attr.field(init=False)
  _converter: converters.TrialToArrayConverter = attr.field(init=False)

  def optimize(
      self,
      converter: converters.TrialToArrayConverter,
      score_fn: BatchArrayScoreFunction,
      count: int = 1,
  ) -> None:
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

    Arguments:
      converter: The converter used to convert Trials to arrays.
      score_fn: A callback that expects 2D Array with dimensions (batch_size,
        features_count) and returns a 1D Array (batch_size,).
      count: The number of suggestions to generate.
    """
    # Create a new strategy using the factory. The effective search space is
    # included in the converter.
    self._strategy = self.strategy_factory(converter, count)
    # Store the converter to generate best_cadidates trials.
    self._converter = converter
    self._start_time = datetime.datetime.now()
    self._evaluated_count = 0

    while not self._should_stop():
      new_features = self._strategy.suggest()
      new_rewards = score_fn(new_features)
      self._strategy.update(new_rewards)
      self._evaluated_count += len(new_rewards)
    logging.info(
        'Optimization completed. Duration: %s. Evalutions: %s. Best Results: %s',
        datetime.datetime.now() - self._start_time, self._evaluated_count,
        self._strategy.best_features_results)

  @property
  def strategy(self) -> VectorizedStrategy:
    """Returns the strategy used for the optimization."""
    if not self._evaluated_count:
      raise Exception("Optimizer hasn't run yet. Call optimize first!")
    return self._strategy

  @property
  def best_candidates(self) -> List[vz.Trial]:
    """Returns the best candidate trials in the original search space."""
    if not self._evaluated_count:
      raise Exception("Optimizer hasn't run yet. Call optimize first!")
    # Convert the array features to trials.
    best_results = self._strategy.best_features_results
    trials = []
    for best_result in best_results:
      # Create trials and convert the strategy features back to parameters.
      trial = vz.Trial(
          parameters=self._converter.to_parameters(
              np.expand_dims(best_result.features, axis=0))[0])
      trial.complete(vz.Measurement({'acquisition': best_result.reward}))
      trials.append(trial)
    return trials

  def _should_stop(self) -> bool:
    """Determines if the optimizer has reached its optimization budget."""
    duration = datetime.datetime.now() - self._start_time
    if self.max_duration and duration >= self.max_duration:
      logging.info(
          'Optimization completed. Reached time limit. Duration: %s. Evalutions: %s',
          duration, self._evaluated_count)
    elif self._evaluated_count >= self.max_evaluations:
      logging.info(
          'Optimization completed. Reched evaluations limit. Duration: %s. Evalutions: %s',
          duration, self._evaluated_count)
      return True
    else:
      return False
