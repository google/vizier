"""Base class for vectorized acquisition optimizers."""

import abc
import datetime
import logging
from typing import Callable, Optional, Tuple, Union

import attr
import chex
import numpy as np

# Support for both JAX and Numpy arrays
Array = Union[np.ndarray, chex.Array]


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
  def best_results(self) -> Tuple[Array, float]:
    """Returns the best features and reward the strategy seen thus far."""

  @abc.abstractmethod
  def update(self, rewards: Array) -> None:
    """Update the strategy state with the results of the last suggestions.

    Arguments:
      rewards: (suggestion_count, )
    """


@attr.define
class VectorizedOptimizerConfiguration:
  """Optimizer configuration.

  Attributes:
    max_evaluations: The maximum number of objective function evaluations.
    max_duration: The maximum duration of the optimization process.
  """
  max_evaluations: int = 15_000
  max_duration: Optional[datetime.timedelta] = None


@attr.define
class VectorizedOptimizer(abc.ABC):
  """Vectorized strategy optimizer.

  The optimizer is responsible for running the iterative optimization process
  using the vectorized strategy, which consists of:
  1. Ask the strategy for suggestions.
  2. Evaluate the suggestions to get rewards.
  3. Tell the strategy about the rewards of its suggestion, so the strategy can
  update its internal state.

  The optimization process will terminate when the time limit or the total
  objective function evaluations limit has exceeded.
  """
  strategy: VectorizedStrategy
  config: VectorizedOptimizerConfiguration
  _start_time: datetime.datetime = attr.field(init=False)
  _evaluated_count: int = attr.field(init=False)

  def optimize(self, obj_func: Callable[[Array], Array]) -> None:
    """Optimize the objective function.

    The ask-evaluate-tell optimization procedure that runs until the allocated
    time or evaluations count exceeds.

    The number of suggestions is determined by the strategy, which is the
    `suggestion_count` property.

    Arguments:
      obj_func: A callback that expects 2D Array with dimensions
        (suggestion_count, features_count) and returns a 1D Array
        (suggestion_count,)
    """
    self._start_time = datetime.datetime.now()
    self._evaluated_count = 0
    while not self._should_stop():
      new_features = self.strategy.suggest()
      new_rewards = obj_func(new_features)
      self.strategy.update(new_rewards)
      self._evaluated_count += len(new_rewards)
    logging.info(
        'Optimization completed. Duration: %s. Evalutions: %s. Best Results: %s',
        datetime.datetime.now() - self._start_time, self._evaluated_count,
        self.best_results)

  @property
  def best_results(self) -> Tuple[Array, float]:
    """Returns the best reward and associated features."""
    if not self._evaluated_count:
      raise Exception("Optimizer hasn't run yet. Call optimize first!")
    return self.strategy.best_results

  def _should_stop(self) -> bool:
    """Determines if the optimizer has reached its optimization budget."""
    duration = datetime.datetime.now() - self._start_time
    if self.config.max_duration and duration >= self.config.max_duration:
      logging.info(
          'Optimization completed. Reached time limit. Duration: %s. Evalutions: %s',
          duration, self._evaluated_count)
    elif self._evaluated_count >= self.config.max_evaluations:
      logging.info(
          'Optimization completed. Reched evaluations limit. Duration: %s. Evalutions: %s',
          duration, self._evaluated_count)
      return True
    else:
      return False
