# Copyright 2024 Google LLC.
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

"""Ensembling design algorithms for choosing between algorithms/experts."""

import abc
import attrs
import numpy as np

# An observation, that is a expert index (int) along with its reward.
IndexWithReward = tuple[int, float]


class EnsembleDesign(abc.ABC):
  """Strategy that samples which index/expert to choose according to rewards."""

  @property
  @abc.abstractmethod
  def ensemble_probs(self) -> np.ndarray:
    """Computes the probability distribution used to choose the expert.

    The returned shape is (num_experts,) and is a probability distribution.
    """
    pass

  @abc.abstractmethod
  def update(self, observation: IndexWithReward, **kwargs):
    """Updates the strategy with observation."""
    pass


@attrs.define
class RandomEnsembleDesign(EnsembleDesign):
  indices: list[int] = attrs.field()

  @property
  def ensemble_probs(self) -> np.ndarray:
    """The probability distribution used to choose the expert."""
    return np.ones(shape=len(self.indices)) / len(self.indices)

  def update(self, observation: IndexWithReward, **kwargs):
    """Updates the strategy with observation."""
    pass


def softmax(x: np.ndarray) -> np.ndarray:
  """Compute softmax values for x."""
  e_x = np.exp(x - np.max(x))
  return e_x / np.sum(e_x)


# https://bjpcjp.github.io/pdfs/math/bandits-exp3-IX-BA.pdf
@attrs.define
class EXP3IXEnsembleDesign(EnsembleDesign):
  """The EXP3-IX Algorithm that is robust against small probabilities."""

  indices: list[int] = attrs.field()
  stepsize: float = attrs.field(
      default=1.0,
      validator=[attrs.validators.instance_of(float), attrs.validators.gt(0)],
  )
  max_reward: float = attrs.field(
      default=1.0,
      validator=[attrs.validators.instance_of(float), attrs.validators.gt(0)],
  )

  def __attrs_post_init__(self):
    self._log_weights = np.zeros(shape=len(self.indices))
    self._history: list[IndexWithReward] = []

  @property
  def ensemble_probs(self) -> np.ndarray:
    return softmax(self._log_weights)

  def update(self, observation: IndexWithReward):
    expert_idx, reward = observation
    reward = min(self.max_reward, reward)
    gamma = 1.0 / np.sqrt(1.0 + len(self._history))
    loss_estimator = (self.max_reward - reward) / (
        self.ensemble_probs[expert_idx] + gamma
    )

    self._log_weights[expert_idx] += self.stepsize * (-loss_estimator)
    self._history.append(observation)


# pytype: disable=attribute-error
# https://www.cs.princeton.edu/courses/archive/fall16/cos402/lectures/402-lec22.pdf.
@attrs.define
class EXP3UniformEnsembleDesign(EnsembleDesign):
  """The EXP3 algorithm with uniform exploration."""

  indices: list[int] = attrs.field()
  stepsize: float = attrs.field(
      default=1.0,
      validator=[attrs.validators.instance_of(float), attrs.validators.gt(0)],
  )
  max_reward: float = attrs.field(
      default=1.0,
      validator=[attrs.validators.instance_of(float), attrs.validators.gt(0)],
  )
  use_loss_formulation: bool = attrs.field(default=False)
  # If the fed-in rewards are estimators, no reward clipping or
  # importance weighting logic are done.
  use_reward_estimator: bool = attrs.field(default=False)

  def __attrs_post_init__(self):
    self._log_weights = np.zeros(shape=len(self.indices))
    self._history: list[IndexWithReward] = []

  @property
  def ensemble_probs(self) -> np.ndarray:
    n_arms = len(self.indices)
    uniform = np.ones(n_arms) * 1.0 / n_arms
    gamma = 1.0 / np.sqrt(1 + len(self._history))
    probs = (1 - gamma) * softmax(self._log_weights) + gamma * uniform
    return probs

  def update(self, observation: IndexWithReward):
    """Update history and weights."""
    expert_idx, reward = observation
    if not self.use_reward_estimator:
      reward = min(self.max_reward, reward)

    gamma = 1.0 / np.sqrt(1 + len(self._history))
    if self.use_loss_formulation:
      if self.use_reward_estimator:
        loss_estimator = self.max_reward - reward
      else:
        loss_estimator = (self.max_reward - reward) / (
            self.ensemble_probs[expert_idx]
        )
      self._log_weights[expert_idx] += -self.stepsize * gamma * loss_estimator
    else:
      if self.use_reward_estimator:
        reward_estimator = reward
      else:
        reward_estimator = reward / (self.ensemble_probs[expert_idx])
      self._log_weights[expert_idx] += self.stepsize * gamma * reward_estimator

    self._history.append(observation)

  @property
  def history(self) -> list[IndexWithReward]:
    return self._history


# Adaptive metalearner that minimizes adaptive regret by ensembling over history
# lengths. See https://arxiv.org/abs/2401.09278 for more details on adaptive
# regret and algorithmic details on time horizon ensembling.
@attrs.define(kw_only=True)
class AdaptiveEnsembleDesign(EnsembleDesign):
  """An EnsembleStrategy that minimizes adaptive regret."""

  indices: list[int] = attrs.field()
  # List of max history lengths.
  max_lengths: list[int] = attrs.field()
  # Base stepsize and meta_stepsize should theoretically be 1/sqrt(n) where
  # n = # of indices (or underlying arms) when using a reward estimator.
  base_stepsize: float = attrs.field(
      default=1.0,
      validator=[attrs.validators.instance_of(float), attrs.validators.gt(0)],
  )
  # Stepsize of the meta-learner.
  meta_stepsize: float = attrs.field(
      default=1.0,
      validator=[attrs.validators.instance_of(float), attrs.validators.gt(0)],
  )
  max_reward: float = attrs.field(
      default=1.0,
      validator=[attrs.validators.instance_of(float), attrs.validators.gt(0)],
  )
  # Whether to use naive sampling for loss estimate.
  naive_sampling: bool = attrs.field(default=False)

  def __attrs_post_init__(self):
    self._log_weights = {}
    self._base_algos = {}
    self._history = []
    for max_length in self.max_lengths:
      # Initialize log_weight = log(1/sqrt(max_length * len(indices))).
      self._base_algos[max_length] = EXP3UniformEnsembleDesign(
          indices=self.indices,
          stepsize=self.base_stepsize,
          use_loss_formulation=False,
          use_reward_estimator=True,
      )
      self._log_weights[max_length] = (
          -np.log(max_length * len(self.indices)) / 2.0
      )

  @property
  def ensemble_probs(self) -> np.ndarray:
    # Return observation probabilities every other observation.
    if len(self._history) % 2 == 1:
      return self.observation_probs

    return self.play_probs

  @property
  def play_probs(self) -> np.ndarray:
    """Returns the probability of playing an expert."""
    ensemble_probs = [algo.ensemble_probs for algo in self._base_algos.values()]
    weights = softmax(np.array(list(self._log_weights.values())))
    final_probs = np.zeros(len(self.indices))
    for weight, probs in zip(weights, ensemble_probs):
      final_probs += weight * probs

    return final_probs

  @property
  def observation_probs(self) -> np.ndarray:
    """Returns the probability of observing an expert."""
    num_experts = len(self.indices)
    uniform = np.ones(num_experts) * 1.0 / num_experts
    if self.naive_sampling:
      return uniform

    algo_prob_max = np.zeros(num_experts)
    algo_prob_sum = np.zeros(num_experts)
    ensemble_probs = [algo.ensemble_probs for algo in self._base_algos.values()]
    for i in range(num_experts):
      algo_prob_max[i] = np.amax(np.array([p[i] ** 2 for p in ensemble_probs]))
      algo_prob_sum[i] = np.sum(np.array([p[i] for p in ensemble_probs]))

    # Take the average of the two max and sum distributions.
    observation_prob = algo_prob_max / (
        2 * np.sum(algo_prob_max)
    ) + algo_prob_sum / (2 * np.sum(algo_prob_sum))
    return observation_prob

  def update(self, observation: IndexWithReward):
    expert_idx, reward = observation
    reward = min(reward, self.max_reward)
    reward_estimator = reward * 1.0 / self.ensemble_probs[expert_idx]

    # Updates the states of base algorithms.
    for max_length, base_algo in self._base_algos.items():
      # If the history is about to be filled, restart.
      if len(base_algo.history) + 1 >= max_length:
        # Initialize log_weight = log(1/sqrt(max_length * len(indices))).
        self._log_weights[max_length] = (
            -np.log(max_length * len(self.indices)) / 2.0
        )
        self._base_algos[max_length] = EXP3UniformEnsembleDesign(
            indices=self.indices,
            stepsize=self.base_stepsize,
            use_loss_formulation=False,
            use_reward_estimator=True,
        )

        continue

      # Otherwise, update meta-weights via unbiased reward estimation.
      reward_estimator_base = reward_estimator * (
          base_algo.ensemble_probs[expert_idx] - self.play_probs[expert_idx]
      )
      # Stepsize is 1/sqrt(max_length).
      gamma = 1.0 / np.sqrt(max_length)
      self._log_weights[max_length] += (
          self.meta_stepsize * gamma * reward_estimator_base
      )

      # Propagate observed reward estimator.
      base_algo.update((expert_idx, reward_estimator))

    self._history.append(observation)


# pytype: enable=attribute-error
