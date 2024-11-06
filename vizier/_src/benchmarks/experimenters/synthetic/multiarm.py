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

"""Multi-arm bandit environments.

Search space is pure 1-D categorical, and rewards are given by fixed
distributions.
"""

import copy
from typing import Optional, Sequence

import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter


def _default_multiarm_problem(
    num_arms: int, arms_as_chars: bool
) -> vz.ProblemStatement:
  """Returns default multi-arm problem statement."""
  problem = vz.ProblemStatement()
  problem.metric_information.append(
      vz.MetricInformation(name="reward", goal=vz.ObjectiveMetricGoal.MAXIMIZE)
  )

  if arms_as_chars:
    # Starts with 'a' character.
    feasible_values = [chr(i + 97) for i in range(num_arms)]
  else:
    feasible_values = [str(i) for i in range(num_arms)]

  problem.search_space.root.add_categorical_param(
      name="arm", feasible_values=feasible_values
  )
  return problem


class BernoulliMultiArmExperimenter(experimenter.Experimenter):
  """Uses a collection of Bernoulli arms with given probabilities."""

  def __init__(
      self,
      probs: Sequence[float],
      arms_as_chars: bool = True,
      seed: Optional[int] = None,
  ):
    self._probs = probs
    self._rng = np.random.RandomState(seed)
    self._problem = _default_multiarm_problem(len(self._probs), arms_as_chars)

  def problem_statement(self) -> vz.ProblemStatement:
    return copy.deepcopy(self._problem)

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    """Each arm has a fixed probability of outputting 0 or 1 reward."""
    feasibles = self._problem.search_space.parameters[0].feasible_values
    for suggestion in suggestions:
      arm_index = feasibles.index(suggestion.parameters["arm"].value)
      prob = self._probs[arm_index]
      reward = self._rng.choice([0, 1], p=[1 - prob, prob])
      suggestion.final_measurement = vz.Measurement(metrics={"reward": reward})


class FixedMultiArmExperimenter(experimenter.Experimenter):
  """Rewards are deterministic."""

  def __init__(self, rewards: Sequence[float], arms_as_chars: bool = True):
    self._rewards = rewards
    self._problem = _default_multiarm_problem(len(self._rewards), arms_as_chars)

  def problem_statement(self) -> vz.ProblemStatement:
    return copy.deepcopy(self._problem)

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    feasibles = self._problem.search_space.parameters[0].feasible_values
    for suggestion in suggestions:
      arm_index = feasibles.index(suggestion.parameters["arm"].value)
      reward = self._rewards[arm_index]
      suggestion.final_measurement = vz.Measurement(metrics={"reward": reward})
