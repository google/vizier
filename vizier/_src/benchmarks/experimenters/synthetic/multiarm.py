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

from typing import Mapping, Optional, Sequence

import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter


def _default_multiarm_problem(arms: Sequence[str]) -> vz.ProblemStatement:
  """Returns default multi-arm problem statement."""
  problem = vz.ProblemStatement()
  problem.metric_information.append(
      vz.MetricInformation(name="reward", goal=vz.ObjectiveMetricGoal.MAXIMIZE)
  )
  problem.search_space.root.add_categorical_param("arm", feasible_values=arms)
  return problem


class BernoulliMultiArmExperimenter(experimenter.Experimenter):
  """Uses a mapping from arm to Bernoulli probability of success."""

  def __init__(
      self, arms_to_probs: Mapping[str, float], seed: Optional[int] = None
  ):
    self._arms_to_probs = arms_to_probs
    self._rng = np.random.RandomState(seed)

  def problem_statement(self) -> vz.ProblemStatement:
    return _default_multiarm_problem(list(self._arms_to_probs.keys()))

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    """Each arm has a fixed probability of outputting 0 or 1 reward."""
    for suggestion in suggestions:
      arm = suggestion.parameters["arm"].value
      prob = self._arms_to_probs[arm]
      reward = self._rng.choice([0, 1], p=[1 - prob, prob])
      suggestion.final_measurement = vz.Measurement(metrics={"reward": reward})


class FixedMultiArmExperimenter(experimenter.Experimenter):
  """Rewards are deterministic."""

  def __init__(self, arms_to_rewards: Mapping[str, float]):
    self._arms_to_rewards = arms_to_rewards

  def problem_statement(self) -> vz.ProblemStatement:
    return _default_multiarm_problem(list(self._arms_to_rewards.keys()))

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    for suggestion in suggestions:
      arm = suggestion.parameters["arm"].value
      reward = self._arms_to_rewards[arm]
      suggestion.final_measurement = vz.Measurement(metrics={"reward": reward})
