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

"""Flips metric goal and value signs."""
from typing import Sequence
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter as experimenter_lib


class SignFlipExperimenter(experimenter_lib.Experimenter):
  """Flips the goal and objective values of the wrapped Experimenter."""

  def __init__(
      self,
      experimenter: experimenter_lib.Experimenter,
      flip_objectives_only: bool = True,
  ):
    """Init.

    Args:
      experimenter: Original experimenter to flip.
      flip_objectives_only: Whether to only flip objective metrics in trials
        corresponding to problem statement objectives. If True, auxiliary
        metrics will not be flipped.
    """
    self._experimenter = experimenter
    self._flip_objectives_only = flip_objectives_only
    self._original_objectives = {
        m_config.name: m_config
        for m_config in self._experimenter.problem_statement().metric_information
    }

  def problem_statement(self) -> pyvizier.ProblemStatement:
    problem = self._experimenter.problem_statement()
    for metric_config in problem.metric_information:
      if metric_config.goal.is_maximize:
        metric_config.goal = pyvizier.ObjectiveMetricGoal.MINIMIZE
      elif metric_config.goal.is_minimize:
        metric_config.goal = pyvizier.ObjectiveMetricGoal.MAXIMIZE
    return problem

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    self._experimenter.evaluate(suggestions)
    for suggestion in suggestions:
      if suggestion.final_measurement is None:
        continue

      metric_dict = {}
      for name, metric in suggestion.final_measurement.metrics.items():
        if self._flip_objectives_only and name in self._original_objectives:
          metric_dict[name] = pyvizier.Metric(value=-1.0 * metric.value)
        elif not self._flip_objectives_only:
          metric_dict[name] = pyvizier.Metric(value=-1.0 * metric.value)
        else:
          metric_dict[name] = metric
      suggestion.final_measurement.metrics = metric_dict
