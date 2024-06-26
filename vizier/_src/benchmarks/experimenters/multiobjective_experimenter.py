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

"""Multiobjective Experimenters."""

import copy
from typing import Dict, Sequence


from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter


class MultiObjectiveExperimenter(experimenter.Experimenter):
  """MultiObjective Experimenter from a Dict of Experimenters."""

  def __init__(
      self,
      exptrs: Dict[str, experimenter.Experimenter],
  ):
    """Creates a MultiObjectiveExperimenter from a Dict of Experimenters.

    Specifically, the Experimenter takes a Dict of metric names and
    single-objective experimenters with the same search space. Each metric
    is evaluated individually by exptr and combined into a final measurement
    whose name is updated to metric name in exptrs. The problem statement
    of this experimenter is multi-objective with objective names given by keys
    in exptrs.

    Args:
      exptrs: Dict of metric name to Experimenters

    Raises:
      ValueError: Mismatching search space or exptr is not single-objective.
    """
    self._exptrs = exptrs

    # Copy and check that problem statements have same search space.
    self._problem_statement = list(exptrs.values())[0].problem_statement()
    first_search_space = self._problem_statement.search_space
    for exptr in exptrs.values():
      if exptr.problem_statement().search_space != first_search_space:
        raise ValueError(
            'Search space must match for all objectives: \n'
            f'{first_search_space} does not match '
            f'{exptr.problem_statement().search_space}'
        )

    metric_infos = []
    # Keeps track of the underlying metric information name of each extpr.
    self._exptr_to_metric = {}
    for name, exptr in exptrs.items():
      metric_info = exptr.problem_statement().metric_information.item()
      self._exptr_to_metric[name] = metric_info.name
      metric_info.name = name
      metric_infos.append(metric_info)

    self._problem_statement.metric_information = pyvizier.MetricsConfig(
        metric_infos
    )

  def problem_statement(self) -> pyvizier.ProblemStatement:
    return copy.deepcopy(self._problem_statement)

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    suggestions_copy = copy.deepcopy(suggestions)
    measurements = [pyvizier.Measurement() for _ in suggestions]
    for name, exptr in self._exptrs.items():
      exptr.evaluate(suggestions_copy)
      exptr_metric_name = self._exptr_to_metric[name]
      for idx, copied in enumerate(suggestions_copy):
        measurement = measurements[idx]
        assert copied.final_measurement is not None
        measurement.metrics[name] = copied.final_measurement.metrics[
            exptr_metric_name
        ]

    for suggestion, measurement in zip(suggestions, measurements):
      suggestion.complete(measurement)

    return suggestions

  def __repr__(self):
    return (
        f'MultiObjectiveExperimenter with {len(self._exptrs)} exptrs:'
        f' {self._exptrs}'
    )
