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

"""Applies the noise function to each metric in final measurement."""

import copy
from typing import Sequence
import attrs
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter


@attrs.define
class SwitchExperimenter(experimenter.Experimenter):
  """Creates conditional search space from multiple experimenters via a switch."""

  experimenters: Sequence[experimenter.Experimenter] = attrs.field()

  _switch_param_name: str = attrs.field(default='switch')
  _metric_name: str = attrs.field(default='switch_metric')

  # Created in __attrs_post_init__.
  _exptr_problems: Sequence[vz.ProblemStatement] = attrs.field(init=False)
  _exptr_objective_names: Sequence[str] = attrs.field(init=False)

  def __attrs_post_init__(self):
    self._exptr_problems = [
        exp.problem_statement() for exp in self.experimenters
    ]
    self._exptr_objective_names = [
        ps.metric_information.item().name for ps in self._exptr_problems
    ]

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    for trial in suggestions:
      exptr_index = trial.parameters[self._switch_param_name].value

      trial_copy = copy.deepcopy(trial)
      self.experimenters[exptr_index].evaluate([trial_copy])

      if trial_copy.final_measurement is None:
        continue

      val = trial_copy.final_measurement.metrics[
          self._exptr_objective_names[exptr_index]
      ]
      trial.complete(vz.Measurement(metrics={self._metric_name: val}))

  def problem_statement(self) -> vz.ProblemStatement:
    problem_statement = vz.ProblemStatement()

    children = []
    for i, problem in enumerate(self._exptr_problems):
      for pc in problem.search_space.parameters:
        children.append(([i], pc))

    switching_param = vz.ParameterConfig.factory(
        self._switch_param_name,
        feasible_values=range(len(self.experimenters)),
        children=children,
    )
    problem_statement.search_space.add(switching_param)
    problem_statement.metric_information.append(
        vz.MetricInformation(
            name=self._metric_name, goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )

    return problem_statement
