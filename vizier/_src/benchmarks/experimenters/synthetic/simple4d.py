# Copyright 2023 Google LLC.
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

"""Simple4D experimenter."""

from typing import Sequence

import attrs
import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter


def _float_term(x: float) -> float:
  return min(
      1.6,
      (x > -0.8) * -((x - 1) ** 2)
      + (x <= -0.8) * (-(1.8**2) + 150 * (x + 0.8) ** 2),
  )


Simple4DCategory = str


def _categorical_term(x: str, best_category: Simple4DCategory) -> float:
  if x != best_category:
    return 0
  elif x == 'corner':
    return 1
  elif x == 'center':
    return 1
  elif x == 'mixed':
    return 1.5
  raise NotImplementedError(f'Unknown categorical parameter: {x}')


_feasible_discrete_values = (1, 2, 5, 6, 8)


def _discrete_term(x: int) -> float:
  if x not in _feasible_discrete_values:
    raise NotImplementedError(f'Unknown discrete parameter: {x}')
  return [1.2, 0.0, 0.6, 0.8, 1.0][_feasible_discrete_values.index(x)]


def _int_term(x: int) -> float:
  return np.power(x - 2.2, 2) / 2.0


@attrs.define
class Simple4D(experimenter.Experimenter):
  """Simple problem with one parameter of each type.

  Simple4D is a carefully constructed function to trap an underexploring
  hillclimb algorithm to local optima. It is parameterized by "best_category",
  which optimization algorithm must get right in order to reach the optimum.

  "best_category" value refers to where the optimum occurs.
  """

  best_category: Simple4DCategory = attrs.field()  # type: ignore

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    for suggestion in suggestions:
      suggestion.complete(
          vz.Measurement({'value': self._unimodal4d(suggestion)})
      )

  def problem_statement(self) -> vz.ProblemStatement:
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('float', -1, 1)
    problem.search_space.root.add_int_param('int', 1, 3)
    problem.search_space.root.add_categorical_param(
        'categorical', ('corner', 'center', 'mixed')
    )
    problem.search_space.root.add_discrete_param(
        'discrete', _feasible_discrete_values
    )
    problem.metric_information.append(
        vz.MetricInformation(name='value', goal=vz.ObjectiveMetricGoal.MINIMIZE)
    )
    return problem

  def _unimodal4d(self, trial: vz.Trial) -> float:
    params = trial.parameters.as_dict()
    if params['categorical'] == 'corner':
      return (
          _categorical_term(params['categorical'], self.best_category)
          + 0.8 * _float_term(params['float'])
          + _discrete_term(params['discrete'])
          + _int_term(params['int'])
      )
    elif params['categorical'] == 'center':
      return (
          _categorical_term(params['categorical'], self.best_category)
          - _float_term(params['float'])
          - _discrete_term(params['discrete'])
          - _int_term(params['int'])
      )
    elif params['categorical'] == 'mixed':
      return (
          _categorical_term(params['categorical'], self.best_category)
          + 0.8 * _float_term(params['float'])
          + _discrete_term(params['discrete'])
          - _int_term(params['int'])
      )
    raise NotImplementedError(
        f'Unknown categorical parameter: {params["categorical"]}'
    )
