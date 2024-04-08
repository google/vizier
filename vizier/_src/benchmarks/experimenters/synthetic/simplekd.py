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

"""SimpleKD experimenter.

The experimenter supports three flavors (corner, center, mixed), each of which
has a different optia location. The categorical parameter takes the same three
values, and the objective function depends on the value of the categorical
parameter which make it harder to optimize.
"""

from typing import Literal, Sequence, Union
import attrs
import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter


def _float_term(x_list: list[float]) -> float:
  """Computes the float term on a list of values.

  Args:
    x_list: Elements in the list correspond to a different dimension/Parameter.

  Returns:
    The float term accounting for all the float Parameters.
  """
  float_term = 0
  for x in x_list:
    float_term += min(
        1.6,
        (x > -0.8) * -((x - 1) ** 2)
        + (x <= -0.8) * (-(1.8**2) + 150 * (x + 0.8) ** 2),
    )
  return float_term


SimpleKDCategory = Literal['corner', 'center', 'mixed']


def _categorical_term(x: str, best_category: SimpleKDCategory) -> float:
  """Computes the categorical term."""
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


def _discrete_term(x_list: list[int]) -> float:
  """Computes the discrete term on a list of values."""
  discrete_term = 0
  for x in x_list:
    discrete_term += [1.2, 0.0, 0.6, 0.8, 1.0][
        _feasible_discrete_values.index(x)
    ]
  return discrete_term


def _int_term(x_list: list[int]) -> float:
  """Computes the int term on a list of values."""
  int_term = 0
  for x in x_list:
    int_term += np.power(x - 2.2, 2) / 2.0
  return int_term


@attrs.define
class SimpleKDExperimenter(experimenter.Experimenter):
  """Simple problem with arbitrary number of parameter of each type.

  SimpleKD is a carefully constructed function to trap an underexploring
  hillclimb algorithm to local optima. It is parameterized by "best_category",
  which optimization algorithm must get right in order to reach the optimum.

  Note that CATEGORICAL type has only a single parameter as there's no notion
  of hillclimb or direction for categorical parameter.

  When 'output_relative_error' is True (default case) the objective value is
  the relative error compared to the optimal objective value (which could be
  negative) and the objective goal is MINIMIZE. This option allows for using the
  experiementer results directly without needing for further processing.
  When 'output_relative_error' is False, the objecitve is the SimpleKd function
  value and the objective goal is MAXIMIZE.

  The "best_category" value refers to where the optimum occurs.
  """

  best_category: SimpleKDCategory = attrs.field()  # type: ignore
  num_float_param: int = 1
  num_discrete_param: int = 1
  num_int_param: int = 1
  output_relative_error: bool = True

  def __attrs_post_init__(self):
    if self.output_relative_error and abs(self.optimal_objective) < 1e-6:
      raise ValueError(
          f"Optimal objective is too small {self.optimal_objective}, can't"
          ' compute relative error.'
      )

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    for suggestion in suggestions:
      param_values = self._get_param_values(suggestion)
      value = self._compute(param_values)
      if self.output_relative_error:
        value = abs((value - self.optimal_objective) / self.optimal_objective)
      suggestion.complete(vz.Measurement({'value': value}))

  def problem_statement(self) -> vz.ProblemStatement:
    problem = vz.ProblemStatement()
    for f in range(self.num_float_param):
      problem.search_space.root.add_float_param(f'float_{f}', -1, 1)
    for d in range(self.num_discrete_param):
      problem.search_space.root.add_discrete_param(
          f'discrete_{d}', _feasible_discrete_values
      )
    for i in range(self.num_int_param):
      problem.search_space.root.add_int_param(f'int_{i}', 1, 3)
    problem.search_space.root.add_categorical_param(
        'categorical', ('corner', 'center', 'mixed')
    )
    if self.output_relative_error:
      problem.metric_information.append(
          vz.MetricInformation(
              name='value', goal=vz.ObjectiveMetricGoal.MINIMIZE
          )
      )
    else:
      problem.metric_information.append(
          vz.MetricInformation(
              name='value', goal=vz.ObjectiveMetricGoal.MAXIMIZE
          )
      )
    return problem

  def _get_param_values(
      self, trial: vz.Trial
  ) -> dict[str, list[Union[float, int, str]]]:
    """Extract the trial parameter values by type."""

    float_list = []
    int_list = []
    discrete_list = []
    for index in range(self.num_float_param):
      float_list.append(trial.parameters[f'float_{index}'].value)
    for index in range(self.num_discrete_param):
      discrete_list.append(trial.parameters[f'discrete_{index}'].value)
    for index in range(self.num_int_param):
      int_list.append(trial.parameters[f'int_{index}'].value)

    return {
        'float': float_list,
        'int': int_list,
        'discrete': discrete_list,
        'categorical': [trial.parameters['categorical'].value],
    }

  def _compute(self, params: dict[str, list[Union[float, int, str]]]) -> float:
    """Computes the SimpleKD objective value."""
    if params['categorical'][0] == 'corner':
      return (
          _categorical_term(params['categorical'][0], self.best_category)
          + 0.8 * _float_term(params['float'])
          + _discrete_term(params['discrete'])
          + _int_term(params['int'])
      )
    elif params['categorical'][0] == 'center':
      return (
          _categorical_term(params['categorical'][0], self.best_category)
          - _float_term(params['float'])
          - _discrete_term(params['discrete'])
          - _int_term(params['int'])
      )
    elif params['categorical'][0] == 'mixed':
      return (
          _categorical_term(params['categorical'][0], self.best_category)
          + 0.8 * _float_term(params['float'])
          + _discrete_term(params['discrete'])
          - _int_term(params['int'])
      )
    else:
      raise NotImplementedError(
          f'Unknown categorical parameter: {params["categorical"]}'
      )

  @property
  def optimal_objective(self) -> float:
    """Computes the optimal objective value of a SimpleKD problem."""
    values = {}
    if self.best_category == 'corner':
      values['float'] = [-1.0 for _ in range(self.num_float_param)]
      values['int'] = [1 for _ in range(self.num_float_param)]
      values['discrete'] = [1 for _ in range(self.num_float_param)]
      values['categorical'] = ['corner']

    if self.best_category == 'center':
      values['float'] = [-0.8 for _ in range(self.num_float_param)]
      values['int'] = [2 for _ in range(self.num_float_param)]
      values['discrete'] = [2 for _ in range(self.num_float_param)]
      values['categorical'] = ['center']

    if self.best_category == 'mixed':
      values['float'] = [-1.0 for _ in range(self.num_float_param)]
      values['int'] = [2 for _ in range(self.num_float_param)]
      values['discrete'] = [1 for _ in range(self.num_float_param)]
      values['categorical'] = ['mixed']

    return self._compute(values)
