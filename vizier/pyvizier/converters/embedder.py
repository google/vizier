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

"""Vizier problem statement and trials scaler.

The ProblemAndTrialsScaler class has two main functionalities:
1. Scale (aka "embed") the ProblemStatment to a new ProblemStatment with scaled
search space.
2. Map and unmap Vizier TrialSuggestion from and to the scaled search space.

The parameters mapping is as follow:
- FLOAT parameters are mapped to FLOAT with [0,1] bounds.
- INTEGER parameters are mapped to FLOAT with [0,1] bounds.
- DISCRETE parameters are mapped to DISCRETE with converted feasible values.
- CATEGORICAL parameters don't change.

The scaling is performed using the DefaultTrialConverter.
"""

import copy
from typing import Sequence, TypeVar
import attr
import numpy as np
from vizier import pyvizier as vz
from vizier.pyvizier.converters import core


_T = TypeVar('_T', vz.Trial, vz.TrialSuggestion)


class ProblemAndTrialsScaler:
  """Vizier problem statement and trials scaler."""

  def __init__(self, problem: vz.ProblemStatement):
    """Initializes the class by creating a converter and scaled problem statement.

    Arguments:
      problem: The ProblemStatemet to be scaled.
    """

    def create_param_converter(pc):
      return core.DefaultModelInputConverter(
          pc, max_discrete_indices=0, scale=True
      )

    self._converter = core.DefaultTrialConverter(
        parameter_converters=[
            create_param_converter(pc) for pc in problem.search_space.parameters
        ]
    )
    # Create the new embedded search space.
    emb_search_space = vz.SearchSpace()
    for param in problem.search_space.parameters:
      if param.type in [vz.ParameterType.DOUBLE, vz.ParameterType.INTEGER]:
        # DOUBLE/INTEGER params are scaled to [0.0, 1.0] and converted to FLOAT.
        emb_search_space.root.add_float_param(
            param.name,
            self._converter.output_specs[param.name].bounds[0],
            self._converter.output_specs[param.name].bounds[1],
        )
      elif param.type == vz.ParameterType.DISCRETE:
        # DISCRETE params feasible values are scaled.
        emb_search_space.root.add_discrete_param(
            param.name,
            feasible_values=self._scale_discrete_feasible_values(
                problem, param
            ),
        )
      elif param.type == vz.ParameterType.CATEGORICAL:
        # CATEGORICAL parameters are left unchanged.
        emb_search_space.root.add_categorical_param(
            param.name, feasible_values=param.feasible_values
        )
      else:
        raise ValueError('Unsupported parameter type (%s)' % param.type)
    # Clone the entire problem statement and only update the search space.
    self._embedded_problem_statement = copy.deepcopy(problem)
    self._embedded_problem_statement.search_space = emb_search_space

  @property
  def problem_statement(self) -> vz.ProblemStatement:
    """Returns the scaled problem statement."""
    return self._embedded_problem_statement

  def _scale_discrete_feasible_values(
      self, problem: vz.ProblemStatement, param: vz.ParameterConfig
  ):
    """Scales the DISCRETE parameter feasible values."""
    if param.type != vz.ParameterType.DISCRETE:
      raise ValueError('Expects DISCRETE parameter but got %s.' % param.type)
    # Obtain the converter for the DISCRETE parameter.
    param_converter = self._converter.parameter_converters_dict[param.name]
    # Convert each original feasible values to form the new feasible values.
    new_feasible_values = []
    for feasible_value in problem.search_space.get(param.name).feasible_values:
      tmp_trial = vz.Trial({param.name: feasible_value})
      new_feasible_value = param_converter.convert([tmp_trial]).item(0, 0)
      new_feasible_values.append(new_feasible_value)
    return new_feasible_values

  def map(self, trials: Sequence[_T]) -> list[_T]:
    """Map trials from the original search space to the embedded space."""
    embedded_trials = []
    for trial in trials:
      parameters = vz.ParameterDict()
      for name, feature in self._converter.to_features([trial]).items():
        if (
            self.problem_statement.search_space.get(name).type
            == vz.ParameterType.CATEGORICAL
        ):
          # CATEGORICAL parameters values are unchanged.
          parameters[name] = trial.parameters[name].value
        else:
          # Assign the converted value.
          parameters[name] = feature.item(0, 0)
      # Create a copy of the trial with updated parameters.
      embedded_trials.append(attr.evolve(trial, parameters=parameters))

    return embedded_trials

  def unmap(self, trials: Sequence[_T]) -> list[_T]:
    """Unmap trials from the embedded search space to the original space."""
    unmapped_trials = []
    for trial in trials:
      parameters = vz.ParameterDict()
      for name, parameter_value in trial.parameters.items():
        value = parameter_value.value
        if (
            self.problem_statement.search_space.get(name).type
            == vz.ParameterType.CATEGORICAL
        ):
          # CATEGORICAL parameter values are left unchanged.
          parameters[name] = value
        else:
          # Get the parameter converter.
          param_converter = self._converter.parameter_converters_dict[name]
          # Convert back the feature to parameters in the original space.
          parameters[name] = param_converter.to_parameter_values(
              np.array(value)
          )[0]
      # Create a copy of the trial with updated parameters.
      unmapped_trials.append(attr.evolve(trial, parameters=parameters))
    return unmapped_trials
