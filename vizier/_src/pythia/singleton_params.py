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

"""A helper functionality to handle singleton parameters."""

import copy
import logging
from typing import Sequence

import attrs
from vizier import pyvizier as vz


@attrs.define
class SingletonParameterHandler:
  """A helper class to handle singleton parameters.

  The class allows to remove singleton parameters (i.e. having a single value)
  from the problem statement's search space and from trial, so that designers
  won't have to handle them. In addition, it allows to re-add the singleton
  values back to the trial suggestions, so that the users can access them if
  needed.
  """

  problem: vz.ProblemStatement
  # ----------------------------------------------------------------------------
  # Internal attributes
  # ----------------------------------------------------------------------------
  _singletons: dict[str, vz.ParameterValueTypes] = attrs.field(init=False)
  _stripped_problem: vz.ProblemStatement = attrs.field(init=False)

  @property
  def stripped_problem(self) -> vz.ProblemStatement:
    """Returns the stripped problem."""
    return self._stripped_problem

  def __attrs_post_init__(self):
    logging.info("problem: %s", self.problem)
    self._singletons = self._find_singletons()
    self._stripped_problem = self._strip_problem()

  def _find_singletons(self) -> dict[str, vz.ParameterValueTypes]:
    """Finds the singleton parameters in the problem."""
    singletons = {}
    for param in self.problem.search_space.parameters:
      if param.type == vz.ParameterType.DOUBLE:
        if param.bounds[0] == param.bounds[1]:
          singletons[param.name] = param.bounds[0]
      elif param.type == vz.ParameterType.INTEGER:
        if param.bounds[0] == param.bounds[1]:
          singletons[param.name] = param.bounds[0]
      elif param.type in (
          vz.ParameterType.CATEGORICAL,
          vz.ParameterType.DISCRETE,
      ):
        if len(param.feasible_values) == 1:
          singletons[param.name] = param.feasible_values[0]
      elif param.type == vz.ParameterType.CUSTOM:
        pass
      else:
        raise ValueError("Unknown parameter type: %s" % param.type)
    return singletons

  def _strip_problem(self) -> vz.ProblemStatement:
    """Strips the problem of the singleton parameters."""
    stripped_problem = copy.deepcopy(self.problem)
    for param in self.problem.search_space.parameters:
      if param.name in self._singletons:
        stripped_problem.search_space.pop(param.name)
    return stripped_problem

  def strip_trials(
      self, trials: Sequence[vz.TrialSuggestion]
  ) -> Sequence[vz.TrialSuggestion]:
    """Strips the trials of the singleton parameters."""
    if not self._singletons:
      return trials
    new_trials = []
    for trial in trials:
      new_trial = copy.deepcopy(trial)
      for param_name in trial.parameters:
        if param_name in self._singletons:
          del new_trial.parameters[param_name]
      new_trials.append(new_trial)
    return new_trials

  def augment_trials(
      self, trials: Sequence[vz.TrialSuggestion]
  ) -> Sequence[vz.TrialSuggestion]:
    """Augments the trials with the singleton parameters."""
    if not self._singletons:
      return trials
    new_trials = []
    for trial in trials:
      new_trial = copy.deepcopy(trial)
      for singleton_name, singleton_value in self._singletons.items():
        new_trial.parameters[singleton_name] = singleton_value
      new_trials.append(new_trial)
    return new_trials
