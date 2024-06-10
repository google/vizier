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

"""Experimenter that permutes the parameters before evaluation."""

import copy
import logging
from typing import Optional, Sequence

import numpy as np
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter


class PermutingExperimenter(experimenter.Experimenter):
  """PermutingExperimenter permutes discrete/categorical parameters."""

  def __init__(
      self,
      exptr: experimenter.Experimenter,
      parameters_to_permute: Sequence[str],
      seed: Optional[int] = None,
  ):
    """PermutingExperiment permutes discrete parameter values before passing to exptr.

    Args:
      exptr: Underlying experimenter to be wrapped.
      parameters_to_permute: Parameter names that are to be permuted.
      seed:

    Raises:
      ValueError: Condtional search space.
    """
    self._exptr = exptr
    exptr_problem_statement = exptr.problem_statement()
    self._problem_statement = exptr_problem_statement

    self._rng = np.random.default_rng(seed)
    if exptr_problem_statement.search_space.is_conditional:
      raise ValueError(
          'Search space should not have conditional'
          f' parameters  {exptr_problem_statement}'
      )

    self._parameter_permutation_dict = {}
    for parameter_name in parameters_to_permute:
      parameter = exptr_problem_statement.search_space.get(parameter_name)
      if not np.isfinite(parameter.num_feasible_values):
        raise ValueError(
            f'Parameter to permute {parameter} is continuous.'
            ' Permuting continuous parameters is not supported.'
        )

      permutation_list = self._rng.permuted(parameter.feasible_values)
      permutation_dict = {
          a: b for a, b in zip(parameter.feasible_values, permutation_list)
      }
      self._parameter_permutation_dict[parameter.name] = permutation_dict

  def problem_statement(self) -> pyvizier.ProblemStatement:
    return copy.deepcopy(self._problem_statement)

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]) -> None:
    """Evaluate the trials after permuting the parameters."""
    previous_parameters = [suggestion.parameters for suggestion in suggestions]
    self._permute(suggestions)
    self._exptr.evaluate(suggestions)
    # Replace stored previous parameters.
    for parameters, suggestion in zip(previous_parameters, suggestions):
      suggestion.parameters = parameters

  def _permute(self, suggestions: Sequence[pyvizier.Trial]) -> None:
    """Permutes parameter values in place."""
    for suggestion in suggestions:
      new_parameters = {}
      for name, parameter in suggestion.parameters.items():
        if name in self._parameter_permutation_dict:
          permutation_dict = self._parameter_permutation_dict[name]
          print('permutation dict', permutation_dict)
          logging.info('Permuting %s ', permutation_dict)
          new_parameters[name] = permutation_dict[parameter.value]
        else:
          new_parameters[name] = parameter
      suggestion.parameters = new_parameters

  def __repr__(self):
    return f'PermutingExperimenter on {str(self._exptr)}'
