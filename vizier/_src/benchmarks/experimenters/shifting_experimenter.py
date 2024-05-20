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

"""Experimenter that shifts the values of the parameters in Evaluation."""

import copy
from typing import Sequence

import numpy as np
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter
from vizier.pyvizier import converters


class ShiftingExperimenter(experimenter.Experimenter):
  """ShiftingExperimenter shifts the parameters of suggestions in Evaluate."""

  def __init__(
      self,
      exptr: experimenter.Experimenter,
      shift: np.ndarray,
      should_restrict=True,
  ):
    """ShiftingExperiment shifts parameter values before passing to exptr.

    Currently only supports flat double search spaces. Note that when
    should_restrict is True, the parameter bounds of the search space are
    RESTRICTED to never cause a parameter to
    exceed the underlying experimenter's parameter bounds, so the problem
    statement will change.

    Args:
      exptr: Underlying experimenter to be wrapped.
      shift: Shift that broadcasts to array of shape (dimension,).
      should_restrict: Whether to restrict the parameter bounds of search space.

    Raises:
      ValueError: Non-positive dimension or non-broadcastable/large shift.
    """
    self._exptr = exptr
    exptr_problem_statement = exptr.problem_statement()

    if exptr_problem_statement.search_space.is_conditional:
      raise ValueError(
          'Search space should not have conditional'
          f' parameters  {exptr_problem_statement}'
      )
    dimension = len(exptr_problem_statement.search_space.parameters)
    if dimension <= 0:
      raise ValueError(f'Invalid dimension: {dimension}')
    try:
      # Attempts a broadcast to check broadcasting.
      self._shift = np.broadcast_to(shift, (dimension,))
    except ValueError as broadcast_err:
      raise ValueError(
          f'Shift {shift} is not broadcastable for dim: {dimension}.\n'
      ) from broadcast_err

    # Converter should be in the underlying extpr space.
    self._converter = converters.TrialToArrayConverter.from_study_config(
        study_config=exptr_problem_statement,
        scale=False,
        should_clip=should_restrict,
    )

    self._problem_statement = copy.deepcopy(exptr_problem_statement)
    if should_restrict:
      self._problem_statement.search_space = pyvizier.SearchSpace()

      for parameter, shift in zip(
          exptr_problem_statement.search_space.parameters, self._shift
      ):
        if parameter.type != pyvizier.ParameterType.DOUBLE:
          raise ValueError(f'Non-double parameters {parameter}')
        if (bounds := parameter.bounds) is None:
          raise ValueError(f'Parameter {parameter} has no bounds')

        if abs(shift) >= bounds[1] - bounds[0]:
          raise ValueError(
              f'Bounds {bounds} may need to be extended'
              f'as shift {shift} is too large '
          )
        # Shift the bounds to maintain valid bounds.
        if shift >= 0:
          new_bounds = (bounds[0] + shift, bounds[1])
        else:
          new_bounds = (bounds[0], bounds[1] + shift)
        self._problem_statement.search_space.add(
            pyvizier.ParameterConfig.factory(
                name=parameter.name,
                bounds=new_bounds,
                scale_type=parameter.scale_type,
                default_value=parameter.default_value,
                external_type=parameter.external_type,
            ),
        )

  def problem_statement(self) -> pyvizier.ProblemStatement:
    return copy.deepcopy(self._problem_statement)

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]) -> None:
    """Evaluate the trials after shifting their parameters by -shift."""
    previous_parameters = [suggestion.parameters for suggestion in suggestions]
    self._offset(suggestions, self._shift)
    self._exptr.evaluate(suggestions)
    # Must replace stored previous parameters since offsetting may clip.
    for parameters, suggestion in zip(previous_parameters, suggestions):
      suggestion.parameters = parameters

  def _offset(
      self, suggestions: Sequence[pyvizier.Trial], shift: np.ndarray
  ) -> None:
    """Offsets parameter values (OOB values are clipped)."""
    for suggestion in suggestions:
      features = self._converter.to_features([suggestion])
      new_parameters = self._converter.to_parameters(features - shift)[0]
      suggestion.parameters = new_parameters

  def __repr__(self):
    return f'ShiftingExperimenter({self._shift}) on {str(self._exptr)}'
