"""Experimenter that shifts the values of the parameters in Evaluation."""

import copy
from typing import Sequence

import numpy as np
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter
from vizier.pyvizier import converters


class ShiftingExperimenter(experimenter.Experimenter):
  """ShiftingExperimenter shifts the parameters of suggestions in Evaluate."""

  def __init__(self, exptr: experimenter.Experimenter, shift: np.ndarray):
    """ShiftingExperiment shifts parameter values before passing to exptr.

    Currently only supports flat double search spaces. Note that the parameter
    bounds of the search space are RESTRICTED to never cause a parameter to
    exceed the underlying experimenter's parameter bounds, so the problem
    statement can change.

    Args:
      exptr: Underlying experimenter to be wrapped.
      shift: Shift that broadcasts to array of shape (dimension,).

    Raises:
      ValueError: Non-positive dimension or non-broadcastable/large shift.
    """
    self._exptr = exptr
    exptr_problem_statement = exptr.problem_statement()

    if exptr_problem_statement.search_space.is_conditional:
      raise ValueError('Search space should not have conditional'
                       f' parameters  {exptr_problem_statement}')
    dimension = len(exptr_problem_statement.search_space.parameters)
    if dimension <= 0:
      raise ValueError(f'Invalid dimension: {dimension}')
    try:
      # Attempts a broadcast to check broadcasting.
      self._shift = np.broadcast_to(shift, (dimension,))
    except ValueError as broadcast_err:
      raise ValueError(
          f'Shift {shift} is not broadcastable for dim: {dimension}.'
          '\n') from broadcast_err

    # Converter should be in the underlying extpr space.
    self._converter = converters.TrialToArrayConverter.from_study_config(
        study_config=exptr_problem_statement, scale=False)

    new_parameter_configs = []
    for parameter, shift in zip(exptr_problem_statement.search_space.parameters,
                                self._shift):
      if parameter.type != pyvizier.ParameterType.DOUBLE:
        raise ValueError(f'Non-double parameters {parameter}')
      if (bounds := parameter.bounds) is not None:
        if abs(shift) >= bounds[1] - bounds[0]:
          raise ValueError(f'Bounds {bounds} may need to be extended'
                           f'as shift {shift} is too large ')
        # Shift the bounds to maintain valid bounds.
        if shift >= 0:
          new_bounds = (bounds[0], bounds[1] - shift)
        else:
          # Shift is negative so this restricts the bounds.
          new_bounds = (bounds[0] - shift, bounds[1])
        new_parameter_configs.append(
            pyvizier.ParameterConfig.factory(
                name=parameter.name,
                bounds=new_bounds,
                scale_type=parameter.scale_type,
                default_value=parameter.default_value,
                external_type=parameter.external_type))

    self._problem_statement = copy.deepcopy(exptr_problem_statement)
    self._problem_statement.search_space = pyvizier.SearchSpace._factory(
        new_parameter_configs)

  def problem_statement(self) -> pyvizier.ProblemStatement:
    return self._problem_statement

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    previous_parameters = []
    for suggestion in suggestions:
      features = self._converter.to_features([suggestion])
      parameters = self._converter.to_parameters(features + self._shift)
      previous_parameters.append(suggestion.parameters)
      suggestion.parameters = parameters[0]
    self._exptr.evaluate(suggestions)
    for parameters, suggestion in zip(previous_parameters, suggestions):
      suggestion.parameters = parameters
