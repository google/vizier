"""Experimenter that shifts the values of the parameters in Evaluation."""

import copy
import logging
from typing import Sequence

from absl import logging
import numpy as np
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter
from vizier.pyvizier import converters


class ShiftingExperimenter(experimenter.Experimenter):
  """ShiftingExperimenter shifts the parameters of suggestions in Evaluate."""

  def __init__(self, exptr: experimenter.Experimenter, shift: np.ndarray):
    """ShiftingExperiment shifts parameter values before passing to exptr.

    Currently only supports flat double search spaces.

    Args:
      exptr: Underlying experimenter to be wrapped.
      shift: Shift that broadcasts to array of shape (dimension,).

    Raises:
      ValueError: Non-positive dimension or non-bradocastable shift.
    """
    self._exptr = exptr
    self._problem_statement = self._exptr.problem_statement()

    if self._problem_statement.search_space.is_conditional:
      raise ValueError('Search space should not have conditional'
                       f' parameters {self._problem_statement}')
    dimension = len(self._problem_statement.search_space.parameters)
    if dimension <= 0:
      raise ValueError(f'Invalid dimension: {dimension}')
    try:
      # Attempts a broadcast to check broadcasting.
      np.broadcast(shift, np.zeros(dimension))
      self._shift = shift
    except ValueError as broadcast_err:
      raise ValueError(
          f'Shift {shift} is not broadcastable for dim: {dimension}.'
          '\n') from broadcast_err

    for parameter in self._problem_statement.search_space.parameters:
      if parameter.type != pyvizier.ParameterType.DOUBLE:
        raise ValueError(f'Non-double parameters {parameter}')
      if parameter.bounds is not None:
        logging.warning(
            'Bounds %s may need to be relaxed'
            'and affected by shift.', parameter.bounds)

    self._converter = converters.TrialToArrayConverter.from_study_config(
        study_config=self._problem_statement, scale=False)

  def problem_statement(self) -> pyvizier.ProblemStatement:
    return copy.deepcopy(self._problem_statement)

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
