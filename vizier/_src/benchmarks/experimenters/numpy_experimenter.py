"""Numpy experimenter for wrapping deterministic functions on ndarrays."""

import copy
import logging
import math
from typing import Callable, List, Sequence

import numpy as np
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter
from vizier.pyvizier import converters


def _get_name(f):
  """Gets the name of underlying objects."""
  if hasattr(f, '__name__'):
    return f.__name__
  # Next clause handles functools.partial objects.
  if hasattr(f, 'func') and hasattr(f.func, '__name__'):
    return f.func.__name__
  return repr(f)


class NumpyExperimenter(experimenter.Experimenter):
  """NumpyExperimenters take a deterministic function on ndarrays."""

  def __init__(self, impl: Callable[[np.ndarray], float],
               problem_statement: pyvizier.ProblemStatement):
    """NumpyExperimenter with analytic function impl for one metric.

    NumpyExperimenter only supports single objectives, and flat numeric search
    spaces.

    Args:
      impl: Function that scalarizes np.ndarray of shape (dimension,).
      problem_statement: Problem statement.

    Raises:
      ValueError: Non-positive dimension or invalid problem statement.
    """
    dimension = len(problem_statement.search_space.parameters)
    logging.info('Initializing NumpyExperimenter with impl=%s, dimension=%s',
                 _get_name(impl), dimension)
    if dimension <= 0:
      raise ValueError(f'Invalid dimension: {dimension}')
    self.impl = impl

    if not problem_statement.metric_information.is_single_objective:
      raise ValueError(
          f'Statement should be single objective {problem_statement}')
    if problem_statement.search_space.is_conditional:
      raise ValueError(f'Statement should be flat {problem_statement}')
    for parameter in problem_statement.search_space.parameters:
      if not parameter.type.is_numeric():
        raise ValueError(f'Non-numeric parameters {parameter}')

    self._metric_name = problem_statement.metric_information.of_type(
        pyvizier.MetricType.OBJECTIVE).item().name
    self._problem_statement = copy.deepcopy(problem_statement)
    self._converter = converters.TrialToArrayConverter.from_study_config(
        study_config=self._problem_statement,
        scale=False,
        flip_sign_for_minimization_metrics=False)

  def problem_statement(self) -> pyvizier.ProblemStatement:
    return copy.deepcopy(self._problem_statement)

  def evaluate(self,
               suggestions: Sequence[pyvizier.Trial]) -> List[pyvizier.Trial]:
    completed_trials = list(copy.deepcopy(suggestions))
    # Features has shape (num_trials, num_features).
    features = self._converter.to_features(completed_trials)
    for idx, completed_trial in enumerate(completed_trials):
      val = self.impl(features[idx])
      if math.isfinite(val):
        completed_trial.complete(
            pyvizier.Measurement(metrics={self._metric_name: val}))
      else:
        # TODO: Add infeasibility to completed_trial.
        completed_trial.complete(pyvizier.Measurement())
      if not completed_trial.is_completed:
        raise RuntimeError(f'Trial {completed_trial} not completed')
    return completed_trials

  def __repr__(self) -> str:
    return f'NumpyExperimenter {{name: {_get_name(self.impl)}'
