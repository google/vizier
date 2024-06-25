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

"""Numpy experimenter for wrapping deterministic functions on ndarrays."""

import copy
import logging
import math
from typing import Callable, Sequence

import numpy as np
from vizier import pyvizier as vz
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

  def __init__(
      self,
      impl: Callable[[np.ndarray], float],
      problem_statement: vz.ProblemStatement,
  ):
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
    self._impl_name = _get_name(impl)
    logging.info(
        'Initializing NumpyExperimenter with impl=%s, dimension=%s',
        self._impl_name,
        dimension,
    )
    if dimension <= 0:
      raise ValueError(f'Invalid dimension: {dimension}')
    self._dimension = dimension
    self.impl = impl

    if not problem_statement.metric_information.is_single_objective:
      raise ValueError(
          f'Statement should be single objective {problem_statement}'
      )
    if problem_statement.search_space.is_conditional:
      raise ValueError(f'Statement should be flat {problem_statement}')
    for parameter in problem_statement.search_space.parameters:
      if not parameter.type.is_numeric():
        raise ValueError(f'Non-numeric parameters {parameter}')

    objective_metrics = problem_statement.metric_information.of_type(
        vz.MetricType.OBJECTIVE
    )
    self._metric_name = objective_metrics.item().name

    self._problem_statement = copy.deepcopy(problem_statement)
    self._converter = converters.TrialToArrayConverter.from_study_config(
        study_config=self._problem_statement,
        scale=False,
        flip_sign_for_minimization_metrics=False,
    )

  def problem_statement(self) -> vz.ProblemStatement:
    return copy.deepcopy(self._problem_statement)

  def evaluate(self, suggestions: Sequence[vz.Trial]):
    # Features has shape (num_trials, num_features).
    features = self._converter.to_features(suggestions)
    for idx, suggestion in enumerate(suggestions):
      val = self.impl(features[idx])
      if math.isfinite(val):
        suggestion.complete(vz.Measurement(metrics={self._metric_name: val}))
      else:
        self.problem_statement().search_space.assert_contains(
            suggestions[idx].parameters
        )
        suggestion.complete(
            vz.Measurement(),
            infeasibility_reason='Objective value is not finite: %f' % val,
        )

  def __repr__(self) -> str:
    return f'NumpyExperimenter {{name: {self._impl_name}}}'


class MultiObjectiveNumpyExperimenter(experimenter.Experimenter):
  """Multiobjective variant with variable number of objectives and dimensions.

  We assume that `impl` returns a list of values, one per objective, ordered by
  the metric information in the problem statement.
  """

  def __init__(
      self,
      impl: Callable[[np.ndarray], Sequence[float]],
      problem_statement: vz.ProblemStatement,
  ):
    self._impl = impl
    self._problem_statement = copy.deepcopy(problem_statement)
    self._converter = converters.TrialToArrayConverter.from_study_config(
        study_config=self.problem_statement(),
        scale=False,
        flip_sign_for_minimization_metrics=False,
    )

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    metric_info = self._problem_statement.metric_information
    features = self._converter.to_features(suggestions)
    for i, suggestion in enumerate(suggestions):
      feat = features[i]
      values = self._impl(feat)
      metrics = {mc.name: value for mc, value in zip(metric_info, values)}
      suggestion.complete(vz.Measurement(metrics))

  def problem_statement(self) -> vz.ProblemStatement:
    return copy.deepcopy(self._problem_statement)
