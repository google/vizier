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

"""Experimenter that normalizes the range of each metric."""
import copy
from typing import Dict, Sequence

import numpy as np
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter
from vizier.pyvizier import converters


# TODO: Improve normalization by using optimization.
class NormalizingExperimenter(experimenter.Experimenter):
  """Normalizes an Experimenter output via dividing by L1 norm at grid points."""

  def __init__(
      self,
      exptr: experimenter.Experimenter,
      num_normalization_samples: int = 100,
  ):
    """Normalizing experimenter uses a grid to estimate a normalization constant.

    Args:
      exptr: Experimenter to be normalized.
      num_normalization_samples: Number of samples to determine normalization.
    """
    self._exptr = exptr
    self._problem_statement = exptr.problem_statement()

    min_vals: list[float] = []
    max_vals: list[float] = []
    for pc in self._problem_statement.search_space.parameters:
      if pc.type != pyvizier.ParameterType.DOUBLE:
        raise ValueError(f'Only DOUBLE parameters can be normalized: {pc}.')
      min_value, max_value = pc.bounds
      min_vals.append(min_value)
      max_vals.append(max_value)

    grid = np.linspace(
        np.asarray(min_vals), np.asarray(max_vals), num_normalization_samples
    )
    converter = converters.TrialToArrayConverter.from_study_config(
        study_config=self._problem_statement, scale=False
    )
    grid_parameters = converter.to_parameters(grid)
    metrics: Dict[str, list[float]] = {}
    for parameters in grid_parameters:
      trial = pyvizier.Trial(parameters=parameters)
      exptr.evaluate([trial])
      measurement = trial.final_measurement
      for name, metric in (measurement.metrics if measurement else {}).items():
        if name in metrics:
          metrics[name].append(metric.value)
        else:
          metrics[name] = [metric.value]

    self._normalizations: Dict[str, float] = {}
    for name, grid_values in metrics.items():
      normalization = np.mean(np.absolute(np.array(grid_values)))
      if normalization == 0:
        raise ValueError(
            f'Cannot normalize {name} due to nonpositive L1 norm'
            f' with grid values {grid_values}'
        )
      self._normalizations[name] = normalization

  def problem_statement(self) -> pyvizier.ProblemStatement:
    return copy.deepcopy(self._problem_statement)

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    self._exptr.evaluate(suggestions)
    for suggestion in suggestions:
      if suggestion.final_measurement is None:
        continue
      normalized_metrics: Dict[str, pyvizier.Metric] = {}
      for name, metric in suggestion.final_measurement.metrics.items():
        normalized_metrics[name] = pyvizier.Metric(
            value=metric.value / self._normalizations[name]
        )
      suggestion.final_measurement.metrics = normalized_metrics

  def __repr__(self):
    return (
        f'NormalizingExperimenter with normalizations {self._normalizations} on'
        f' {self._exptr}'
    )
