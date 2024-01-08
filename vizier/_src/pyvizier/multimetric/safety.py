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

"""Multimetric utility classes for Studies/Trials."""

from typing import Iterable, Sequence
import numpy as np
from vizier import pyvizier


class SafetyChecker:
  """Returns the safety status of Measurements/Trials."""

  def __init__(self, metrics_config: pyvizier.MetricsConfig):
    """Initializes with metric information (can have non-safety info).

    Example usage from a study:

      pyconfig = pyvizier.StudyConfig.from_proto(study.study_config)
      checker = multimetric_util.SafetyChecker(pyconfig.metric_information)
      are_trials_safe = checker.are_trials_safe(
              converters.TrialConverter.from_protos(study.trials))
    Args:
      metrics_config: MetricsConfig for a Study.
    """
    self._objective_metrics = list(
        metrics_config.of_type(pyvizier.MetricType.OBJECTIVE)
    )
    self._safety_metrics = list(
        metrics_config.of_type(pyvizier.MetricType.SAFETY))

  def are_trials_safe(self, trials: Iterable[pyvizier.Trial]) -> Iterable[bool]:
    """Returns whether the final measurements of Trials are safe.

    Args:
      trials: Iterable of pyvizier Trials.

    Returns:
      Iterable of whether Trials are safe. Safety is assumed if no final
      measurement is found or safety metric is not found in final measurement.
    """
    return self.are_measurements_safe([
        t.final_measurement if t.final_measurement else pyvizier.Measurement()
        for t in trials
    ])

  def are_measurements_safe(
      self, measurements: Iterable[pyvizier.Measurement]) -> Iterable[bool]:
    """Returns whether measurements are safe.

    Args:
      measurements: Iterable of pyvizier measurements.

    Returns:
      Iterable of whether measurements are safe. Safety is assumed if safety
      metric is not found in measurement.
    """
    safety_results = []
    for measurement in measurements:
      is_safe = True
      for safety_metric in self._safety_metrics:
        # If safety metric is not in metrics, we assume it is safe.
        if safety_metric.name not in measurement.metrics:
          continue

        # Otherwise, we check if there is a safety violation.
        metric_value = measurement.metrics[safety_metric.name].value
        if safety_metric.goal == pyvizier.ObjectiveMetricGoal.MAXIMIZE:
          is_safe = is_safe and metric_value >= safety_metric.safety_threshold
        else:
          is_safe = is_safe and metric_value <= safety_metric.safety_threshold
      safety_results.append(is_safe)
    return safety_results

  def warp_unsafe_trials(
      self, trials: Sequence[pyvizier.Trial]
  ) -> Sequence[pyvizier.Trial]:
    """Sets the final measurements of unsafe Trials to worst values in place.

    Args:
      trials: Sequence of pyvizier Trials.

    Returns:
      Trials whose final measurements are warped to the worst value (inf/-inf)
      for all OBJECTIVE metrics in unsafe Trials.
    """
    are_safe = self.are_trials_safe(trials)
    objective_goals = {m.name: m.goal for m in self._objective_metrics}
    for is_safe, trial in zip(are_safe, trials):
      if is_safe:
        continue
      if trial.final_measurement is None:
        continue

      for name in trial.final_measurement.metrics.keys():
        if name in objective_goals:
          worst_value = (
              np.inf
              if objective_goals[name] == pyvizier.ObjectiveMetricGoal.MINIMIZE
              else -np.inf
          )
          trial.final_measurement.metrics[name] = pyvizier.Metric(
              value=worst_value
          )

    return trials
