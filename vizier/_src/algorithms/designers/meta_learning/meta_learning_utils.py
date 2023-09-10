# Copyright 2023 Google LLC.
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

"""Meta learning helper functionality."""

import attrs
from vizier import pyvizier as vz


@attrs.define(kw_only=True)
class MetaLearningUtils:
  """Meta learning utils class."""

  # The objective goal of 'meta' and 'tuned' algorithms.
  _goal: vz.ObjectiveMetricGoal
  # The metric name associated with the 'meta' designer trials.
  _meta_metric_name: str
  # The 'tuned' algorithms params to be tuned (meta designer's search space).
  _tuning_params: vz.SearchSpace

  @property
  def meta_problem(self) -> vz.ProblemStatement:
    """Create meta problem."""
    problem = vz.ProblemStatement(search_space=self._tuning_params)
    problem.metric_information = vz.MetricsConfig(
        metrics=[vz.MetricInformation(self._meta_metric_name, goal=self._goal)]
    )
    return problem

  def complete_meta_suggestion(
      self,
      meta_suggestion: vz.TrialSuggestion,
      score: float,
  ) -> vz.Trial:
    """Complete meta trial suggestion and assign score."""
    meta_trial = meta_suggestion.to_trial()
    meta_trial.complete(vz.Measurement(metrics={self._meta_metric_name: score}))
    return meta_trial

  def get_default_hyperparameters(self) -> vz.TrialSuggestion:
    """Generate trial suggestion with default tuned designer parameter values."""
    suggestion = vz.TrialSuggestion()
    for param in self.meta_problem.search_space.parameters:
      if param.default_value is None:
        raise ValueError(
            "Hyper-param (%s) doesn't have default value!" % param.name
        )
      else:
        suggestion.parameters[param.name] = param.default_value
    return suggestion

  def _metric_name(self, trial: vz.Trial) -> str:
    """Return the metric name."""
    if (
        trial.final_measurement is None
        or len(trial.final_measurement.metrics) != 1
    ):
      raise ValueError("There should exactly one final measurement metric.")
    return next(iter(trial.final_measurement.metrics))

  def get_best_trial_score(self, trials: list[vz.Trial]) -> float:
    """Return the best trial score."""
    best_trial = self.get_best_trial(trials)
    if best_trial.final_measurement is None:
      raise ValueError("'final_measurement' is None; this should not happen.")
    return best_trial.final_measurement.metrics[
        self._metric_name(best_trial)
    ].value

  def get_best_trial(self, trials: list[vz.Trial]) -> vz.Trial:
    """Return the best trial with the metric specified by `trial_type`."""
    best_trial = trials[0]
    for trial in trials:
      if self._is_trial_better(trial, best_trial):
        best_trial = trial
    return best_trial

  def _is_trial_better(
      self,
      trial1: vz.Trial,
      trial2: vz.Trial,
  ) -> bool:
    """Returns whether trial1 is better than trial2 in terms of `metric_name`."""
    if self._metric_name(trial1) != self._metric_name(trial2):
      raise ValueError("trial1 and trial2 should have the same metric name.")

    metric_name = self._metric_name(trial1)
    if trial1.final_measurement is None or trial2.final_measurement is None:
      raise ValueError("Both trial1 and trial2 should have final measurements.")
    if trial1.infeasible:
      return False
    if trial2.infeasible:
      return True
    val1 = trial1.final_measurement.metrics[metric_name].value
    val2 = trial2.final_measurement.metrics[metric_name].value
    if self._goal == vz.ObjectiveMetricGoal.MAXIMIZE and val1 > val2:
      return True
    elif self._goal == vz.ObjectiveMetricGoal.MINIMIZE and val1 < val2:
      return True
    else:
      return False
