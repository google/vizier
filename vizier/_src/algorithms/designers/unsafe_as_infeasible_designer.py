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

"""Designer that maps unsafe Trials as infeasible."""

import copy
from typing import Optional, Sequence
from absl import logging
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier.pyvizier import multimetric


class UnsafeAsInfeasibleDesigner(vza.Designer):
  """Designer that maps unsafe Trials as infeasible."""

  def __init__(
      self,
      problem_statement: vz.ProblemStatement,
      designer_factory: vza.DesignerFactory[vza.Designer],
      *,
      seed: Optional[int] = None,
  ):
    """Init.

    Args:
      problem_statement:
      designer_factory:
      seed:
    """
    self._safety_checker = multimetric.SafetyChecker(
        problem_statement.metric_information
    )
    problem_statement_safety_removed = copy.deepcopy(problem_statement)
    problem_statement_safety_removed.metric_information = (
        problem_statement.metric_information.exclude_type(vz.MetricType.SAFETY)
    )
    self._designer = designer_factory(
        problem_statement_safety_removed, seed=seed
    )

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    are_safe = self._safety_checker.are_trials_safe(completed.trials)
    mutated_trials = []
    mutated_trial_count = 0
    for is_safe, trial in zip(are_safe, completed.trials):
      if is_safe:
        mutated_trials.append(trial)
      else:
        measurement = trial.final_measurement or vz.Measurement()
        mutated_trial = trial.complete(
            measurement, infeasibility_reason='unsafe', inplace=False
        )
        mutated_trial_count += 1
        mutated_trials.append(mutated_trial)

    logging.info(
        'Update designer with %d completed trials (%d converted from safe to'
        ' infeasible)',
        len(mutated_trials),
        mutated_trial_count,
    )
    self._designer.update(vza.CompletedTrials(mutated_trials), all_active)

  def suggest(
      self, count: Optional[int] = None
  ) -> Sequence[vz.TrialSuggestion]:
    """Make new suggestions.

    Args:
      count: Makes best effort to generate this many suggestions. If None,
        suggests as many as the algorithm wants.

    Returns:
      New suggestions.
    """
    return self._designer.suggest(count)
