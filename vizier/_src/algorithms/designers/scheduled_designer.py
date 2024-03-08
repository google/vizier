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

"""Scheduled designer for budget-aware optimization.
"""

import abc
import logging
from typing import Protocol, Sequence
import attrs
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz

ParamsValues = dict[str, float]


@attrs.define
class ScheduledParam(abc.ABC):
  """Represents a scheduled parameter.

  The scheduling starts from `init_value` (the first step) and ends at
  `final_value` (last step).

  The `value` method returns the value of the scheduled param in the current
  step, accounting for the total number of steps in the scheduling process.

  Note that some types of scheduling (e.g. random, cyclic) are not supported.
  """

  init_value: float
  final_value: float

  def __attrs_post_init__(self):
    self._validate()

  def _validate(self) -> None:
    """Validate the scheduling param initial and final values."""
    total_steps = 10  # arbitrarily choose the total steps for validation.
    tol = 1e-5

    if abs(self.value(total_steps, 0) - self.init_value) > tol:
      raise ValueError("Initial value mismatch (step 0).")

    if abs(self.value(total_steps, total_steps - 1) - self.final_value) > tol:
      raise ValueError("Final value mismatch (last step).")

  @abc.abstractmethod
  def value(self, total_steps: int, step: int) -> float:
    """Compute the scheduling param value in the current step.

    Args:
      total_steps: The total number of steps for which the scheduling applied to
        (e.g. total number of study trials).
      step: The current step in the scheduling process (e.g. current study
        trial).
    """


@attrs.define
class LinearScheduledParam(ScheduledParam):
  """Linear scheduler."""

  def value(self, total_steps: int, step: int) -> float:
    slope = (self.final_value - self.init_value) / (total_steps - 1)
    return self.init_value + slope * step


@attrs.define
class ExponentialScheduledParam(ScheduledParam):
  """Exponential scheduler."""

  rate: float

  def value(self, total_steps: int, step: int) -> float:
    alpha = 1 / self.rate
    beta = -np.log(self.final_value / self.init_value) / (
        (total_steps - 1) ** alpha
    )
    return self.init_value * np.exp(-beta * step**alpha)


class DesignerStateUpdater(Protocol):
  """Update designer state based on params.

  Allows to efficiently modify the designer state without re-instantiating it.
  """

  def __call__(self, designer: vza.Designer, params: ParamsValues) -> None:
    """Update the designer state (inplace) based on parameter values.

    Note: The `params` dictionary does not need to directly correspond to
      `designer` attributes. The protocol implementations have the flexibility
      to determine how to use the `params`.

    Args:
      designer: The designer to update its state inplace.
      params: A dictionary of params to be used to update the designer state.
    """
    pass


# TODO: serialize the designer.
@attrs.define(auto_attribs=False)
class ScheduledDesigner(vza.Designer):
  """Scheduled designer."""

  _problem: vz.ProblemStatement = attrs.field(kw_only=False)
  _designer_factory: vza.DesignerFactory = attrs.field(kw_only=True)
  _designer_state_updater: DesignerStateUpdater = attrs.field(kw_only=True)
  _scheduled_params: dict[str, ScheduledParam] = attrs.field(kw_only=True)
  # The total number of study trials the designer is expected to generate. This
  # is used to determine the scheduling rate of change across the study.
  _expected_total_num_trials: int = attrs.field(
      kw_only=True, validator=attrs.validators.ge(0)
  )
  # ------------------------------------------------------------------
  # Internal attributes which should not be set by callers.
  # ------------------------------------------------------------------
  _designer: vza.Designer = attrs.field(init=False)
  _suggested_num_trials: int = attrs.field(init=False, default=0)
  _metadata_ns: str = attrs.field(default="scheduled_designer", init=False)

  def __attrs_post_init__(self):
    self._designer = self._designer_factory(self._problem)
    self._update_designer_state()

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    """Update the underlying designer based on completed and pending trials."""
    self._designer.update(completed, all_active)

  def _update_designer_state(self) -> ParamsValues:
    """Efficiently update the underlying designer state (inplace).

    Returns:
      The current scheduled parameters values, which is used for logging.
    """
    params_values = {}
    for name, scheduled_param in self._scheduled_params.items():
      params_values[name] = scheduled_param.value(
          total_steps=self._expected_total_num_trials,
          step=self._suggested_num_trials,
      )
    self._designer_state_updater(self._designer, params_values)
    logging.info("Updated designer state with params: %s", params_values)
    return params_values

  def suggest(self, count: int = 1) -> Sequence[vz.TrialSuggestion]:
    """Suggest trials."""
    # Compute the scheduled param values and update the designer state.
    params_values = self._update_designer_state()
    # Suggest 'count' trials in batch, using the same scheduled params values.
    suggest_trials = self._designer.suggest(count)
    self._suggested_num_trials += len(suggest_trials)
    # Log the parameter values in the suggested trials for debugging.
    for trial in suggest_trials:
      metadata = trial.metadata.ns(self._metadata_ns).ns("devinfo")
      for name in self._scheduled_params.keys():
        metadata[name] = str(params_values[name])
    # Check that the maximum number of trials hasn't surpassed.
    if self._suggested_num_trials >= self._expected_total_num_trials:
      logging.warning(
          "Suggested trial count (%s) exceeded the configured maximum (%s).",
          self._suggested_num_trials,
          self._expected_total_num_trials,
      )
    return suggest_trials
