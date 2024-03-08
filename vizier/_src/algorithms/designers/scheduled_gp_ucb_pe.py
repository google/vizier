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

"""Scheduled GP-UCB-PE for budget-aware optimization.
"""

import dataclasses as dc
from typing import Callable
import attrs
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import gp_ucb_pe
from vizier._src.algorithms.designers import scheduled_designer


# TODO: Add scheduling to more params.
@attrs.define(kw_only=True)
class ScheduledGPUCBPEFactory:
  """Scheduled GP-UCB-PE factory."""

  _gp_ucb_pe_factory: Callable[
      [vz.ProblemStatement], gp_ucb_pe.VizierGPUCBPEBandit
  ]
  _init_ucb_coef: float
  _final_ucb_coef: float
  _ucb_coef_decay_rate: float
  _expected_total_num_trials: int
  _ucb_coef_param_name: str = attrs.field(default='ucb_coef', init=False)

  def __call__(
      self,
      problem: vz.ProblemStatement,
  ) -> scheduled_designer.ScheduledDesigner:
    """Creates a scheduled GP-Bandit designer."""

    def _gp_ucb_pe_state_updater(designer, params):
      # Create a new copy of the config and replace the ucb coefficient value.
      updated_config = dc.replace(
          designer._config, ucb_coefficient=params[self._ucb_coef_param_name]  # pylint: disable=protected-access
      )
      # Update the designer config. As the GP_UCB_PE state is recomputed every
      # suggest() call based on the config, this effectively updates the state.
      designer._config = updated_config  # pylint: disable=protected-access

    ucb_coef_param = scheduled_designer.ExponentialScheduledParam(
        init_value=self._init_ucb_coef,
        final_value=self._final_ucb_coef,
        rate=self._ucb_coef_decay_rate,
    )

    return scheduled_designer.ScheduledDesigner(
        problem,
        designer_factory=self._gp_ucb_pe_factory,
        designer_state_updater=_gp_ucb_pe_state_updater,
        scheduled_params={self._ucb_coef_param_name: ucb_coef_param},
        expected_total_num_trials=self._expected_total_num_trials,
    )
