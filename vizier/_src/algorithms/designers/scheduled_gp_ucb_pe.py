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


@attrs.define(kw_only=True)
class ScheduledGPUCBPEFactory:
  """Scheduled GP-UCB-PE factory."""

  _gp_ucb_pe_factory: Callable[
      [vz.ProblemStatement], gp_ucb_pe.VizierGPUCBPEBandit
  ]
  _expected_total_num_trials: int
  # ucb_coefficient
  _init_ucb_coefficient: float
  _final_ucb_coefficient: float
  _decay_ucb_coefficient: float

  # explore_region_ucb_coefficient
  _init_explore_region_ucb_coefficient: float
  _final_explore_region_ucb_coefficient: float
  _decay_explore_region_ucb_coefficient: float

  # ucb_overwrite_probability
  _init_ucb_overwrite_probability: float
  _final_ucb_overwrite_probability: float
  _decay_ucb_overwrite_probability: float

  def __call__(
      self,
      problem: vz.ProblemStatement,
  ) -> scheduled_designer.ScheduledDesigner:
    """Creates a scheduled GP-Bandit designer."""

    def _gp_ucb_pe_state_updater(designer, params):
      # Create a new copy of the config and replace the ucb coefficient value.
      updated_config = dc.replace(
          designer._config,  # pylint: disable=protected-access
          ucb_coefficient=params['ucb_coefficient'],
          explore_region_ucb_coefficient=params[
              'explore_region_ucb_coefficient'
          ],
          ucb_overwrite_probability=params['ucb_overwrite_probability'],
      )
      # Update the designer config. As the GP_UCB_PE state is recomputed every
      # suggest() call based on the config, this effectively updates the state.
      # TODO: Instead of changing internal state, use public method.
      designer._config = updated_config  # pylint: disable=protected-access

    ucb_coefficient_param = scheduled_designer.ExponentialScheduledParam(
        init_value=self._init_ucb_coefficient,
        final_value=self._final_ucb_coefficient,
        rate=self._decay_ucb_coefficient,
    )

    explore_region_ucb_coefficient_param = (
        scheduled_designer.ExponentialScheduledParam(
            init_value=self._init_explore_region_ucb_coefficient,
            final_value=self._final_explore_region_ucb_coefficient,
            rate=self._decay_explore_region_ucb_coefficient,
        )
    )

    ucb_overwrite_probability_param = (
        scheduled_designer.ExponentialScheduledParam(
            init_value=self._init_ucb_overwrite_probability,
            final_value=self._final_ucb_overwrite_probability,
            rate=self._decay_ucb_overwrite_probability,
        )
    )

    scheduled_params = {
        'ucb_coefficient': ucb_coefficient_param,
        'explore_region_ucb_coefficient': explore_region_ucb_coefficient_param,
        'ucb_overwrite_probability': ucb_overwrite_probability_param,
    }

    return scheduled_designer.ScheduledDesigner(
        problem,
        designer_factory=self._gp_ucb_pe_factory,
        designer_state_updater=_gp_ucb_pe_state_updater,
        scheduled_params=scheduled_params,
        expected_total_num_trials=self._expected_total_num_trials,
    )
