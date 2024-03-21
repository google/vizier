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

"""Scheduled GP-Bandit for budget-aware optimization.
"""

from typing import Callable
import attrs
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import gp_bandit
from vizier._src.algorithms.designers import scheduled_designer
from vizier._src.algorithms.designers.gp import acquisitions


@attrs.define(kw_only=True)
class ScheduledGPBanditFactory:
  """Scheduled GP-Bandit factory."""

  _gp_bandit_factory: Callable[[vz.ProblemStatement], gp_bandit.VizierGPBandit]
  _init_ucb_coefficient: float
  _final_ucb_coefficient: float
  _decay_ucb_coefficient: float
  _expected_total_num_trials: int

  def __call__(
      self,
      problem: vz.ProblemStatement,
  ) -> scheduled_designer.ScheduledDesigner:
    """Creates a scheduled GP-Bandit designer."""

    def _gp_bandit_state_updater(designer, params):
      designer._scoring_function_factory = (  # pylint: disable=protected-access
          acquisitions.bayesian_scoring_function_factory(
              lambda _: acquisitions.UCB(params['ucb_coefficient'])
          )
      )

    ucb_coef_param = scheduled_designer.ExponentialScheduledParam(
        init_value=self._init_ucb_coefficient,
        final_value=self._final_ucb_coefficient,
        rate=self._decay_ucb_coefficient,
    )

    return scheduled_designer.ScheduledDesigner(
        problem,
        designer_factory=self._gp_bandit_factory,
        designer_state_updater=_gp_bandit_state_updater,
        scheduled_params={'ucb_coefficient': ucb_coef_param},
        expected_total_num_trials=self._expected_total_num_trials,
    )
