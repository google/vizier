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

import random
from typing import Sequence
from absl import logging
import attrs
import jax
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import gp_bandit
from vizier._src.algorithms.designers.gp import acquisitions


@attrs.define(auto_attribs=False)
class VizierScheduledGPBandit(vza.Designer):
  """Scheduled GP-Bandit.

  Attributes:
    problem: Must be a flat study with a single metric.
    max_num_trials: The designer budget in trials.
    rng: If not set, uses random numbers.
    init_ucb_coef: The initial UCB coefficient.
    final_ucb_coef: The final UCB coefficient.
    decay_rate: The decay rate.
    metadata_ns: Metadata namespace that this designer writes to.
  """

  _problem: vz.ProblemStatement = attrs.field(kw_only=False)
  _max_num_trials: int = attrs.field(
      kw_only=True, validator=attrs.validators.ge(0)
  )
  _rng: jax.Array = attrs.field(
      factory=lambda: jax.random.PRNGKey(random.getrandbits(32)), kw_only=True
  )
  _init_ucb_coef: float = attrs.field(default=3.26359, kw_only=True)
  _final_ucb_coef: float = attrs.field(default=0.01577, kw_only=True)
  _decay_rate: float = attrs.field(
      default=1.02390, kw_only=True, validator=attrs.validators.ge(0)
  )
  # ------------------------------------------------------------------
  # Internal attributes which should not be set by callers.
  # ------------------------------------------------------------------
  _gp_bandit_designer: gp_bandit.VizierGPBandit = attrs.field(init=False)
  _trials_count: int = attrs.field(init=False, default=0)
  _metadata_ns: str = attrs.field(
      default="oss_scheduled_gp_bandit", kw_only=True, init=False
  )

  def __attrs_post_init__(self):
    self._gp_bandit_designer = gp_bandit.VizierGPBandit(
        self._problem, rng=self._rng
    )

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    """Update the designer based on completed and pending trials."""
    self._trials_count += len(completed.trials)
    self._gp_bandit_designer.update(completed, all_active)

  def _compute_ucb_coefficient(self):
    """Compute the UCB coefficient."""
    alpha = 1 / self._decay_rate
    a = self._init_ucb_coef
    b = -np.log(self._final_ucb_coef / self._init_ucb_coef) / (
        (self._max_num_trials - 1) ** alpha
    )
    return a * np.exp(-b * self._trials_count**alpha)

  def suggest(self, count: int = 1) -> Sequence[vz.TrialSuggestion]:
    """Suggest trials."""
    ucb_coef = self._compute_ucb_coefficient()
    logging.info(
        "UCB coefficient (trials count = %s): %s", self._trials_count, ucb_coef
    )
    self._gp_bandit_designer._scoring_function_factory = (  # pylint: disable=protected-access
        acquisitions.bayesian_scoring_function_factory(
            lambda _: acquisitions.UCB(ucb_coef)
        )
    )
    suggest_trials = self._gp_bandit_designer.suggest(count)
    for trial in suggest_trials:
      metadata = trial.metadata.ns(self._metadata_ns).ns("devinfo")
      metadata["ucb_coefficient"] = str(ucb_coef)
    return suggest_trials
