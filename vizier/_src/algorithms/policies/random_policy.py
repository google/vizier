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

"""Random Pythia Policy which produces uniform sampling of Trial parameter values.

Since this is a RandomPolicy (i.e. stateless), we don't use the PolicySupporter
API when suggesting trials, but we do for the early stopping in order to
showcase how the policy supporter should be used.
"""
import random
from vizier import pythia
from vizier import pyvizier
from vizier._src.algorithms.designers import random as random_designer_lib


class RandomPolicy(pythia.Policy):
  """A policy that picks random hyper-parameter values."""

  def __init__(self, policy_supporter: pythia.PolicySupporter):
    self._policy_supporter = policy_supporter

  def suggest(self, request: pythia.SuggestRequest) -> pythia.SuggestDecision:
    """Gets number of Trials to propose, and produces random Trials."""
    suggestions = random_designer_lib.RandomDesigner(
        request.study_config.search_space
    ).suggest(request.count)
    return pythia.SuggestDecision(suggestions)

  def early_stop(
      self, request: pythia.EarlyStopRequest
  ) -> pythia.EarlyStopDecisions:
    """Selects a random ACTIVE/PENDING trial to stop from datastore."""
    decisions = []

    all_active_trials = self._policy_supporter.GetTrials(
        study_guid=request.study_guid,
        status_matches=pyvizier.TrialStatus.ACTIVE,
    )
    trial_to_stop_id = None
    if all_active_trials:
      trial_to_stop_id = random.choice(all_active_trials).id
      decisions.append(
          pythia.EarlyStopDecision(
              id=trial_to_stop_id, reason='Random early stopping.'
          )
      )

    for trial_id in list(request.trial_ids):
      if trial_id != trial_to_stop_id:
        decisions.append(
            pythia.EarlyStopDecision(
                id=trial_id, reason='Trial should not stop.', should_stop=False
            )
        )

    return pythia.EarlyStopDecisions(decisions, pyvizier.MetadataDelta())
