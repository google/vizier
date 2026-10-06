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

"""Tests for vizier.pythia.policies.random_policy."""
from vizier import pythia
from vizier import pyvizier
from vizier._src.algorithms.policies import random_policy
from vizier.testing import test_studies

from absl.testing import absltest


class RandomPolicyTest(absltest.TestCase):
  # TODO: Add conditional test case.

  def setUp(self):
    """Setups up search space and policy."""
    self.study_config = pyvizier.ProblemStatement(
        test_studies.flat_space_with_all_types()
    )
    self.policy_supporter = pythia.InRamPolicySupporter(self.study_config)
    self.policy = random_policy.RandomPolicy(
        policy_supporter=self.policy_supporter
    )
    super().setUp()

  def test_make_suggestions(self):
    """Tests random parameter generation wrapped around Policy."""
    num_suggestions = 5
    num_params = len(self.study_config.search_space.parameters)

    suggest_request = pythia.SuggestRequest(
        study_descriptor=self.policy_supporter.study_descriptor(),
        count=num_suggestions,
    )
    decisions = self.policy.suggest(suggest_request)
    self.assertLen(decisions.suggestions, num_suggestions)
    self.assertFalse(decisions.metadata)
    for suggestion in decisions.suggestions:
      self.assertLen(suggestion.parameters, num_params)

  def test_make_early_stopping_decisions(self):
    """Checks if all ACTIVE/PENDING trials become completed in random order."""
    count = 10
    _ = self.policy_supporter.SuggestTrials(self.policy, count=count)

    request_trial_ids = [1, 2]
    trial_ids_stopped = set()
    for _ in range(count):
      request = pythia.EarlyStopRequest(
          study_descriptor=self.policy_supporter.study_descriptor(),
          trial_ids=request_trial_ids,
      )
      early_stop_decisions = self.policy.early_stop(request)
      self.assertContainsSubset(
          request_trial_ids,
          [decision.id for decision in early_stop_decisions.decisions],
      )

      for decision in early_stop_decisions.decisions:
        if decision.should_stop:
          # Stop trials that need to be stopped.
          self.policy_supporter.trials[decision.id - 1].complete(
              pyvizier.Measurement()
          )
          trial_ids_stopped.add(decision.id)
    self.assertEqual(trial_ids_stopped, set(range(1, count + 1)))


if __name__ == '__main__':
  absltest.main()
