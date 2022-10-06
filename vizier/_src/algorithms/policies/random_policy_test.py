# Copyright 2022 Google LLC.
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

"""Tests for vizier.pythia.policies.random_policy."""
from vizier import pythia
from vizier import pyvizier
from vizier._src.algorithms.policies import random_policy
from absl.testing import absltest


class RandomPolicyTest(absltest.TestCase):
  # TODO: Add conditional test case.

  def setUp(self):
    """Setups up search space."""
    self.study_config = pyvizier.ProblemStatement()
    self.study_config.search_space.root.add_float_param(
        name='double', min_value=-1.0, max_value=1.0)
    self.study_config.search_space.root.add_categorical_param(
        name='categorical', feasible_values=['a', 'b', 'c'])
    self.study_config.search_space.root.add_discrete_param(
        name='discrete', feasible_values=[0.1, 0.3, 0.5])
    self.study_config.search_space.root.add_int_param(
        name='int', min_value=1, max_value=5)

    self.policy_supporter = pythia.InRamPolicySupporter(self.study_config)
    self.policy = random_policy.RandomPolicy(
        policy_supporter=self.policy_supporter)
    super().setUp()

  def test_make_random_parameters(self):
    """Tests the internal random parameters function."""
    parameter_dict = random_policy.make_random_parameters(self.study_config)
    self.assertLen(parameter_dict, 4)

  def test_make_suggestions(self):
    """Tests random parameter generation wrapped around Policy."""
    num_suggestions = 5
    suggest_request = pythia.SuggestRequest(
        study_descriptor=self.policy_supporter.study_descriptor(),
        count=num_suggestions)

    decisions = self.policy.suggest(suggest_request)
    self.assertLen(decisions.suggestions, num_suggestions)
    self.assertFalse(decisions.metadata)
    for suggestion in decisions.suggestions:
      self.assertLen(suggestion.parameters, 4)

  def test_make_early_stopping_decisions(self):
    """Checks if all ACTIVE/PENDING trials become completed in random order."""
    count = 10
    _ = self.policy_supporter.SuggestTrials(self.policy, count=count)

    request_trial_ids = [1, 2]
    trial_ids_stopped = set()
    for _ in range(count):
      request = pythia.EarlyStopRequest(
          study_descriptor=self.policy_supporter.study_descriptor(),
          trial_ids=request_trial_ids)
      early_stop_decisions = self.policy.early_stop(request)
      self.assertContainsSubset(
          request_trial_ids,
          [decision.id for decision in early_stop_decisions.decisions])

      for decision in early_stop_decisions.decisions:
        if decision.should_stop:
          # Stop trials that need to be stopped.
          self.policy_supporter.trials[decision.id - 1].complete(
              pyvizier.Measurement())
          trial_ids_stopped.add(decision.id)
    self.assertEqual(trial_ids_stopped, set(range(1, count + 1)))


if __name__ == '__main__':
  absltest.main()
