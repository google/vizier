"""Tests for vizier.pythia.policies.random_policy."""
from vizier import pythia
from vizier import pyvizier
from vizier._src.algorithms.policies import random_policy
from absl.testing import absltest


class RandomPolicyTest(absltest.TestCase):
  # TODO: Add conditional test case.

  def setUp(self):
    """Setups up search space."""
    self.study_config = pyvizier.StudyConfig()
    self.study_config.search_space.select_root().add_float_param(
        'double', min_value=-1.0, max_value=1.0)
    self.study_config.search_space.select_root().add_categorical_param(
        name='categorical', feasible_values=['a', 'b', 'c'])
    self.study_config.search_space.select_root().add_discrete_param(
        name='discrete', feasible_values=[0.1, 0.3, 0.5])
    self.study_config.search_space.select_root().add_int_param(
        name='int', min_value=1, max_value=5)

    self.policy_supporter = pythia.LocalPolicyRunner(self.study_config)
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

    suggestions = self.policy.suggest(suggest_request)
    self.assertLen(suggestions, num_suggestions)
    for suggestion in suggestions:
      self.assertLen(suggestion.parameters, 4)

  def test_make_early_stopping_decisions(self):
    """Checks if all ACTIVE/PENDING trials become completed in random order."""
    count = 10
    _ = self.policy_supporter.SuggestTrials(self.policy, count=10)

    request_trial_ids = [1, 2]
    trial_ids_stopped = set()
    for _ in range(count):
      request = pythia.EarlyStopRequest(
          study_descriptor=self.policy_supporter.study_descriptor(),
          trial_ids=request_trial_ids)
      early_stop_decisions = self.policy.early_stop(request)
      self.assertContainsSubset(
          request_trial_ids, [decision.id for decision in early_stop_decisions])

      for decision in early_stop_decisions:
        if decision.should_stop:
          # Stop trials that need to be stopped.
          self.policy_supporter.trials[decision.id - 1].complete(
              pyvizier.Measurement())
          trial_ids_stopped.add(decision.id)
    self.assertEqual(trial_ids_stopped, set(range(1, count + 1)))


if __name__ == '__main__':
  absltest.main()
