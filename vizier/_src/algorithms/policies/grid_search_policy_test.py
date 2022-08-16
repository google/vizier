"""Tests for vizier.pythia.policies.random_policy."""
from vizier import pythia
from vizier import pyvizier
from vizier._src.algorithms.policies import grid_search_policy
from absl.testing import absltest


class GridSearchPolicyTest(absltest.TestCase):
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

    self.policy_supporter = pythia.InRamPolicySupporter(self.study_config)
    self.policy = grid_search_policy.GridSearchPolicy(
        policy_supporter=self.policy_supporter)
    super().setUp()

  def test_make_grid_values(self):
    grid_values = grid_search_policy.make_grid_values(self.study_config)
    self.assertLen(grid_values['double'], grid_search_policy.GRID_RESOLUTION)
    self.assertLen(grid_values['categorical'], 3)
    self.assertLen(grid_values['discrete'], 3)
    self.assertLen(grid_values['int'], 5)

  def test_make_grid_search_parameters(self):
    """Tests the internal grid search parameters function."""
    parameter_dict = grid_search_policy.make_grid_search_parameters(
        [0], self.study_config)[0]
    self.assertLen(parameter_dict, 4)

  def test_make_suggestions(self):
    """Tests random parameter generation wrapped around Policy."""

    # Total size of StudyConfig search space.
    num_suggestions = grid_search_policy.GRID_RESOLUTION * 3 * 3 * 5
    suggest_request = pythia.SuggestRequest(
        study_descriptor=self.policy_supporter.study_descriptor(),
        count=num_suggestions)

    decisions = self.policy.suggest(suggest_request)
    self.assertLen(decisions.suggestions, num_suggestions)
    for suggestion in decisions.suggestions:
      self.assertLen(suggestion.parameters, 4)

    distinct_suggestions = set([
        tuple(suggestion.parameters.as_dict().values())
        for suggestion in decisions.suggestions
    ])

    self.assertLen(distinct_suggestions, num_suggestions)
    self.assertFalse(decisions.metadata)

  def test_make_early_stopping_decisions(self):
    """Checks if all ACTIVE/PENDING trials become completed in order."""
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
