"""Tests for vizier.pythia.policies.random_policy."""
from vizier.pythia import base
from vizier.pythia.policies import random_policy
from vizier.pyvizier import oss as pyvizier
from vizier.pyvizier import pythia

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

    self.study_descriptor = pythia.StudyDescriptor(
        config=self.study_config.to_pythia())
    self.policy = random_policy.RandomPolicy()
    super().setUp()

  def test_make_random_parameters(self):
    """Tests the internal random parameters function."""
    parameter_dict = random_policy.make_random_parameters(self.study_config)
    self.assertLen(parameter_dict, 4)

  def test_make_suggestions(self):
    """Generates random parameters for trials."""
    num_suggestions = 5
    suggest_request = base.SuggestRequest(
        study_descriptor=self.study_descriptor, count=num_suggestions)

    suggestions = self.policy.suggest(suggest_request)
    self.assertLen(suggestions, num_suggestions)
    for suggestion in suggestions:
      self.assertLen(suggestion.parameters, 4)

  def test_make_early_stopping_decisions(self):
    """Selects a random ACTIVE trial to stop."""
    request = base.EarlyStopRequest(
        study_descriptor=self.study_descriptor, trial_ids=[1, 2])
    early_stop_decisions = self.policy.early_stop(request)
    self.assertLen(early_stop_decisions, 1)


if __name__ == '__main__':
  absltest.main()
