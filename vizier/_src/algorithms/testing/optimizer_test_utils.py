"""Tests for gradient-free optimizers."""

from absl import logging
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz


def assert_passes_on_random_single_metric_function(
    self, search_space: vz.SearchSpace, optimizer: vza.GradientFreeOptimizer, *,
    np_random_seed: int):
  """Smoke test on random score."""
  rng = np.random.default_rng(np_random_seed)

  logging.info('search space: %s', search_space)

  problem = vz.ProblemStatement(
      search_space=search_space,
      metric_information=[
          vz.MetricInformation(
              'acquisition', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
      ])

  def mock_score(trials):
    return {'acquisition': rng.uniform(size=[len(trials), 1])}

  suggestions = optimizer.optimize(mock_score, problem, count=5)
  self.assertNotEmpty(suggestions)

  logging.info('suggestions: %s', suggestions)
  for suggestion in suggestions:
    problem.search_space.assert_contains(suggestion.parameters)
