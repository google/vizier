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


def assert_passes_on_random_multi_metric_function(
    self,
    search_space: vz.SearchSpace,
    optimizer: vza.GradientFreeOptimizer,
    *,
    np_random_seed: int
):
  """Bi-objective test on random score."""
  rng = np.random.default_rng(np_random_seed)

  logging.info('search space: %s', search_space)

  problem = vz.ProblemStatement(
      search_space=search_space,
      metric_information=[
          vz.MetricInformation(
              'acquisition_1', goal=vz.ObjectiveMetricGoal.MAXIMIZE
          ),
          vz.MetricInformation(
              'acquisition_2', goal=vz.ObjectiveMetricGoal.MAXIMIZE
          ),
      ],
  )

  def mock_score(trials):
    return {
        'acquisition_1': rng.uniform(size=[len(trials), 1]),
        'acquisition_2': rng.uniform(size=[len(trials), 1]),
    }

  suggestions = optimizer.optimize(mock_score, problem, count=5)
  self.assertNotEmpty(suggestions)

  logging.info('suggestions: %s', suggestions)
  for suggestion in suggestions:
    problem.search_space.assert_contains(suggestion.parameters)
