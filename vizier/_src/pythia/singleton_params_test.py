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

"""Tests for singleton_params module which allows to handle singleton parameters."""

from vizier import pyvizier as vz
from vizier._src.pythia import singleton_params
from absl.testing import absltest
from absl.testing import parameterized


def get_problem_with_singletons() -> vz.ProblemStatement:
  """Returns a problem with singleton parameters."""
  problem = vz.ProblemStatement()
  problem.search_space.root.add_float_param('sf', min_value=1, max_value=1)
  problem.search_space.root.add_int_param('si', min_value=5, max_value=5)
  problem.search_space.root.add_categorical_param('sc', feasible_values=['a'])
  problem.search_space.root.add_discrete_param('sd', feasible_values=[3])

  problem.search_space.root.add_float_param('f', min_value=0.0, max_value=5.0)
  problem.search_space.root.add_int_param('i', min_value=0, max_value=10)
  problem.search_space.root.add_categorical_param(
      'c', feasible_values=['a', '1']
  )
  problem.search_space.root.add_discrete_param('d', feasible_values=[3, 1])
  problem.metric_information = [
      vz.MetricInformation(name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
  ]
  return problem


def get_problem_without_singletons() -> vz.ProblemStatement:
  """Returns a problem with singleton parameters."""
  problem = vz.ProblemStatement()
  problem.search_space.root.add_float_param('f', min_value=0.0, max_value=5.0)
  problem.search_space.root.add_int_param('i', min_value=0, max_value=10)
  problem.search_space.root.add_categorical_param(
      'c', feasible_values=['a', '1']
  )
  problem.search_space.root.add_discrete_param('d', feasible_values=[3, 1])
  problem.metric_information = [
      vz.MetricInformation(name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
  ]
  return problem


class SingletonParameterHandlerTest(parameterized.TestCase):
  """Tests for the `singletonParameterHandler`."""

  @parameterized.named_parameters(
      ('with_singletons', get_problem_with_singletons()),
      ('without_singletons', get_problem_without_singletons()),
  )
  def test_stripped_problem(self, problem):
    handler = singleton_params.SingletonParameterHandler(problem)
    expected_search_search = get_problem_without_singletons().search_space
    self.assertEqual(
        handler.stripped_problem.search_space, expected_search_search
    )

  def test_strip_trials(self):
    problem = get_problem_with_singletons()
    handler = singleton_params.SingletonParameterHandler(problem)
    trials = [
        vz.TrialSuggestion(
            parameters={
                'sf': 1.0,
                'si': 5,
                'sc': 'a',
                'sd': 3,
                'f': 1.0,
                'i': 1,
                'c': 'a',
                'd': 3,
            }
        )
    ]
    stripped_trials = handler.strip_trials(trials)
    self.assertEqual(
        stripped_trials,
        [
            vz.TrialSuggestion(
                parameters={
                    'f': 1.0,
                    'i': 1,
                    'c': 'a',
                    'd': 3,
                }
            )
        ],
    )

  def test_augment_trials(self):
    problem = get_problem_with_singletons()
    handler = singleton_params.SingletonParameterHandler(problem)
    trials = [
        vz.TrialSuggestion(
            parameters={
                'f': 1.0,
                'i': 1,
                'c': 'a',
                'd': 3,
            }
        )
    ]
    augmented_trials = handler.augment_trials(trials)
    self.assertEqual(
        augmented_trials,
        [
            vz.TrialSuggestion(
                parameters={
                    'sf': 1.0,
                    'si': 5,
                    'sc': 'a',
                    'sd': 3,
                    'f': 1.0,
                    'i': 1,
                    'c': 'a',
                    'd': 3,
                }
            )
        ],
    )

  def test_find_singletons(self):
    problem = get_problem_with_singletons()
    handler = singleton_params.SingletonParameterHandler(problem)
    self.assertEqual(
        handler._singletons,
        {
            'sf': 1.0,
            'si': 5,
            'sc': 'a',
            'sd': 3,
        },
    )

  def test_apply_strip_multiple_times(self):
    problem = get_problem_with_singletons()
    handler = singleton_params.SingletonParameterHandler(problem)
    trials = [
        vz.TrialSuggestion(
            parameters={
                'sf': 1.0,
                'si': 5,
                'sc': 'a',
                'sd': 3,
                'f': 1.0,
                'i': 1,
                'c': 'a',
                'd': 3,
            }
        )
    ]
    for _ in range(5):
      trials = handler.strip_trials(trials)
    self.assertEqual(
        trials,
        [
            vz.TrialSuggestion(
                parameters={
                    'f': 1.0,
                    'i': 1,
                    'c': 'a',
                    'd': 3,
                }
            )
        ],
    )


if __name__ == '__main__':
  absltest.main()
