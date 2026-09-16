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

"""Tests for bocs."""
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import harmonica
from vizier._src.algorithms.testing import test_runners
from vizier._src.benchmarks.experimenters import combo_experimenter
from vizier.testing import test_studies

from absl.testing import absltest


class HarmonicaTest(absltest.TestCase):

  def test_make_suggestions(self):
    experimenter = combo_experimenter.IsingExperimenter(lamda=0.01)
    designer = harmonica.HarmonicaDesigner(
        experimenter.problem_statement(), num_init_samples=1)

    num_trials = 10
    trials = test_runners.run_with_random_metrics(
        designer,
        experimenter.problem_statement(),
        iters=num_trials,
        batch_size=1,
        verbose=1,
        validate_parameters=True)
    self.assertLen(trials, num_trials)

  def test_make_suggestions_with_two_values(self):
    space = vz.SearchSpace()
    root = space.root
    root.add_int_param('integer_0', 11, 12)
    root.add_discrete_param('discrete_0', feasible_values=[1001, 2034])
    root.add_categorical_param('categorical_0', feasible_values=['a', 'b'])
    root.add_bool_param('bool_0')
    problem_statement = vz.ProblemStatement(space)
    problem_statement.metric_information.append(
        vz.MetricInformation(
            name='main_objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    designer = harmonica.HarmonicaDesigner(
        problem_statement, num_init_samples=1
    )
    num_trials = 10
    trials = test_runners.run_with_random_metrics(
        designer,
        problem_statement,
        iters=num_trials,
        batch_size=1,
        verbose=1,
        validate_parameters=True,
    )
    self.assertLen(trials, num_trials)

  def test_categorical_search_space_with_more_than_two_values_raises_error(
      self,
  ):
    experimenter = combo_experimenter.CentroidExperimenter(centroid_n_choice=3)
    with self.assertRaisesRegex(
        ValueError, 'Only boolean search spaces are supported'
    ):
      _ = harmonica.HarmonicaDesigner(
          experimenter.problem_statement(), num_init_samples=1
      )

  def test_integer_search_space_with_more_than_two_values_raises_error(
      self,
  ):
    space = vz.SearchSpace()
    root = space.root
    root.add_int_param('integer_0', 11, 13)
    problem_statement = vz.ProblemStatement(space)
    problem_statement.metric_information.append(
        vz.MetricInformation(
            name='main_objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    with self.assertRaisesRegex(
        ValueError, 'Only boolean search spaces are supported'
    ):
      _ = harmonica.HarmonicaDesigner(problem_statement, num_init_samples=1)

  def test_discrete_search_space_with_more_than_two_values_raises_error(
      self,
  ):
    space = vz.SearchSpace()
    root = space.root
    root.add_discrete_param('discrete_0', feasible_values=[1001, 2034, 3000])
    problem_statement = vz.ProblemStatement(space)
    problem_statement.metric_information.append(
        vz.MetricInformation(
            name='main_objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    with self.assertRaisesRegex(
        ValueError, 'Only boolean search spaces are supported'
    ):
      _ = harmonica.HarmonicaDesigner(problem_statement, num_init_samples=1)

  def test_continuous_search_space_raises_error(self):
    with self.assertRaisesRegex(
        ValueError, 'Only boolean search spaces are supported'
    ):
      _ = harmonica.HarmonicaDesigner(
          vz.ProblemStatement(
              test_studies.flat_continuous_space_with_scaling()
          ),
          num_init_samples=1,
      )


if __name__ == '__main__':
  absltest.main()
