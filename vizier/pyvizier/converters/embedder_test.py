# Copyright 2023 Google LLC.
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

"""Tests for problem statement converter."""

from vizier import pyvizier as vz
from vizier.pyvizier.converters import embedder
from absl.testing import absltest


class ProblemStatementTest(absltest.TestCase):

  def test_convert_search_space(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_int_param('i1', 0, 10)
    problem.search_space.root.add_discrete_param('d1', [0, 10, 20])
    problem.search_space.root.add_categorical_param('c1', ['a', 'b', 'c'])
    converter = embedder.ProblemAndTrialsScaler(problem)

    new_space = converter.problem_statement.search_space
    self.assertEqual(new_space.get('f1').type, vz.ParameterType.DOUBLE)
    self.assertEqual(new_space.get('f1').bounds, (0.0, 1.0))

    self.assertEqual(new_space.get('i1').type, vz.ParameterType.DOUBLE)
    self.assertEqual(new_space.get('i1').bounds, (0.0, 1.0))

    self.assertEqual(new_space.get('d1').type, vz.ParameterType.DISCRETE)
    self.assertEqual(new_space.get('d1').feasible_values, [0.0, 0.5, 1.0])

    self.assertEqual(new_space.get('c1').type, vz.ParameterType.CATEGORICAL)
    self.assertEqual(new_space.get('c1').feasible_values, ['a', 'b', 'c'])

  def test_map_trials(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_int_param('i1', 0, 10)
    problem.search_space.root.add_discrete_param('d1', [0, 10, 20])
    problem.search_space.root.add_categorical_param('c1', ['a', 'b', 'c'])
    converter = embedder.ProblemAndTrialsScaler(problem)

    trial1 = vz.Trial(parameters={'f1': 1.0, 'i1': 0, 'd1': 10, 'c1': 'b'})
    trial2 = vz.Trial(parameters={'f1': 4.0, 'i1': 8, 'd1': 0, 'c1': 'a'})
    suggestion_trials = [trial1, trial2]

    mapped_trials = converter.map(suggestion_trials)
    self.assertAlmostEqual(mapped_trials[0].parameters['f1'].value, 0.1)
    self.assertAlmostEqual(mapped_trials[0].parameters['i1'].value, 0.0)
    self.assertAlmostEqual(mapped_trials[0].parameters['d1'].value, 0.5)
    self.assertEqual(mapped_trials[0].parameters['c1'].value, 'b')

    self.assertAlmostEqual(mapped_trials[1].parameters['f1'].value, 0.4)
    self.assertAlmostEqual(mapped_trials[1].parameters['i1'].value, 0.8)
    self.assertAlmostEqual(mapped_trials[1].parameters['d1'].value, 0.0)
    self.assertEqual(mapped_trials[1].parameters['c1'].value, 'a')

  def test_unmap_trials(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_int_param('i1', 0, 10)
    problem.search_space.root.add_discrete_param('d1', [0, 10, 20])
    problem.search_space.root.add_categorical_param('c1', ['a', 'b', 'c'])
    converter = embedder.ProblemAndTrialsScaler(problem)

    trial1 = vz.Trial(parameters={'f1': 1.0, 'i1': 0, 'd1': 10, 'c1': 'b'})
    trial2 = vz.Trial(parameters={'f1': 4.0, 'i1': 8, 'd1': 0, 'c1': 'c'})
    trial3 = vz.Trial(parameters={'f1': 0.0, 'i1': 8, 'd1': 20, 'c1': 'b'})
    trial4 = vz.Trial(parameters={'f1': 10.0, 'i1': 2, 'd1': 10, 'c1': 'a'})
    suggestion_trials = [trial1, trial2, trial3, trial4]
    recovered_trials = converter.unmap(converter.map(suggestion_trials))

    for i, trial in enumerate(suggestion_trials):
      for param_config in problem.search_space.parameters:
        name = param_config.name
        if param_config.type == vz.ParameterType.CATEGORICAL:
          self.assertEqual(
              trial.parameters[name].value,
              recovered_trials[i].parameters[name].value,
          )
        else:
          self.assertAlmostEqual(
              trial.parameters[name].value,
              recovered_trials[i].parameters[name].value,
              places=5,
          )


if __name__ == '__main__':
  absltest.main()
