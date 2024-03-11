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

"""Test suite for SparseExperimenter."""

import attr
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter_factory
from vizier._src.benchmarks.experimenters import sparse_experimenter

from absl.testing import absltest
from absl.testing import parameterized


class SparseExperimenterTest(parameterized.TestCase):

  @parameterized.parameters(
      vz.ParameterConfig.factory('_', bounds=(0.0, 5.0)),
      vz.ParameterConfig.factory('_', bounds=(0, 5)),
      vz.ParameterConfig.factory('_', feasible_values=[1, 2, 3]),
      vz.ParameterConfig.factory('_', feasible_values=['a', 'b', 'c']),
  )
  def test_sparse_problem_statement_and_evaluate(self, sparse_param_config):
    sphere_experimenter = experimenter_factory.BBOBExperimenterFactory(
        name='Sphere', dim=2
    )()
    search_space = vz.SearchSpace()
    for idx in range(8):
      search_space.add(attr.evolve(sparse_param_config, name=str(idx)))

    experimenter = sparse_experimenter.SparseExperimenter(
        experiment=sphere_experimenter,
        search_space=search_space,
        prefix='__SPARSE',
    )
    # Test that sparse parameters were added.
    self.assertLen(experimenter.problem_statement().search_space.parameters, 10)
    trial = vz.Trial()
    for pc in experimenter.problem_statement().search_space.parameters:
      trial.parameters[pc.name] = 2.0
    # Test that the evaluation uses only the non-sparse parameters.
    experimenter.evaluate([trial])
    self.assertEqual(
        trial.final_measurement_or_die.metrics['bbob_eval'].value, 8.0
    )
    # Test that the evaluated trial parameters remained the same.
    self.assertLen(trial.parameters, 10)
    for trial_param_name, exptr_param_config in zip(
        trial.parameters,
        sphere_experimenter.problem_statement().search_space.parameters,
    ):
      self.assertEqual(trial_param_name, exptr_param_config.name)

  @parameterized.parameters(
      {
          'float_count': 1,
          'int_count': 2,
          'discrete_count': 4,
          'cat_count': 3,
      },
      {'float_count': 10, 'int_count': 0, 'discrete_count': 0, 'cat_count': 0},
      {'float_count': 0, 'int_count': 0, 'discrete_count': 0, 'cat_count': 0},
      {'float_count': 0, 'int_count': 0, 'discrete_count': 3, 'cat_count': 5},
  )
  def test_sparse_experimenter_create(
      self, float_count, int_count, discrete_count, cat_count
  ):
    sphere_experimenter = experimenter_factory.BBOBExperimenterFactory(
        name='Sphere', dim=2
    )()
    experimenter = sparse_experimenter.SparseExperimenter.create(
        experiment=sphere_experimenter,
        float_count=float_count,
        int_count=int_count,
        discrete_count=discrete_count,
        categorical_count=cat_count,
    )

    # Test that sparse parameters were added corretly to the search space.
    def _count_param_by_type(problem):
      """Count the number of parameters by parameter type."""
      counts = {type: 0 for type in vz.ParameterType}
      for param in problem.search_space.parameters:
        counts[param.type] += 1
      return counts

    counts = _count_param_by_type(experimenter.problem_statement())
    self.assertEqual(counts[vz.ParameterType.DOUBLE], float_count + 2)
    self.assertEqual(counts[vz.ParameterType.INTEGER], int_count)
    self.assertEqual(counts[vz.ParameterType.CATEGORICAL], cat_count)
    self.assertEqual(counts[vz.ParameterType.DISCRETE], discrete_count)

    trial = vz.Trial()
    for pc in experimenter.problem_statement().search_space.parameters:
      trial.parameters[pc.name] = 2.0
    # Test that the evaluation uses only the non-sparse parameters.
    experimenter.evaluate([trial])
    self.assertEqual(
        trial.final_measurement_or_die.metrics['bbob_eval'].value, 8.0
    )
    # Test that the evaluated trial parameters remained the same.
    total_param_count = 2 + int_count + cat_count + discrete_count + float_count
    self.assertLen(trial.parameters, total_param_count)
    for trial_param_name, exptr_param_config in zip(
        trial.parameters,
        sphere_experimenter.problem_statement().search_space.parameters,
    ):
      self.assertEqual(trial_param_name, exptr_param_config.name)


if __name__ == '__main__':
  absltest.main()
