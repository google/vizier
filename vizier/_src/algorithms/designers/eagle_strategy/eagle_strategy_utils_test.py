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

"""Tests for Eagle Strategy utils."""

import copy
from typing import Optional

import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy_utils
from vizier._src.algorithms.designers.eagle_strategy import testing
from absl.testing import absltest
from absl.testing import parameterized

EagleStrategyUtils = eagle_strategy_utils.EagleStrategyUtils
FireflyAlgorithmConfig = eagle_strategy_utils.FireflyAlgorithmConfig
EagleStrategyUtils = eagle_strategy_utils.EagleStrategyUtils
FireflyPool = eagle_strategy_utils.FireflyPool
Firefly = eagle_strategy_utils.Firefly


def _get_parameter_config(search_space: vz.SearchSpace,
                          param_name: str) -> Optional[vz.ParameterConfig]:
  """Iterates over the search space parameters to find parameter by name."""
  for param_config in search_space.parameters:
    if param_config.name == param_name:
      return param_config


class UtilsTest(parameterized.TestCase):
  """Tests for the EagleStrategyUtuls class."""

  def setUp(self):
    super(UtilsTest, self).setUp()
    self.rng = np.random.default_rng(seed=0)
    self.search_space = vz.SearchSpace()
    root = self.search_space.root
    root.add_bool_param('b1')
    root.add_discrete_param('d1', [1.0, 2.0, 9.0, 10.0])
    root.add_float_param('f1', 0.0, 15.0, scale_type=vz.ScaleType.LINEAR)
    root.add_float_param('f2', 0.0, 10.0, scale_type=vz.ScaleType.LINEAR)
    root.add_int_param('i1', 0, 10, scale_type=vz.ScaleType.LINEAR)
    root.add_categorical_param('c1', ['a', 'b', 'c'])

    self.param_dict1 = vz.ParameterDict()
    self.param_dict1['f1'] = 5.0
    self.param_dict1['f2'] = 1.0
    self.param_dict1['c1'] = 'a'
    self.param_dict1['d1'] = 10.0
    self.param_dict1['i1'] = 3
    self.param_dict1['b1'] = 'True'

    self.param_dict2 = vz.ParameterDict()
    self.param_dict2['f1'] = 2.0
    self.param_dict2['f2'] = 8.0
    self.param_dict2['c1'] = 'b'
    self.param_dict2['d1'] = 1.0
    self.param_dict2['i1'] = 1
    self.param_dict2['b1'] = 'True'

    metric_information_max = vz.MetricInformation(
        name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    metric_information_min = vz.MetricInformation(
        name='obj', goal=vz.ObjectiveMetricGoal.MINIMIZE)

    self.problem_max = vz.ProblemStatement(
        search_space=self.search_space,
        metric_information=[metric_information_max])
    self.problem_min = vz.ProblemStatement(
        search_space=self.search_space,
        metric_information=[metric_information_min])

    self.utils = EagleStrategyUtils(self.problem_max, FireflyAlgorithmConfig(),
                                    self.rng)

  def test_compute_pool_capacity(self):
    factor = FireflyAlgorithmConfig().pool_size_factor
    expected_capacity = 10 + round((6**factor + 6) * 0.5)
    self.assertEqual(self.utils.compute_pool_capacity(), expected_capacity)

  def test_compute_cononical_distance(self):
    dist = self.utils.compute_cononical_distance(
        self.param_dict1, self.param_dict2
    )
    self.assertIsInstance(dist, float)
    self.assertAlmostEqual(dist, 2.57)

  def test_compute_canonical_distance_squared_by_type(self):
    dists = self.utils._compute_canonical_distance_squared_by_type(
        self.param_dict1, self.param_dict2
    )
    self.assertEqual(dists[vz.ParameterType.CATEGORICAL], 1.0 + 0.0)
    self.assertEqual(dists[vz.ParameterType.INTEGER], ((3 - 1) / 10) ** 2)
    self.assertEqual(
        dists[vz.ParameterType.DOUBLE], (3.0 / 15) ** 2 + (7.0 / 10) ** 2
    )
    self.assertEqual(dists[vz.ParameterType.DISCRETE], (9.0 / 9.0) ** 2)

  def test_is_better_than(self):
    trial1 = vz.Trial(parameters=self.param_dict1)
    trial1.complete(
        vz.Measurement(
            metrics={eagle_strategy_utils.OBJECTIVE_NAME: vz.Metric(value=2.0)}
        ),
        inplace=True,
    )
    trial2 = vz.Trial(parameters=self.param_dict2)
    trial2.complete(
        vz.Measurement(
            metrics={eagle_strategy_utils.OBJECTIVE_NAME: vz.Metric(value=1.5)}
        ),
        inplace=True,
    )
    # Test for maximization problem
    self.assertTrue(self.utils.is_better_than(trial1, trial2))
    self.assertFalse(self.utils.is_better_than(trial2, trial1))
    # Test for minimization problem
    min_problem_utils = copy.deepcopy(self.utils)
    min_problem_utils.problem_statement = self.problem_min
    min_problem_utils.__attrs_post_init__()
    self.assertFalse(min_problem_utils.is_better_than(trial1, trial2))
    self.assertTrue(min_problem_utils.is_better_than(trial2, trial1))

  def test_is_better_than_infeasible(self):
    trial1 = vz.Trial(parameters=self.param_dict1)
    trial1.complete(
        vz.Measurement(metrics={'obj': vz.Metric(value=2.0)}),
        inplace=True,
        infeasibility_reason='infeasible reason',
    )
    trial2 = vz.Trial(parameters=self.param_dict2)
    trial2.complete(
        vz.Measurement(metrics={'obj': vz.Metric(value=1.5)}), inplace=True
    )
    self.assertFalse(self.utils.is_better_than(trial1, trial2))
    self.assertTrue(self.utils.is_better_than(trial2, trial1))

  def test_is_pure_categorical(self):
    pure_categorical_space = vz.SearchSpace()
    pure_categorical_space.root.add_bool_param('b1')
    pure_categorical_space.root.add_bool_param('b2')
    pure_categorical_space.root.add_categorical_param('c1', ['a', 'b'])
    pure_categorical_space.root.add_categorical_param('c2', ['a', 'b'])
    pure_categorical_space_utils = copy.deepcopy(self.utils)
    pure_categorical_space_utils._search_space = pure_categorical_space
    self.assertTrue(pure_categorical_space_utils.is_pure_categorical())
    self.assertFalse(self.utils.is_pure_categorical())

  def test_combine_two_parameters_integer(self):
    int_param_config = _get_parameter_config(self.search_space, 'i1')
    new_value = self.utils.combine_two_parameters(
        int_param_config, self.param_dict1, self.param_dict2, 0.1
    )
    self.assertEqual(new_value, round(3 * 0.1 + 1 * 0.9))

  def test_combine_two_parameters_float(self):
    float_param_config = _get_parameter_config(self.search_space, 'f1')
    new_value = self.utils.combine_two_parameters(
        float_param_config, self.param_dict1, self.param_dict2, 0.1
    )
    self.assertEqual(new_value, 5.0 * 0.1 + 2.0 * 0.9)

  def test_combine_two_parameters_discrete(self):
    float_param_config = _get_parameter_config(self.search_space, 'd1')
    new_value = self.utils.combine_two_parameters(
        float_param_config, self.param_dict1, self.param_dict2, 0.1
    )
    self.assertEqual(new_value, 2.0)

  @parameterized.named_parameters(
      dict(testcase_name='prob=0', prob=0.0, target='b'),
      dict(testcase_name='prob=1.0', prob=1.0, target='a'),
      dict(testcase_name='prob=1.5', prob=1.5, target='a'),
      dict(testcase_name='prob=-0.1', prob=-0.1, target='b'),
  )
  def test_combine_two_parameters_categorical1(self, prob, target):
    categorical_param_config = _get_parameter_config(self.search_space, 'c1')
    new_value = self.utils.combine_two_parameters(
        categorical_param_config, self.param_dict1, self.param_dict2, prob
    )
    self.assertEqual(new_value, target)

  def test_combine_two_parameters_categorical2(self):
    categorical_param_config = _get_parameter_config(self.search_space, 'c1')
    new_value = self.utils.combine_two_parameters(
        categorical_param_config, self.param_dict1, self.param_dict2, 0.5
    )
    self.assertIn(new_value, ['a', 'b'])

  @parameterized.named_parameters(
      dict(testcase_name='Above1', value=9.0, prob=0.999, target=10.0),
      dict(testcase_name='Above2', value=10.0, prob=0.999, target=10.0),
      dict(testcase_name='Below1', value=2.0, prob=-0.999, target=0.0),
      dict(testcase_name='Below2', value=0.0, prob=-0.999, target=0.0),
      dict(testcase_name='Change1', value=5.0, prob=0.212, target=7.12),
      dict(testcase_name='Change2', value=5.0, prob=-0.1, target=4.0),
      dict(testcase_name='Change3', value=2.0, prob=0.111, target=3.11),
  )
  def test_perturb_parameter_float(self, value, prob, target):
    decimal = vz.ParameterConfig.factory('f1', bounds=(0.0, 10.0))
    new_value = self.utils.perturb_parameter(decimal, value, prob)
    self.assertIsInstance(new_value, float)
    self.assertAlmostEqual(new_value, target)

  def test_perturb_parameter_categorical(self):
    categorical = vz.ParameterConfig.factory(
        'c1', feasible_values=['a', 'b', 'c']
    )
    new_value1 = self.utils.perturb_parameter(categorical, 'b', 0.2)
    self.assertIn(new_value1, ['a', 'b', 'c'])
    new_value2 = self.utils.perturb_parameter(categorical, 'b', 0.0)
    self.assertEqual(new_value2, 'b')

  @parameterized.named_parameters(
      dict(testcase_name='Above1', value=9, prob=0.999, target=10),
      dict(testcase_name='Above2', value=10, prob=0.999, target=10),
      dict(testcase_name='Below1', value=2, prob=-0.999, target=0),
      dict(testcase_name='Below2', value=0, prob=-0.999, target=0),
      dict(testcase_name='NoChange', value=2, prob=0.000001, target=2),
      dict(testcase_name='Change1', value=1, prob=0.09, target=2),
      dict(testcase_name='Change2', value=1, prob=0.51, target=6),
  )
  def test_perturb_parameter_integer(self, value, prob, target):
    integer = vz.ParameterConfig.factory('i1', bounds=(0, 10))
    new_value = self.utils.perturb_parameter(integer, value, prob)
    self.assertIsInstance(new_value, int)
    self.assertEqual(new_value, target)

  @parameterized.named_parameters(
      dict(testcase_name='Above1', value=9.0, prob=0.999, target=10.0),
      dict(testcase_name='Above2', value=10.0, prob=0.999, target=10.0),
      dict(testcase_name='Below1', value=2.0, prob=-0.999, target=1.0),
      dict(testcase_name='Below2', value=1.0, prob=-0.999, target=1.0),
      dict(testcase_name='NoChange', value=2.0, prob=0.000001, target=2.0),
      dict(testcase_name='Change', value=1.0, prob=0.09, target=2.0),
  )
  def test_perturb_parameter_discrete(self, value, prob, target):
    discrete = vz.ParameterConfig.factory(
        'd1', feasible_values=[1.0, 2.0, 9.0, 10.0]
    )
    new_value = self.utils.perturb_parameter(discrete, value, prob)
    self.assertIsInstance(new_value, float)
    self.assertEqual(new_value, target)

  def test_degrees_of_freedom(self):
    dof = self.utils._degrees_of_freedom
    self.assertEqual(dof[vz.ParameterType.DOUBLE], 2)
    self.assertEqual(dof[vz.ParameterType.CATEGORICAL], 2)
    self.assertEqual(dof[vz.ParameterType.INTEGER], 1)
    self.assertEqual(dof[vz.ParameterType.DISCRETE], 1)

  def test_replace_trial_metric_name(self):
    search_space = vz.SearchSpace()
    root = search_space.root
    root.add_float_param('f1', 0.0, 15.0, scale_type=vz.ScaleType.LINEAR)
    metric_information = vz.MetricInformation(
        name='obj123', goal=vz.ObjectiveMetricGoal.MAXIMIZE
    )
    problem = vz.ProblemStatement(
        search_space=search_space, metric_information=[metric_information]
    )

    utils = EagleStrategyUtils(problem, FireflyAlgorithmConfig(), self.rng)
    metadata = vz.Metadata()
    metadata.ns('eagle')['parent_fly_id'] = '123'
    trial = vz.Trial(parameters={'f1': 0.0}, metadata=metadata)
    trial.complete(measurement=vz.Measurement(metrics={'obj123': 1123.3}))
    new_trial = utils.standardize_trial_metric_name(trial)
    self.assertEqual(
        new_trial.final_measurement_or_die.metrics['objective'].value, 1123.3
    )
    self.assertEqual(new_trial.parameters['f1'].value, 0.0)
    self.assertEqual(new_trial.metadata.ns('eagle')['parent_fly_id'], '123')


class FireflyPoolTest(parameterized.TestCase):
  """Tests for the FireflyPool class."""

  def test_generate_new_fly_id(self):
    firefly_pool = testing.create_fake_empty_firefly_pool(capacity=2)
    self.assertEqual(firefly_pool.generate_new_fly_id(), 0)
    self.assertEqual(firefly_pool.generate_new_fly_id(), 1)
    self.assertEqual(firefly_pool.generate_new_fly_id(), 2)
    self.assertEqual(firefly_pool.generate_new_fly_id(), 3)

  def test_create_or_update_fly(self):
    # Test creating a new fly in the pool.
    firefly_pool = testing.create_fake_empty_firefly_pool()
    trial = testing.create_fake_trial(
        parent_fly_id=112, x_value=0, obj_value=0.8
    )
    firefly_pool.create_or_update_fly(trial, 112)
    self.assertEqual(firefly_pool.size, 1)
    self.assertLen(firefly_pool._pool, 1)
    self.assertIs(firefly_pool._pool[112].trial, trial)
    # Test that another trial with the same parent id updates the fly.
    trial2 = testing.create_fake_trial(
        parent_fly_id=112, x_value=1, obj_value=1.5
    )
    firefly_pool.create_or_update_fly(trial2, 112)
    self.assertEqual(firefly_pool.size, 1)
    self.assertLen(firefly_pool._pool, 1)
    self.assertIs(firefly_pool._pool[112].trial, trial2)

  @parameterized.parameters(
      {'x_values': [1, 2, 5], 'obj_values': [2, 10, -2]},
      {'x_values': [1, 2, 5], 'obj_values': [None, 10, -2]},
  )
  def test_find_closest_parent(self, x_values, obj_values):
    """Tests that the find_closest_parent method returns the closest fly (with infeasible trials)."""
    firefly_pool = testing.create_fake_populated_firefly_pool(
        x_values=x_values, obj_values=obj_values, capacity=4
    )
    trial = testing.create_fake_trial(
        parent_fly_id=123, x_value=4.2, obj_value=8
    )
    parent_fly = firefly_pool.find_closest_parent(trial)
    self.assertEqual(parent_fly.id_, 2)

  @parameterized.parameters(
      {'x_values': [1, 2, 5], 'obj_values': [2, 10, -2]},
      {'x_values': [1, 2, 5], 'obj_values': [None, 10, -2]},
  )
  def test_is_best_fly(self, x_values, obj_values):
    """Tests that the is_best_fly method returns true if the fly is the best fly (with infeasible trials)."""
    firefly_pool = testing.create_fake_populated_firefly_pool(
        x_values=x_values, obj_values=obj_values, capacity=4
    )
    self.assertTrue(firefly_pool.is_best_fly(firefly_pool._pool[1]))
    self.assertFalse(firefly_pool.is_best_fly(firefly_pool._pool[0]))
    self.assertFalse(firefly_pool.is_best_fly(firefly_pool._pool[2]))

  def test_get_next_moving_fly_copy(self):
    firefly_pool = testing.create_fake_populated_firefly_pool(
        x_values=[1, 2, 5], obj_values=[2, 10, -2], capacity=5
    )
    firefly_pool._last_id = 1
    moving_fly1 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly1.id_, 2)
    moving_fly2 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly2.id_, 0)
    moving_fly3 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly3.id_, 1)

  @parameterized.parameters(
      {
          'x_values': [1, 2, 5, 3],
          'obj_values': [2, 10, None, None],
          'next_fly_indices': [0, 1, 0],
      },
      {
          'x_values': [1, 2, 5, 3, 0, 4],
          'obj_values': [2, 10, None, None, 3, None],
          'next_fly_indices': [4, 0, 1, 4],
      },
  )
  def test_get_next_moving_fly_copy_with_infeasible(
      self, x_values, obj_values, next_fly_indices
  ):
    """Tests that the get_next_moving_fly_copy method doesn't return infeasible fly."""
    firefly_pool = testing.create_fake_populated_firefly_pool(
        x_values=x_values, obj_values=obj_values, capacity=5
    )
    firefly_pool._last_id = 1
    for index in next_fly_indices:
      moving_fly = firefly_pool.get_next_moving_fly_copy()
      self.assertEqual(moving_fly.id_, index)

  def test_get_next_moving_fly_copy_after_removing_last_id_fly(self):
    firefly_pool = testing.create_fake_populated_firefly_pool(
        x_values=[1, 2, 5], obj_values=[2, 10, -2], capacity=5
    )
    firefly_pool._last_id = 1
    # Remove the fly associated with `_last_id` from the pool.
    del firefly_pool._pool[1]
    moving_fly1 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly1.id_, 2)
    moving_fly2 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly2.id_, 0)
    moving_fly3 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly3.id_, 2)

  def test_get_next_moving_fly_copy_after_removing_multiple_flies(self):
    firefly_pool = testing.create_fake_populated_firefly_pool(
        x_values=[1, 2, 5, -1], obj_values=[2, 10, -2, 8], capacity=5
    )
    firefly_pool._last_id = 3
    # Remove the several flies
    del firefly_pool._pool[0]
    del firefly_pool._pool[2]
    moving_fly1 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly1.id_, 1)
    moving_fly2 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly2.id_, 3)
    moving_fly3 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly3.id_, 1)


def test_pool_size_with_infeasible(self):
  """Tests that the pool size doesn't change when adding an infeasible fly."""
  firefly_pool = testing.create_fake_populated_firefly_pool(
      x_values=[1, 2, 5, -1], obj_values=[2, 10, -2, 8], capacity=5
  )
  infeasible_firefly_id = firefly_pool.generate_new_fly_id()
  infeasible_trial = testing.create_fake_trial(
      parent_fly_id=infeasible_firefly_id, x_value=-1, obj_value=None
  )
  self.assertEqual(firefly_pool.size, 4)
  firefly_pool.create_or_update_fly(
      infeasible_trial, parent_fly_id=infeasible_firefly_id
  )
  self.assertEqual(firefly_pool.capacity, 5)
  # Test that adding the infeasible trial doesn't change the pool size.
  self.assertEqual(firefly_pool.size, 4)
  self.assertEqual(firefly_pool._infeasible_count, 1)


if __name__ == '__main__':
  absltest.main()
