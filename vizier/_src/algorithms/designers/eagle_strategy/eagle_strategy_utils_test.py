"""Tests for Eagle Strategy utils."""

import copy
from typing import Optional

from jax import numpy as jnp
from jax import random
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy_utils

from absl.testing import absltest
from absl.testing import parameterized


class UtilsTest(parameterized.TestCase):

  def setUp(self):
    super(UtilsTest, self).setUp()
    self.key = random.PRNGKey(0)
    search_space = vz.SearchSpace()
    root = search_space.root
    root.add_bool_param('b1')
    root.add_discrete_param('d1', [1.0, 2.0, 9.0, 10.0])
    root.add_float_param('f1', 0.0, 15.0, scale_type=vz.ScaleType.LINEAR)
    root.add_float_param('f2', 0.0, 10.0, scale_type=vz.ScaleType.LINEAR)
    root.add_int_param('i1', 0, 10, scale_type=vz.ScaleType.LINEAR)
    root.add_categorical_param('c1', ['a', 'b', 'c'])

    self.search_space = search_space
    # Create the main EagleStrategyUtils object to be tested
    self.utils = eagle_strategy_utils.EagleStrategyUtils(
        search_space=search_space,
        config=eagle_strategy_utils.FireflyAlgorithmConfig(),
        metric_name='obj',
        goal=vz.ObjectiveMetricGoal.MINIMIZE)

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

  def _get_parameter_config(self,
                            param_name: str) -> Optional[vz.ParameterConfig]:
    """Iterates over the search space parameters to find parameter by name."""
    for param_config in self.search_space.parameters:
      if param_config.name == param_name:
        return param_config

  def test_compute_cononical_distance(self):
    dist = self.utils.compute_cononical_distance(self.param_dict1,
                                                 self.param_dict2)
    self.assertIsInstance(dist, float)
    self.assertAlmostEqual(dist, 2.57)

  def test_compute_canonical_distance_squared_by_type(self):
    dists = self.utils.compute_canonical_distance_squared_by_type(
        self.param_dict1, self.param_dict2)
    self.assertEqual(dists[vz.ParameterType.CATEGORICAL], 1.0 + 0.0)
    self.assertEqual(dists[vz.ParameterType.INTEGER], ((3 - 1) / 10)**2)
    self.assertEqual(dists[vz.ParameterType.DOUBLE],
                     (3.0 / 15)**2 + (7.0 / 10)**2)
    self.assertEqual(dists[vz.ParameterType.DISCRETE], (9.0 / 9.0)**2)

  def test_better_than(self):
    trial1 = vz.Trial(parameters=self.param_dict1)
    trial1.complete(
        vz.Measurement(metrics={'obj': vz.Metric(value=2.0)}), inplace=True)
    trial2 = vz.Trial(parameters=self.param_dict2)
    trial2.complete(
        vz.Measurement(metrics={'obj': vz.Metric(value=1.5)}), inplace=True)
    # Test for minimization problem
    self.assertFalse(self.utils.better_than(trial1, trial2))
    self.assertTrue(self.utils.better_than(trial2, trial1))
    # Test for maximization problem
    utils_max = copy.deepcopy(self.utils)
    utils_max.goal = vz.ObjectiveMetricGoal.MAXIMIZE
    self.assertTrue(utils_max.better_than(trial1, trial2))
    self.assertFalse(utils_max.better_than(trial2, trial1))

  def test_better_than_infeasible(self):
    trial1 = vz.Trial(parameters=self.param_dict1)
    trial1.complete(
        vz.Measurement(metrics={'obj': vz.Metric(value=2.0)}), inplace=True)
    trial1._infeasibility_reason = 'infeasible reason'
    trial2 = vz.Trial(parameters=self.param_dict2)
    trial2.complete(
        vz.Measurement(metrics={'obj': vz.Metric(value=1.5)}), inplace=True)

    self.assertFalse(self.utils.better_than(trial1, trial2))
    self.assertFalse(self.utils.better_than(trial2, trial1))

  def test_is_pure_categorical(self):
    # Validate the the current search space is not pure categorical
    self.assertFalse(self.utils.is_pure_categorical())
    # Create pure categorical search space and validate
    pure_categorical_space = vz.SearchSpace()
    pure_categorical_space.root.add_bool_param('b1')
    pure_categorical_space.root.add_bool_param('b2')
    pure_categorical_space.root.add_categorical_param('c1', ['a', 'b'])
    pure_categorical_space.root.add_categorical_param('c2', ['a', 'b'])
    utils_pure_categorical = copy.deepcopy(self.utils)
    utils_pure_categorical.search_space = pure_categorical_space
    self.assertTrue(utils_pure_categorical.is_pure_categorical())

  def test_combine_two_parameters_integer(self):
    int_param_config = self._get_parameter_config('i1')
    _, new_value = self.utils.combine_two_parameters(self.key, int_param_config,
                                                     self.param_dict1,
                                                     self.param_dict2, 0.1)
    self.assertEqual(new_value, round(3 * 0.1 + 1 * 0.9))

  def test_combine_two_parameters_float(self):
    float_param_config = self._get_parameter_config('f1')
    _, new_value = self.utils.combine_two_parameters(self.key,
                                                     float_param_config,
                                                     self.param_dict1,
                                                     self.param_dict2, 0.1)
    self.assertEqual(new_value, 5.0 * 0.1 + 2.0 * 0.9)

  def test_combine_two_parameters_discrete(self):
    float_param_config = self._get_parameter_config('d1')
    _, new_value = self.utils.combine_two_parameters(self.key,
                                                     float_param_config,
                                                     self.param_dict1,
                                                     self.param_dict2, 0.1)
    self.assertEqual(new_value, 2.0)

  @parameterized.named_parameters(
      dict(testcase_name='prob=0', prob=0.0, target='b'),
      dict(testcase_name='prob=1.0', prob=1.0, target='a'),
      dict(testcase_name='prob=1.5', prob=1.5, target='a'),
      dict(testcase_name='prob=-0.1', prob=-0.1, target='b'),
  )
  def test_combine_two_parameters_categorical1(self, prob, target):
    categorical_param_config = self._get_parameter_config('c1')
    _, new_value = self.utils.combine_two_parameters(self.key,
                                                     categorical_param_config,
                                                     self.param_dict1,
                                                     self.param_dict2, prob)
    self.assertEqual(new_value, target)

  def test_combine_two_parameters_categorical2(self):
    categorical_param_config = self._get_parameter_config('c1')
    _, new_value = self.utils.combine_two_parameters(self.key,
                                                     categorical_param_config,
                                                     self.param_dict1,
                                                     self.param_dict2, 0.5)
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
    _, new_value = self.utils.perturb_parameter(self.key, decimal, value, prob)
    self.assertIsInstance(new_value, float)
    self.assertAlmostEqual(new_value, target)

  def test_perturb_parameter_categorical(self):
    categorical = vz.ParameterConfig.factory(
        'c1', feasible_values=['a', 'b', 'c'])
    _, new_value1 = self.utils.perturb_parameter(self.key, categorical, 'b',
                                                 0.2)
    self.assertIn(new_value1, ['a', 'b', 'c'])
    _, new_value2 = self.utils.perturb_parameter(self.key, categorical, 'b',
                                                 0.0)
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
    _, new_value = self.utils.perturb_parameter(self.key, integer, value, prob)
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
        'd1', feasible_values=[1.0, 2.0, 9.0, 10.0])
    _, new_value = self.utils.perturb_parameter(self.key, discrete, value, prob)
    self.assertIsInstance(new_value, float)
    self.assertEqual(new_value, target)

  def test_create_perturbations(self):
    key = random.PRNGKey(0)
    key, perturbations = self.utils.create_perturbations(key, 0.0)
    self.assertTrue(
        jnp.allclose(
            jnp.array(perturbations),
            jnp.zeros(len(self.utils.search_space.parameters))))
    key, perturbations = self.utils.create_perturbations(key, 0.5)
    self.assertLen(perturbations, len(self.utils.search_space.parameters))


if __name__ == '__main__':
  absltest.main()
