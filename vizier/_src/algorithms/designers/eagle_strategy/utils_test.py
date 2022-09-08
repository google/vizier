"""Tests for Eagle Strategy utils."""

from typing import Optional
from jax import random
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.eagle_strategy import utils

from absl.testing import absltest
from absl.testing import parameterized


def _get_parameter_config(search_space: vz.SearchSpace,
                          param_name: str) -> Optional[vz.ParameterConfig]:
  """Iterates over the search space parameters to find parameter by name."""
  for param_config in search_space.parameters:
    if param_config.name == param_name:
      return param_config


class UtilsTest(parameterized.TestCase):

  def setUp(self):
    super(UtilsTest, self).setUp()
    self.key = random.PRNGKey(0)
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

  def test_compute_cononical_distance(self):
    dist = utils.compute_cononical_distance(self.param_dict1, self.param_dict2,
                                            self.search_space)
    self.assertIsInstance(dist, float)
    self.assertAlmostEqual(dist, 2.57)

  def test_compute_canonical_distance_squared_by_type(self):
    dists = utils.compute_canonical_distance_squared_by_type(
        self.param_dict1, self.param_dict2, self.search_space)
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
    # Test for maximization problem
    self.assertTrue(utils.better_than(self.problem_max, trial1, trial2))
    self.assertFalse(utils.better_than(self.problem_max, trial2, trial1))
    # Test for minimization problem
    self.assertFalse(utils.better_than(self.problem_min, trial1, trial2))
    self.assertTrue(utils.better_than(self.problem_min, trial2, trial1))

  def test_better_than_infeasible(self):
    trial1 = vz.Trial(parameters=self.param_dict1)
    trial1.complete(
        vz.Measurement(metrics={'obj': vz.Metric(value=2.0)}), inplace=True)
    trial1._infeasibility_reason = 'infeasible reason'
    trial2 = vz.Trial(parameters=self.param_dict2)
    trial2.complete(
        vz.Measurement(metrics={'obj': vz.Metric(value=1.5)}), inplace=True)

    self.assertFalse(utils.better_than(self.problem_max, trial1, trial2))
    self.assertFalse(utils.better_than(self.problem_min, trial2, trial1))

  def test_is_pure_categorical(self):
    pure_categorical_space = vz.SearchSpace()
    pure_categorical_space.root.add_bool_param('b1')
    pure_categorical_space.root.add_bool_param('b2')
    pure_categorical_space.root.add_categorical_param('c1', ['a', 'b'])
    pure_categorical_space.root.add_categorical_param('c2', ['a', 'b'])
    self.assertTrue(utils.is_pure_categorical(pure_categorical_space))
    self.assertFalse(utils.is_pure_categorical(self.search_space))

  def test_combine_two_parameters_integer(self):
    int_param_config = _get_parameter_config(self.search_space, 'i1')
    _, new_value = utils.combine_two_parameters(self.key, int_param_config,
                                                self.param_dict1,
                                                self.param_dict2, 0.1)
    self.assertEqual(new_value, round(3 * 0.1 + 1 * 0.9))

  def test_combine_two_parameters_float(self):
    float_param_config = _get_parameter_config(self.search_space, 'f1')
    _, new_value = utils.combine_two_parameters(self.key, float_param_config,
                                                self.param_dict1,
                                                self.param_dict2, 0.1)
    self.assertEqual(new_value, 5.0 * 0.1 + 2.0 * 0.9)

  def test_combine_two_parameters_discrete(self):
    float_param_config = _get_parameter_config(self.search_space, 'd1')
    _, new_value = utils.combine_two_parameters(self.key, float_param_config,
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
    categorical_param_config = _get_parameter_config(self.search_space, 'c1')
    _, new_value = utils.combine_two_parameters(self.key,
                                                categorical_param_config,
                                                self.param_dict1,
                                                self.param_dict2, prob)
    self.assertEqual(new_value, target)

  def test_combine_two_parameters_categorical2(self):
    categorical_param_config = _get_parameter_config(self.search_space, 'c1')
    _, new_value = utils.combine_two_parameters(self.key,
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
    _, new_value = utils.perturb_parameter(self.key, decimal, value, prob)
    self.assertIsInstance(new_value, float)
    self.assertAlmostEqual(new_value, target)

  def test_perturb_parameter_categorical(self):
    categorical = vz.ParameterConfig.factory(
        'c1', feasible_values=['a', 'b', 'c'])
    _, new_value1 = utils.perturb_parameter(self.key, categorical, 'b', 0.2)
    self.assertIn(new_value1, ['a', 'b', 'c'])
    _, new_value2 = utils.perturb_parameter(self.key, categorical, 'b', 0.0)
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
    _, new_value = utils.perturb_parameter(self.key, integer, value, prob)
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
    _, new_value = utils.perturb_parameter(self.key, discrete, value, prob)
    self.assertIsInstance(new_value, float)
    self.assertEqual(new_value, target)


if __name__ == '__main__':
  absltest.main()
