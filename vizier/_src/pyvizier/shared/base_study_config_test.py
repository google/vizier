"""Tests for vizier.pyvizier.shared.base_study_config."""

import numpy as np
from vizier._src.pyvizier.shared import base_study_config
from vizier._src.pyvizier.shared import parameter_config as pc
from vizier._src.pyvizier.shared import trial
from absl.testing import absltest
from absl.testing import parameterized


class ObjectiveMetricGoalTest(absltest.TestCase):

  def test_basics(self):
    self.assertTrue(base_study_config.ObjectiveMetricGoal.MAXIMIZE.is_maximize)
    self.assertFalse(base_study_config.ObjectiveMetricGoal.MAXIMIZE.is_minimize)
    self.assertTrue(base_study_config.ObjectiveMetricGoal.MINIMIZE.is_minimize)
    self.assertFalse(base_study_config.ObjectiveMetricGoal.MINIMIZE.is_maximize)


class MetricTypeTest(absltest.TestCase):

  def test_basics(self):
    self.assertTrue(base_study_config.MetricType.SAFETY.is_safety)
    self.assertTrue(base_study_config.MetricType.OBJECTIVE.is_objective)


class MetricInformationTest(absltest.TestCase):

  def testMinMaxValueDefault(self):
    info = base_study_config.MetricInformation(
        goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE)
    self.assertEqual(info.min_value, -np.inf)
    self.assertEqual(info.max_value, np.inf)

  def testMinMaxValueSet(self):
    info = base_study_config.MetricInformation(
        goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE,
        min_value=-1.,
        max_value=1.)
    self.assertEqual(info.min_value, -1.)
    self.assertEqual(info.max_value, 1.)

  def testMinMaxBadValueInit(self):
    with self.assertRaises(ValueError):
      base_study_config.MetricInformation(
          goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE,
          min_value=1.,
          max_value=-1.)

  def testMinMaxBadValueSet(self):
    info = base_study_config.MetricInformation(
        goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE,
        min_value=-1.,
        max_value=1.)
    with self.assertRaises(ValueError):
      info.min_value = 2.
    with self.assertRaises(ValueError):
      info.max_value = -2.


class MetricsConfigTest(parameterized.TestCase):

  def testBasics(self):
    config = base_study_config.MetricsConfig()
    config.append(
        base_study_config.MetricInformation(
            name='max1', goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE))
    config.extend([
        base_study_config.MetricInformation(
            name='max_safe1',
            goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE,
            safety_threshold=0.0),
        base_study_config.MetricInformation(
            name='max2', goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE),
        base_study_config.MetricInformation(
            name='min1', goal=base_study_config.ObjectiveMetricGoal.MINIMIZE),
        base_study_config.MetricInformation(
            name='min_safe2',
            goal=base_study_config.ObjectiveMetricGoal.MINIMIZE,
            safety_threshold=0.0)
    ])
    self.assertLen(config, 5)
    self.assertLen(config.of_type(base_study_config.MetricType.OBJECTIVE), 3)
    self.assertLen(config.of_type(base_study_config.MetricType.SAFETY), 2)

  def testDuplicateNames(self):
    config = base_study_config.MetricsConfig()
    config.append(
        base_study_config.MetricInformation(
            name='max1', goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE))
    with self.assertRaises(ValueError):
      config.append(
          base_study_config.MetricInformation(
              name='max1', goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE))


class SearchSpaceTest(parameterized.TestCase):

  def testAddFloatParamMinimal(self):
    space = base_study_config.SearchSpace()
    self.assertEmpty(space.parameters)
    selector = space.select_root().add_float_param('f1', 1.0, 15.0)
    # Test the returned selector.
    self.assertEqual(selector.path_string, '')
    self.assertEqual(selector.parameter_name, 'f1')
    self.assertEqual(selector.parameter_values, [])
    # Test the search space.
    self.assertLen(space.parameters, 1)
    self.assertEqual(space.parameters[0].name, 'f1')
    self.assertEqual(space.parameters[0].type, pc.ParameterType.DOUBLE)
    self.assertEqual(space.parameters[0].bounds, (1.0, 15.0))
    self.assertEqual(space.parameters[0].scale_type, pc.ScaleType.LINEAR)
    self.assertEmpty(space.parameters[0].matching_parent_values)
    self.assertEmpty(space.parameters[0].child_parameter_configs)
    with self.assertRaisesRegex(ValueError, 'feasible_values is invalid.*'):
      _ = space.parameters[0].feasible_values
    self.assertIsNone(space.parameters[0].default_value)

    _ = space.select_root().add_float_param('f2', 2.0, 16.0)
    self.assertLen(space.parameters, 2)
    self.assertEqual(space.parameters[0].name, 'f1')
    self.assertEqual(space.parameters[0].type, pc.ParameterType.DOUBLE)
    self.assertEqual(space.parameters[0].bounds, (1.0, 15.0))
    self.assertEqual(space.parameters[1].name, 'f2')
    self.assertEqual(space.parameters[1].type, pc.ParameterType.DOUBLE)
    self.assertEqual(space.parameters[1].bounds, (2.0, 16.0))

  def testAddFloatParam(self):
    space = base_study_config.SearchSpace()
    self.assertEmpty(space.parameters)
    _ = space.select_root().add_float_param(
        'f1', 1.0, 15.0, default_value=3.0, scale_type=pc.ScaleType.LOG)
    self.assertLen(space.parameters, 1)
    self.assertEqual(space.parameters[0].name, 'f1')
    self.assertEqual(space.parameters[0].type, pc.ParameterType.DOUBLE)
    self.assertEqual(space.parameters[0].bounds, (1.0, 15.0))
    self.assertEqual(space.parameters[0].scale_type, pc.ScaleType.LOG)
    self.assertEmpty(space.parameters[0].matching_parent_values)
    self.assertEmpty(space.parameters[0].child_parameter_configs)
    with self.assertRaisesRegex(ValueError, 'feasible_values is invalid.*'):
      _ = space.parameters[0].feasible_values
    self.assertEqual(space.parameters[0].default_value, 3.0)

  def testAddDiscreteParamIntegerFeasibleValues(self):
    """Test a Discrete parameter with integer feasible values."""
    space = base_study_config.SearchSpace()
    self.assertEmpty(space.parameters)
    _ = space.select_root().add_discrete_param(
        'd1', [101, 15.0, 21.0], default_value=15.0)
    self.assertLen(space.parameters, 1)
    self.assertEqual(space.parameters[0].name, 'd1')
    self.assertEqual(space.parameters[0].type, pc.ParameterType.DISCRETE)
    self.assertEqual(space.parameters[0].bounds, (15.0, 101.0))
    self.assertEqual(space.parameters[0].scale_type, pc.ScaleType.LINEAR)
    self.assertEmpty(space.parameters[0].matching_parent_values)
    self.assertEmpty(space.parameters[0].child_parameter_configs)
    self.assertEqual(space.parameters[0].feasible_values, [15.0, 21.0, 101])
    self.assertEqual(space.parameters[0].default_value, 15.0)
    self.assertEqual(space.parameters[0].external_type, pc.ExternalType.INTEGER)

  def testAddDiscreteParamFloatFeasibleValues(self):
    """Test a Discrete parameter with float feasible values."""
    space = base_study_config.SearchSpace()
    self.assertEmpty(space.parameters)
    _ = space.select_root().add_discrete_param(
        'd1', [15.1, 21.0, 101], default_value=15.1)
    self.assertLen(space.parameters, 1)
    self.assertEqual(space.parameters[0].external_type, pc.ExternalType.FLOAT)

  def testAddBooleanParam(self):
    """Test a Boolean parameter."""
    space = base_study_config.SearchSpace()
    self.assertEmpty(space.parameters)
    _ = space.select_root().add_bool_param('b1', default_value=True)
    self.assertLen(space.parameters, 1)
    self.assertEqual(space.parameters[0].name, 'b1')
    self.assertEqual(space.parameters[0].type, pc.ParameterType.CATEGORICAL)
    with self.assertRaisesRegex(ValueError,
                                'Accessing bounds of a categorical.*'):
      _ = space.parameters[0].bounds
    self.assertIsNone(space.parameters[0].scale_type)
    self.assertEmpty(space.parameters[0].matching_parent_values)
    self.assertEmpty(space.parameters[0].child_parameter_configs)
    self.assertEqual(space.parameters[0].feasible_values, ['False', 'True'])
    self.assertEqual(space.parameters[0].default_value, 'True')
    self.assertEqual(space.parameters[0].external_type, pc.ExternalType.BOOLEAN)

  def testAddBooleanParamWithFalseDefault(self):
    """Test a Boolean parameter."""
    space = base_study_config.SearchSpace()
    self.assertEmpty(space.parameters)
    _ = space.select_root().add_bool_param('b1', default_value=False)
    self.assertLen(space.parameters, 1)
    self.assertEqual(space.parameters[0].default_value, 'False')

  def testAddTwoFloatParams(self):
    space = base_study_config.SearchSpace()
    self.assertEmpty(space.parameters)
    _ = space.select_root().add_float_param(
        'f1', 1.0, 15.0, default_value=3.0, scale_type=pc.ScaleType.LOG)
    _ = space.select_root().add_float_param(
        'f2', 2.0, 16.0, default_value=4.0, scale_type=pc.ScaleType.REVERSE_LOG)

    self.assertLen(space.parameters, 2)

    self.assertEqual(space.parameters[0].name, 'f1')
    self.assertEqual(space.parameters[0].type, pc.ParameterType.DOUBLE)
    self.assertEqual(space.parameters[0].bounds, (1.0, 15.0))
    self.assertEqual(space.parameters[0].scale_type, pc.ScaleType.LOG)
    self.assertEmpty(space.parameters[0].matching_parent_values)
    self.assertEmpty(space.parameters[0].child_parameter_configs)
    with self.assertRaisesRegex(ValueError, 'feasible_values is invalid.*'):
      _ = space.parameters[0].feasible_values
    self.assertEqual(space.parameters[0].default_value, 3.0)

    self.assertEqual(space.parameters[1].name, 'f2')
    self.assertEqual(space.parameters[1].type, pc.ParameterType.DOUBLE)
    self.assertEqual(space.parameters[1].bounds, (2.0, 16.0))
    self.assertEqual(space.parameters[1].scale_type, pc.ScaleType.REVERSE_LOG)
    self.assertEmpty(space.parameters[1].matching_parent_values)
    self.assertEmpty(space.parameters[1].child_parameter_configs)
    with self.assertRaisesRegex(ValueError, 'feasible_values is invalid.*'):
      _ = space.parameters[1].feasible_values
    self.assertEqual(space.parameters[1].default_value, 4.0)

  def testChainAddTwoFloatParams(self):
    space = base_study_config.SearchSpace()
    self.assertEmpty(space.parameters)
    root = space.select_root()
    root.add_float_param(
        'f1', 1.0, 15.0, default_value=3.0, scale_type=pc.ScaleType.LOG)
    root.add_float_param(
        'f2', 2.0, 16.0, default_value=4.0, scale_type=pc.ScaleType.REVERSE_LOG)

    self.assertLen(space.parameters, 2)

    self.assertEqual(space.parameters[0].name, 'f1')
    self.assertEqual(space.parameters[0].type, pc.ParameterType.DOUBLE)
    self.assertEqual(space.parameters[0].bounds, (1.0, 15.0))
    self.assertEqual(space.parameters[0].scale_type, pc.ScaleType.LOG)
    self.assertEmpty(space.parameters[0].matching_parent_values)
    self.assertEmpty(space.parameters[0].child_parameter_configs)
    with self.assertRaisesRegex(ValueError, 'feasible_values is invalid.*'):
      _ = space.parameters[0].feasible_values
    self.assertEqual(space.parameters[0].default_value, 3.0)

    self.assertEqual(space.parameters[1].name, 'f2')
    self.assertEqual(space.parameters[1].type, pc.ParameterType.DOUBLE)
    self.assertEqual(space.parameters[1].bounds, (2.0, 16.0))
    self.assertEqual(space.parameters[1].scale_type, pc.ScaleType.REVERSE_LOG)
    self.assertEmpty(space.parameters[1].matching_parent_values)
    self.assertEmpty(space.parameters[1].child_parameter_configs)
    with self.assertRaisesRegex(ValueError, 'feasible_values is invalid.*'):
      _ = space.parameters[1].feasible_values
    self.assertEqual(space.parameters[1].default_value, 4.0)

  def testMultidimensionalParameters(self):
    space = base_study_config.SearchSpace()
    self.assertEmpty(space.parameters)
    selector0 = space.select_root().add_float_param(
        'f', 1.0, 15.0, default_value=3.0, scale_type=pc.ScaleType.LOG, index=0)
    selector1 = space.select_root().add_float_param(
        'f',
        2.0,
        10.0,
        default_value=4.0,
        scale_type=pc.ScaleType.LINEAR,
        index=1)
    # Test the returned selectors.
    self.assertEqual(selector0.path_string, '')
    self.assertEqual(selector0.parameter_name, 'f[0]')
    self.assertEqual(selector0.parameter_values, [])
    self.assertEqual(selector1.path_string, '')
    self.assertEqual(selector1.parameter_name, 'f[1]')
    self.assertEqual(selector1.parameter_values, [])
    # Test the search space.
    self.assertLen(space.parameters, 2)
    self.assertEqual(space.parameters[0].name, 'f[0]')
    self.assertEqual(space.parameters[0].type, pc.ParameterType.DOUBLE)
    self.assertEqual(space.parameters[0].bounds, (1.0, 15.0))
    self.assertEqual(space.parameters[0].scale_type, pc.ScaleType.LOG)
    self.assertEmpty(space.parameters[0].matching_parent_values)
    self.assertEmpty(space.parameters[0].child_parameter_configs)
    with self.assertRaisesRegex(ValueError, 'feasible_values is invalid.*'):
      _ = space.parameters[0].feasible_values
    self.assertEqual(space.parameters[0].default_value, 3.0)

    self.assertEqual(space.parameters[1].name, 'f[1]')
    self.assertEqual(space.parameters[1].type, pc.ParameterType.DOUBLE)
    self.assertEqual(space.parameters[1].bounds, (2.0, 10.0))
    self.assertEqual(space.parameters[1].scale_type, pc.ScaleType.LINEAR)
    self.assertEmpty(space.parameters[1].matching_parent_values)
    self.assertEmpty(space.parameters[1].child_parameter_configs)
    with self.assertRaisesRegex(ValueError, 'feasible_values is invalid.*'):
      _ = space.parameters[1].feasible_values
    self.assertEqual(space.parameters[1].default_value, 4.0)

  def testConditionalParameters(self):
    space = base_study_config.SearchSpace()
    self.assertEmpty(space.parameters)
    root = space.select_root()
    root.add_categorical_param(
        'model_type', ['linear', 'dnn'], default_value='dnn')
    # Test the selector.
    self.assertEqual(root.path_string, '')
    self.assertEqual(root.parameter_name, '')
    self.assertEqual(root.parameter_values, [])
    # Test the search space.
    self.assertLen(space.parameters, 1)
    self.assertEqual(space.parameters[0].name, 'model_type')
    self.assertEqual(space.parameters[0].type, pc.ParameterType.CATEGORICAL)
    with self.assertRaisesRegex(ValueError,
                                'Accessing bounds of a categorical.*'):
      _ = space.parameters[0].bounds
    self.assertIsNone(space.parameters[0].scale_type)
    self.assertEmpty(space.parameters[0].matching_parent_values)
    self.assertEmpty(space.parameters[0].child_parameter_configs)
    self.assertEqual(space.parameters[0].feasible_values, ['dnn', 'linear'])
    self.assertEqual(space.parameters[0].default_value, 'dnn')

    dnn = root.select('model_type', ['dnn'])
    # Test the selector.
    self.assertEqual(dnn.path_string, '')
    self.assertEqual(dnn.parameter_name, 'model_type')
    self.assertEqual(dnn.parameter_values, ['dnn'])
    dnn.add_float_param(
        'learning_rate',
        0.0001,
        1.0,
        default_value=0.001,
        scale_type=base_study_config.ScaleType.LOG)
    # Test the search space.
    self.assertLen(space.parameters, 1)

    linear = root.select('model_type', ['linear'])
    # Test the selector.
    self.assertEqual(linear.path_string, '')
    self.assertEqual(linear.parameter_name, 'model_type')
    self.assertEqual(linear.parameter_values, ['linear'])
    linear.add_float_param(
        'learning_rate',
        0.1,
        1.0,
        default_value=0.1,
        scale_type=base_study_config.ScaleType.LOG)
    # Test the search space.
    self.assertLen(space.parameters, 1)

    dnn_optimizer = dnn.add_categorical_param('optimizer_type',
                                              ['adam', 'adagrad'])
    # Test the search space.
    self.assertLen(space.parameters, 1)
    # Test the selector.
    self.assertEqual(dnn_optimizer.path_string, 'model_type=dnn')
    self.assertEqual(dnn_optimizer.parameter_name, 'optimizer_type')
    self.assertEqual(dnn_optimizer.parameter_values, [])

    # Chained select() calls, path length of 1.
    lr = root.select('model_type', ['dnn']).select(
        'optimizer_type', ['adam']).add_float_param(
            'learning_rate',
            0.1,
            1.0,
            default_value=0.1,
            scale_type=base_study_config.ScaleType.LOG)
    # Test the search space.
    self.assertLen(space.parameters, 1)
    # Test the selector.
    self.assertEqual(lr.parameter_name, 'learning_rate')
    self.assertEqual(lr.parameter_values, [])
    self.assertEqual(lr.path_string, 'model_type=dnn/optimizer_type=adam')

    # Chained select() calls, path length of 2.
    ko = root.select('model_type', ['dnn']).select('optimizer_type',
                                                   ['adam']).add_bool_param(
                                                       'use_keras_optimizer',
                                                       default_value=False)
    # Test the search space.
    self.assertLen(space.parameters, 1)
    # Test the selector.
    self.assertEqual(ko.parameter_name, 'use_keras_optimizer')
    self.assertEqual(ko.parameter_values, [])
    self.assertEqual(ko.path_string, 'model_type=dnn/optimizer_type=adam')

    ko.select_values(['True'])
    self.assertEqual(ko.parameter_values, ['True'])

    selector = ko.add_float_param('keras specific', 1.3, 2.4, default_value=2.1)
    # Test the search space.
    self.assertLen(space.parameters, 1)
    # Test the selector.
    self.assertEqual(selector.parameter_name, 'keras specific')
    self.assertEqual(selector.parameter_values, [])
    self.assertEqual(
        selector.path_string,
        'model_type=dnn/optimizer_type=adam/use_keras_optimizer=True')

    # Selects more than one node.
    # selectors = dnn.select_all('optimizer_type', ['adam'])
    # self.assertLen(selectors, 2)

  def testConditionalParametersWithReturnedSelectors(self):
    space = base_study_config.SearchSpace()
    self.assertEmpty(space.parameters)
    root = space.select_root()
    model_type = root.add_categorical_param('model_type', ['linear', 'dnn'])
    learning_rate = model_type.select_values(['dnn']).add_float_param(
        'learning_rate',
        0.1,
        1.0,
        default_value=0.001,
        scale_type=base_study_config.ScaleType.LOG)
    # Test the search space.
    self.assertLen(space.parameters, 1)
    # Test the selectors.
    self.assertEqual(model_type.parameter_values, ['dnn'])
    self.assertEqual(learning_rate.parameter_name, 'learning_rate')
    self.assertEqual(learning_rate.parameter_values, [])
    self.assertEqual(learning_rate.path_string, 'model_type=dnn')

    # It is possible to select different values for the same selector.
    optimizer_type = model_type.select_values(['linear',
                                               'dnn']).add_categorical_param(
                                                   'optimizer_type',
                                                   ['adam', 'adagrad'])
    # Test the search space.
    self.assertLen(space.parameters, 1)
    # Test the selectors.
    self.assertEqual(model_type.parameter_values, ['linear', 'dnn'])
    self.assertEqual(optimizer_type.parameter_name, 'optimizer_type')
    self.assertEqual(optimizer_type.parameter_values, [])
    self.assertEqual(optimizer_type.path_string, 'model_type=linear')

  @parameterized.named_parameters(
      ('Multi', 'units[0]', ('units', 0)),
      ('Multi2', 'with_underscore[1]', ('with_underscore', 1)),
      ('NotMulti', 'units', None),
      ('NotMulti2', 'with space', None),
      ('NotMulti3', 'with[8]space', None),
      ('NotMulti4', 'units[0][4]', ('units[0]', 4)),
      ('GinStyle', '_gin.ambient_net_exp_from_vec.block_type[3]',
       ('_gin.ambient_net_exp_from_vec.block_type', 3)),
  )
  def testParseMultiDimensionalParameterName(self, name, expected):
    base_name_index = base_study_config.SearchSpaceSelector.parse_multi_dimensional_parameter_name(
        name)
    self.assertEqual(base_name_index, expected)


class SearchSpaceContainsTest(absltest.TestCase):

  def _space(self):
    space = base_study_config.SearchSpace()
    root = space.select_root()
    root.add_float_param('learning-rate', 1e-4, 1e-2)
    root.add_categorical_param('optimizer', ['adagrad', 'adam', 'experimental'])
    return space

  def testFloatCat1(self):
    self._space().assert_contains(
        trial.ParameterDict({
            'optimizer': 'adagrad',
            'learning-rate': 1e-2
        }))

  def testFloatCat2(self):
    self.assertFalse(self._space().contains(
        trial.ParameterDict({
            'optimizer': 'adagrad',
            'BADPARAM': 1e-2
        })))

  def testFloatCat3(self):
    self.assertFalse(self._space().contains(
        trial.ParameterDict({
            'optimizer': 'adagrad',
            'learning-rate': 1e-2,
            'BADPARAM': 1e-2
        })))

  def testFloatCat4(self):
    self.assertFalse(self._space().contains(
        trial.ParameterDict({
            'optimizer': 'adagrad',
            'learning-rate': 1e2
        })))


if __name__ == '__main__':
  absltest.main()
