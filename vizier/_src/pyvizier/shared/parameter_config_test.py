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

"""Tests for vizier.pyvizier.shared.parameter_config."""

from typing import Any

from absl import logging
from vizier._src.pyvizier.shared import parameter_config as pc
from vizier._src.pyvizier.shared import trial
from vizier.testing import test_studies

from absl.testing import absltest
from absl.testing import parameterized


class ParameterConfigFactoryTest(parameterized.TestCase):

  def testCreatesDoubleConfig(self):
    parameter_config = pc.ParameterConfig.factory(
        'name',
        bounds=(-1.0, 1.0),
        scale_type=pc.ScaleType.LINEAR,
        default_value=0.1)
    self.assertEqual(parameter_config.name, 'name')
    self.assertEqual(parameter_config.type, pc.ParameterType.DOUBLE)
    self.assertEqual(parameter_config.bounds, (-1, 1))
    self.assertEqual(parameter_config.scale_type, pc.ScaleType.LINEAR)
    self.assertEqual(parameter_config.default_value, 0.1)
    self.assertIsInstance(parameter_config.default_value, float)
    with self.assertRaises(ValueError):
      _ = parameter_config.feasible_values

    self.assertEqual(parameter_config.continuify(), parameter_config)

  def testCreatesIntegerConfig(self):
    parameter_config = pc.ParameterConfig.factory(
        'name', bounds=(1, 3), scale_type=pc.ScaleType.LOG, default_value=1)
    self.assertEqual(parameter_config.name, 'name')
    self.assertEqual(parameter_config.type, pc.ParameterType.INTEGER)
    self.assertEqual(parameter_config.feasible_values, [1, 2, 3])
    self.assertEqual(parameter_config.bounds, (1, 3))
    self.assertEqual(parameter_config.scale_type, pc.ScaleType.LOG)
    self.assertEqual(parameter_config.default_value, 1)
    self.assertIsInstance(parameter_config.default_value, int)

    self.assertEqual(
        parameter_config.continuify(),
        pc.ParameterConfig.factory(
            'name',
            bounds=(1.0, 3.0),
            scale_type=pc.ScaleType.LOG,
            default_value=1.0))

  def testCreatesDiscreteConfig(self):
    feasible_values = (-1, 3, 2)
    parameter_config = pc.ParameterConfig.factory(
        'name',
        feasible_values=feasible_values,
        scale_type=pc.ScaleType.UNIFORM_DISCRETE,
        default_value=2,
        external_type=pc.ExternalType.INTEGER)
    self.assertEqual(parameter_config.name, 'name')
    self.assertEqual(parameter_config.type, pc.ParameterType.DISCRETE)
    self.assertEqual(parameter_config.feasible_values, [-1, 2, 3])
    self.assertEqual(parameter_config.bounds, (-1, 3))
    self.assertEqual(parameter_config.scale_type, pc.ScaleType.UNIFORM_DISCRETE)
    self.assertEqual(parameter_config.default_value, 2)
    self.assertIsInstance(parameter_config.default_value, float)
    self.assertEqual(parameter_config.external_type, pc.ExternalType.INTEGER)

    self.assertEqual(
        parameter_config.continuify(),
        pc.ParameterConfig.factory(
            'name', bounds=(-1.0, 3.0), default_value=2.0))

  def testCreatesCategoricalConfig(self):
    feasible_values = ('b', 'a', 'c')
    parameter_config = pc.ParameterConfig.factory(
        'name', feasible_values=feasible_values, default_value='c')
    self.assertEqual(parameter_config.name, 'name')
    self.assertEqual(parameter_config.feasible_values, ['a', 'b', 'c'])
    self.assertEqual(parameter_config.default_value, 'c')
    with self.assertRaises(ValueError):
      _ = parameter_config.bounds

  def testCreatesDoubleConfigIntDefault(self):
    parameter_config = pc.ParameterConfig.factory(
        'name',
        bounds=(-1.0, 1.0),
        scale_type=pc.ScaleType.LINEAR,
        default_value=1)
    self.assertEqual(parameter_config.default_value, 1.0)
    self.assertIsInstance(parameter_config.default_value, float)

  def testCreatesDiscreteConfigDoubleDefault(self):
    feasible_values = (-1, 3, 2)
    parameter_config = pc.ParameterConfig.factory(
        'name',
        feasible_values=feasible_values,
        scale_type=pc.ScaleType.UNIFORM_DISCRETE,
        default_value=2.0)
    self.assertEqual(parameter_config.default_value, 2.0)
    self.assertIsInstance(parameter_config.default_value, float)

  def testCreatesIntegerConfigDoubleDefault(self):
    parameter_config = pc.ParameterConfig.factory(
        'name', bounds=(1, 3), scale_type=pc.ScaleType.LOG, default_value=2.0)
    self.assertEqual(parameter_config.default_value, 2.0)
    self.assertIsInstance(parameter_config.default_value, int)

  def testCreatesIntegerConfigInvalidDoubleDefault(self):
    with self.assertRaisesRegex(ValueError, 'default_value for an.*'):
      pc.ParameterConfig.factory(
          'name',
          bounds=(1, 3),
          scale_type=pc.ScaleType.LOG,
          default_value=2.0001)

  def testCreatesCategoricalConfigNoDefault(self):
    feasible_values = ('b', 'a', 'c')
    parameter_config = pc.ParameterConfig.factory(
        'name', feasible_values=feasible_values)
    self.assertIsNone(parameter_config.default_value)

  def testCreatesCategoricalConfigBadDefault(self):
    feasible_values = ('b', 'a', 'c')
    with self.assertRaisesRegex(ValueError,
                                'default_value has an incorrect type.*'):
      pc.ParameterConfig.factory(
          'name', feasible_values=feasible_values, default_value=0.1)

  def testRaisesErrorWhenNameIsEmpty(self):
    with self.assertRaises(ValueError):
      _ = pc.ParameterConfig.factory('', bounds=(-1.0, 1.0))

  def testRaisesErrorWhenOverSpecified(self):
    with self.assertRaises(ValueError):
      _ = pc.ParameterConfig.factory(
          'name', bounds=(-1.0, 1.0), feasible_values=['a', 'b', 'c'])

  @parameterized.named_parameters(
      ('HaveInfinity', (-float('inf'), 1)), ('HaveNan', (1, float('nan'))),
      ('HaveMixedTypes', (1, float(1))), ('AreWronglyOrdered', (1, -1)))
  def testRaisesErrorWhenBounds(self, bounds):
    with self.assertRaises(ValueError):
      _ = pc.ParameterConfig.factory('name', bounds=bounds)

  @parameterized.named_parameters(('HaveDuplicateCategories', ['a', 'a', 'b']),
                                  ('HaveDuplicateNumbers', [1.0, 2.0, 2.0]),
                                  ('HaveMixedTypes', ['a', 1, 2]))
  def testRaisesErrorWhenFeasibleValues(self, feasible_values):
    with self.assertRaises(ValueError):
      _ = pc.ParameterConfig.factory('name', feasible_values=feasible_values)


_child1 = pc.ParameterConfig.factory('double_child', bounds=(0.0, 1.0))
_child2 = pc.ParameterConfig.factory('integer_child', bounds=(0, 1))


class ParameterConfigFactoryTestWithChildren(parameterized.TestCase):

  @parameterized.named_parameters(
      ('IntParentValues', [([0], _child1), ([0, 1], _child2)]),
      ('FloatParentValues', [([0.0], _child1), ([0.0, 1.0], _child2)]))
  def testIntegerWithValid(self, children):
    p = pc.ParameterConfig.factory('parent', bounds=(0, 1), children=children)
    self.assertCountEqual(p.subspace(0.0).parameters, [_child1, _child2])
    self.assertCountEqual(p.subspace(1.0).parameters, [_child2])

  @parameterized.named_parameters(
      ('FloatParentValues', [([0.5], _child1)]),
      ('StringParentValues', [(['0'], _child1), (['0.0', '1.0'], _child2)]))
  def testIntegerWithInvalid(self, children):
    with self.assertRaises(TypeError):
      _ = pc.ParameterConfig.factory('parent', bounds=(0, 1), children=children)

  @parameterized.named_parameters(
      ('IntParentValues', [([0], _child1), ([0, 1], _child2)]),
      ('FloatParentValues', [([0.0], _child1), ([0.0, 1.0], _child2)]))
  def testDiscreteWithValid(self, children):
    p = pc.ParameterConfig.factory(
        'parent', feasible_values=[0.0, 1.0], children=children)
    self.assertCountEqual(p.subspace(0.0).parameters, [_child1, _child2])
    self.assertCountEqual(p.subspace(1.0).parameters, [_child2])

  @parameterized.named_parameters(('StringParentValues', [(['0.0'], _child1),
                                                          (['0.0',
                                                            '1.0'], _child2)]))
  def testDiscreteWithInvalid(self, children):
    with self.assertRaises(TypeError):
      _ = pc.ParameterConfig.factory(
          'parent', feasible_values=[0.0, 1.0], children=children)

  @parameterized.named_parameters(  # pyformat: disable
      ('StringParentValues', [(['a'], _child1), (['a', 'b'], _child2)]))
  def testCategoricalWithValid(self, children):
    p = pc.ParameterConfig.factory(
        'parent', feasible_values=['a', 'b'], children=children)
    self.assertCountEqual(p.subspace('a').parameters, [_child1, _child2])
    self.assertCountEqual(p.subspace('b').parameters, [_child2])

  @parameterized.named_parameters(('StringParentValues', [(['0.0'], _child1),
                                                          (['1.0'], _child2)]))
  def testCategoricalWithInvalid(self, children):
    with self.assertRaises(TypeError):
      _ = pc.ParameterConfig.factory(
          'parent', feasible_values=[0.0, 1.0], children=children)


class MergeTest(parameterized.TestCase):

  def test_merge_bounds(self):
    pc1 = pc.ParameterConfig.factory('pc1', bounds=(0.0, 2.0))
    pc2 = pc.ParameterConfig.factory('pc2', bounds=(-1.0, 1.0))
    self.assertEqual(
        pc.ParameterConfig.merge(pc1, pc2),
        pc.ParameterConfig.factory('pc1', bounds=(-1.0, 2.0)))

  def test_merge_discrete(self):
    pc1 = pc.ParameterConfig.factory(
        'pc1', feasible_values=[0.0, 2.0], scale_type=pc.ScaleType.LINEAR)
    pc2 = pc.ParameterConfig.factory('pc2', feasible_values=[-1.0, 0.0])
    self.assertEqual(
        pc.ParameterConfig.merge(pc1, pc2),
        pc.ParameterConfig.factory(
            'pc1',
            feasible_values=[-1.0, 0.0, 2.0],
            scale_type=pc.ScaleType.LINEAR))

  def test_merge_categorical(self):
    pc1 = pc.ParameterConfig.factory('pc1', feasible_values=['a', 'b'])
    pc2 = pc.ParameterConfig.factory('pc2', feasible_values=['a', 'c'])
    self.assertEqual(
        pc.ParameterConfig.merge(pc1, pc2),
        pc.ParameterConfig.factory('pc1', feasible_values=['a', 'b', 'c']))


class ParameterConfigContainsTest(parameterized.TestCase):

  @parameterized.parameters((1.0, True), (-2.0, False), (3.0, False))
  def testFloat(self, value: Any, expected: bool):
    config = pc.ParameterConfig.factory('pc1', bounds=(-1., 2.))
    self.assertEqual(config.contains(value), expected)

  @parameterized.parameters((1, True), (-2, False), (3, False), (1.5, False))
  def testInt(self, value: Any, expected: bool):
    config = pc.ParameterConfig.factory('pc1', bounds=(-1, 2))
    self.assertEqual(config.contains(value), expected)

  @parameterized.parameters((1.0, False), (2, True), (-1, True))
  def testDiscrete(self, value: Any, expected: bool):
    config = pc.ParameterConfig.factory('pc1', feasible_values=[-1., 0., 2.])
    self.assertEqual(config.contains(value), expected)

  @parameterized.parameters(('a', True), ('b', False), ('c', False))
  def testCategorical(self, value: Any, expected: bool):
    config = pc.ParameterConfig.factory(
        'pc1', feasible_values=['a', 'aa', 'aaa'])
    self.assertEqual(config.contains(value), expected)

  @parameterized.parameters((True, True), ('a', False), (0, False))
  def testBoolean(self, value: Any, expected: bool):
    config = pc.ParameterConfig.factory(
        'pc1', feasible_values=[trial.TRUE_VALUE, trial.FALSE_VALUE])
    self.assertEqual(config.contains(value), expected)


class ParameterConfigPropertyTest(parameterized.TestCase):

  @parameterized.parameters(
      ((-1.0, 1.0), None), ((-1, 1), None), ((-2.0, -2.0), -2.0), ((-2, -2), -2)
  )
  def testFloatandInt(self, bounds: Any, expected: Any):
    config = pc.ParameterConfig.factory('pc1', bounds=bounds)
    value = config.deterministic_value
    if expected is None:
      self.assertIsNone(value)
    else:
      self.assertEqual(value, expected)

  @parameterized.parameters(
      ([-1.0, 2.0], None), ([-1.0], -1.0), (['a', 'b'], None), (['a'], 'a')
  )
  def testDiscreteandCategorical(self, feasible_values: Any, expected: Any):
    config = pc.ParameterConfig.factory('pc1', feasible_values=feasible_values)
    value = config.deterministic_value
    if expected is None:
      self.assertIsNone(value)
    else:
      self.assertEqual(value, expected)


class TraverseTest(parameterized.TestCase):

  @parameterized.named_parameters(('ShowChildrenTrue', True),
                                  ('ShowChildrenFalse', False))
  def testTraverse(self, show_children):
    grandchild1 = pc.ParameterConfig.factory('grandchild1', bounds=(-1.0, 1.0))
    grandchildren = [(['a'], grandchild1), (['b'], grandchild1)]
    child1 = pc.ParameterConfig.factory(
        'child1', feasible_values=['a', 'b'], children=grandchildren)

    child2 = pc.ParameterConfig.factory('child2', bounds=(0.0, 1.0))
    children = [([0], child1), ([1], child1), ([0, 1], child2)]
    parent = pc.ParameterConfig.factory(
        'parent', bounds=(0, 1), children=children)
    traversed_names = [
        pc.name for pc in parent.traverse(show_children=show_children)
    ]
    # Some parameter names are reused for separate child nodes, so they
    # will appear multiple times.
    self.assertEqual(traversed_names, [
        'parent', 'child1', 'grandchild1', 'grandchild1', 'child2', 'child1',
        'grandchild1', 'grandchild1', 'child2'
    ])


class SearchSpaceTest(parameterized.TestCase):
  """Check basic functionalities."""

  def testRootAndSelectRootEqual(self):
    space = pc.SearchSpace()
    select_root = space.select_root()
    shortcut_root = space.root
    self.assertIs(select_root._selected[0], shortcut_root._selected[0])

  def testAddFloatParamMinimal(self):
    # Remove this test once we deprecate select_root().
    space = pc.SearchSpace()
    self.assertEmpty(space.parameters)
    _ = space.select_root().add_float_param('f1', 1.0, 15.0)
    _ = space.select_root().add_float_param('f2', 2.0, 16.0)

    self.assertLen(space.parameters, 2)
    self.assertCountEqual(space.parameter_names, ['f1', 'f2'])

  def testMultidimensionalParameters(self):
    space = pc.SearchSpace()
    _ = space.select_root().add_float_param(
        'f', 1.0, 15.0, default_value=3.0, scale_type=pc.ScaleType.LOG, index=0)
    _ = space.select_root().add_float_param(
        'f',
        2.0,
        10.0,
        default_value=4.0,
        scale_type=pc.ScaleType.LINEAR,
        index=1)
    self.assertCountEqual(space.parameter_names, ['f[0]', 'f[1]'])


class SearchSpaceAddParamtest(parameterized.TestCase):
  """Check `add_xx_param` methods relay the arguments correctly."""

  def testAddFloatParam(self):
    space = pc.SearchSpace()
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
    space = pc.SearchSpace()
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
    space = pc.SearchSpace()
    self.assertEmpty(space.parameters)
    _ = space.select_root().add_discrete_param(
        'd1', [15.1, 21.0, 101], default_value=15.1)
    self.assertLen(space.parameters, 1)
    self.assertEqual(space.parameters[0].external_type, pc.ExternalType.FLOAT)

  def testAddBooleanParam(self):
    """Test a Boolean parameter."""
    space = pc.SearchSpace()
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
    space = pc.SearchSpace()
    self.assertEmpty(space.parameters)
    _ = space.select_root().add_bool_param('b1', default_value=False)
    self.assertLen(space.parameters, 1)
    self.assertEqual(space.parameters[0].default_value, 'False')

  def testAddCustomParam(self):
    """Test a Boolean parameter."""
    space = pc.SearchSpace()
    self.assertEmpty(space.parameters)
    _ = space.select_root().add_custom_param('c1', default_value='default')
    self.assertLen(space.parameters, 1)
    self.assertEqual(space.parameters[0].name, 'c1')
    self.assertEqual(space.parameters[0].default_value, 'default')

  def testConditionalParameters(self):
    space = pc.SearchSpace()
    self.assertEmpty(space.parameters)
    root = space.select_root()
    root.add_categorical_param(
        'model_type', ['linear', 'dnn'], default_value='dnn')

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
    dnn.add_float_param(
        'learning_rate',
        0.0001,
        1.0,
        default_value=0.001,
        scale_type=pc.ScaleType.LOG)
    # Test the search space.
    self.assertLen(space.parameters, 1)
    self.assertLen(space.get('model_type').subspaces(), 1)

    linear = root.select('model_type', ['linear'])
    linear.add_float_param(
        'learning_rate',
        0.1,
        1.0,
        default_value=0.1,
        scale_type=pc.ScaleType.LOG)
    # Test the search space.
    self.assertLen(space.parameters, 1)
    self.assertLen(space.get('model_type').subspaces(), 2)

    _ = dnn.add_categorical_param('optimizer_type', ['adam', 'adagrad'])
    # Test the search space.
    self.assertLen(space.parameters, 1)

    # Chained select() calls, path length of 1.
    selected = root.select('model_type',
                           ['dnn']).select('optimizer_type',
                                           ['adam']).add_float_param(
                                               'learning_rate',
                                               0.1,
                                               1.0,
                                               default_value=0.1,
                                               scale_type=pc.ScaleType.LOG)
    self.assertLen(selected, 1)

    # Test the search space.
    self.assertLen(space.parameters, 1)

    # Chained select() calls, path length of 2.
    ko = root.select('model_type', ['dnn']).select('optimizer_type',
                                                   ['adam']).add_bool_param(
                                                       'use_keras_optimizer',
                                                       default_value=False)
    self.assertLen(ko, 1)
    # Test the search space.
    self.assertLen(space.parameters, 1)

    ko2 = ko.select_values(['True'])
    _ = ko2.add_float_param('keras specific', 1.3, 2.4, default_value=2.1)
    # Test the search space.
    self.assertLen(space.parameters, 1)

  def testConditionalParametersWithReturnedSelectors(self):
    space = pc.SearchSpace()
    self.assertEmpty(space.parameters)
    root = space.select_root()
    model_type = root.add_categorical_param('model_type', ['linear', 'dnn'])
    _ = model_type.select_values(['dnn']).add_float_param(
        'learning_rate',
        0.1,
        1.0,
        default_value=0.001,
        scale_type=pc.ScaleType.LOG)
    # Test the search space.
    self.assertLen(space.parameters, 1)

    # It is possible to select different values for the same selector.
    self.assertLen(
        model_type.select_values(['linear', 'dnn']).add_categorical_param(
            'optimizer_type', ['adam', 'adagrad']), 2)
    # Test the search space.
    self.assertLen(space.parameters, 1)

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
    base_name_index = pc.SearchSpaceSelector.parse_multi_dimensional_parameter_name(
        name)
    self.assertEqual(base_name_index, expected)

  def testValidateCategoricalInput(self):
    space = pc.SearchSpace()
    root = space.select_root()
    with self.assertRaises(ValueError):
      root.add_categorical_param('categorical', ['3.2', '2', 5])


class FlattenAndMergeTest(absltest.TestCase):

  def testFlattenAndMerge(self):
    space = test_studies.conditional_automl_space()
    parameters = space.root.select_all().merge()
    logging.info('Merged: %s', parameters)
    self.assertCountEqual(
        [p.name for p in parameters],
        [
            'model_type',
            'learning_rate',
            'optimizer_type',
            'use_special_logic',
            'special_logic_parameter',
        ],
    )


class SearchSpaceContainsTest(absltest.TestCase):

  def _space(self):
    space = pc.SearchSpace()
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
