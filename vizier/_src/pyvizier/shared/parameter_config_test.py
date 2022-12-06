# Copyright 2022 Google LLC.
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

"""Tests for vizier.pyvizier.shared.parameter_config."""

from typing import Any

from vizier._src.pyvizier.shared import parameter_config as pc
from vizier._src.pyvizier.shared import trial

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
    self.assertLen(p.child_parameter_configs, 2)
    self.assertEmpty(p.matching_parent_values)
    self.assertSameElements(p.child_parameter_configs[0].matching_parent_values,
                            children[0][0])
    self.assertSameElements(p.child_parameter_configs[1].matching_parent_values,
                            children[1][0])

  @parameterized.named_parameters(
      ('FloatParentValues', [([0.5], _child1)]),
      ('StringParentValues', [(['0'], _child1), (['0.0', '1.0'], _child2)]))
  def testIntegerWithInvalid(self, children):
    with self.assertRaises(TypeError):
      _ = pc.ParameterConfig.factory('parent', bounds=(0, 1), children=children)

  @parameterized.named_parameters(
      ('IntParentValues', [([0], _child1), ([1], _child2)]),
      ('FloatParentValues', [([0.0], _child1), ([0.0, 1.0], _child2)]))
  def testDiscreteWithValid(self, children):
    p = pc.ParameterConfig.factory(
        'parent', feasible_values=[0.0, 1.0], children=children)
    self.assertLen(p.child_parameter_configs, 2)
    self.assertEmpty(p.matching_parent_values)
    self.assertSameElements(p.child_parameter_configs[0].matching_parent_values,
                            children[0][0])
    self.assertSameElements(p.child_parameter_configs[1].matching_parent_values,
                            children[1][0])

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
    self.assertLen(p.child_parameter_configs, 2)
    self.assertEmpty(p.matching_parent_values)
    self.assertSameElements(p.child_parameter_configs[0].matching_parent_values,
                            children[0][0])
    self.assertSameElements(p.child_parameter_configs[1].matching_parent_values,
                            children[1][0])

  @parameterized.named_parameters(('StringParentValues', [(['0.0'], _child1),
                                                          (['1.0'], _child2)]))
  def testCategoricalWithInvalid(self, children):
    with self.assertRaises(TypeError):
      _ = pc.ParameterConfig.factory(
          'parent', feasible_values=[0.0, 1.0], children=children)

  def testAddChildren(self):
    children = [(['a'], _child1), (['a', 'b'], _child2)]
    p = pc.ParameterConfig.factory(
        'parent', feasible_values=['a', 'b'], children=children)
    new_children = [
        (['a'], pc.ParameterConfig.factory('double_child2', bounds=(1.0, 2.0))),
        (['b'],
         pc.ParameterConfig.factory(
             'categorical_child', feasible_values=['c', 'd'])),
    ]
    p2 = p.add_children(new_children)
    self.assertLen(p.child_parameter_configs, 2)
    self.assertSameElements([c.name for c in p.child_parameter_configs],
                            [c[1].name for c in children])

    self.assertLen(p2.child_parameter_configs, 4)
    expected_names = [c[1].name for c in children]
    expected_names += [c[1].name for c in new_children]
    got_names = [c.name for c in p2.child_parameter_configs]
    self.assertSameElements(got_names, expected_names)


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
        'pc1', feasible_values=['true', 'false'])
    self.assertEqual(config.contains(value), expected)


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
    # will appear multiple times, but they are indeed separate parameters.
    self.assertEqual(traversed_names, [
        'parent', 'child1', 'grandchild1', 'grandchild1', 'child1',
        'grandchild1', 'grandchild1', 'child2'
    ])


class SearchSpaceTest(parameterized.TestCase):

  def testRootAndSelectRootEqual(self):
    space = pc.SearchSpace()
    space.root.add_float_param('f', 0.0, 1.0)
    space.root.add_int_param('i', 0, 1)
    space.root.add_discrete_param('d', [0.0, 0.5, 1.0])
    space.root.add_categorical_param('c', ['a', 'b', 'c'])

    select_root = space.select_root()
    shortcut_root = space.root
    self.assertEqual(select_root.parameter_name, shortcut_root.parameter_name)
    self.assertEqual(select_root.parameter_values,
                     shortcut_root.parameter_values)
    self.assertEqual(select_root.path_string, shortcut_root.path_string)

  def testAddFloatParamMinimal(self):
    # Remove this test once we deprecate select_root().
    space = pc.SearchSpace()
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

  def testAddFloatParamMinimalFromRoot(self):
    """Same test as above, but using `space.root` property."""
    space = pc.SearchSpace()
    self.assertEmpty(space.parameters)
    selector = space.root.add_float_param('f1', 1.0, 15.0)
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

    _ = space.root.add_float_param('f2', 2.0, 16.0)
    self.assertLen(space.parameters, 2)
    self.assertEqual(space.parameters[0].name, 'f1')
    self.assertEqual(space.parameters[0].type, pc.ParameterType.DOUBLE)
    self.assertEqual(space.parameters[0].bounds, (1.0, 15.0))
    self.assertEqual(space.parameters[1].name, 'f2')
    self.assertEqual(space.parameters[1].type, pc.ParameterType.DOUBLE)
    self.assertEqual(space.parameters[1].bounds, (2.0, 16.0))

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

  def testAddTwoFloatParams(self):
    space = pc.SearchSpace()
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
    space = pc.SearchSpace()
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
    space = pc.SearchSpace()
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
    space = pc.SearchSpace()
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
        scale_type=pc.ScaleType.LOG)
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
        scale_type=pc.ScaleType.LOG)
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
    lr = root.select('model_type',
                     ['dnn']).select('optimizer_type',
                                     ['adam']).add_float_param(
                                         'learning_rate',
                                         0.1,
                                         1.0,
                                         default_value=0.1,
                                         scale_type=pc.ScaleType.LOG)
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
    space = pc.SearchSpace()
    self.assertEmpty(space.parameters)
    root = space.select_root()
    model_type = root.add_categorical_param('model_type', ['linear', 'dnn'])
    learning_rate = model_type.select_values(['dnn']).add_float_param(
        'learning_rate',
        0.1,
        1.0,
        default_value=0.001,
        scale_type=pc.ScaleType.LOG)
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
    base_name_index = pc.SearchSpaceSelector.parse_multi_dimensional_parameter_name(
        name)
    self.assertEqual(base_name_index, expected)


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
