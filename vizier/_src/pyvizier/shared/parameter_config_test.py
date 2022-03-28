"""Tests for vizier.pyvizier.shared.parameter_config."""

from typing import Any

from vizier._src.pyvizier.shared import parameter_config as pc
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


if __name__ == '__main__':
  absltest.main()
