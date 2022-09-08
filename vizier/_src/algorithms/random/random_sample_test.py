"""Tests for random_sample."""

from jax import random
from vizier import pyvizier as vz
from vizier._src.algorithms.random import random_sample

from absl.testing import absltest
from absl.testing import parameterized


class RandomSampleTest(parameterized.TestCase):

  def setUp(self):
    super(RandomSampleTest, self).setUp()
    self.key = random.PRNGKey(0)

  @parameterized.named_parameters(
      dict(testcase_name='prob=0', value=0.2, target=0.0),
      dict(testcase_name='prob=1.0', value=-0.5, target=0.0),
      dict(testcase_name='prob=1.5', value=0.6, target=1.0),
      dict(testcase_name='prob=-0.1', value=3.0, target=2.0),
  )
  def test_get_closest_element(self, value, target):
    items = [0.0, 1.0, 2.0]
    self.assertEqual(random_sample.get_closest_element(items, value), target)

  def test_shuffle_list(self):
    items = ['a', 'b', 'c', 'd']
    _, shuffled_items = random_sample.shuffle_list(self.key, items)
    # Check that all items appear once
    items_set = set(items)
    for item in shuffled_items:
      self.assertIn(item, items_set)
      items_set.remove(item)

  def test_sample_uniform(self):
    new_key, value1 = random_sample.sample_uniform(self.key)
    self.assertTrue(0.0 <= value1 <= 1.0)
    _, value2 = random_sample.sample_uniform(new_key)
    self.assertNotEqual(value1, value2)
    _, value3 = random_sample.sample_uniform(self.key, 10.0, 20.0)
    self.assertTrue(10.0 <= value3 <= 20.0)

  def test_sample_bernoulli(self):
    _, value1 = random_sample.sample_bernoulli(self.key, 0.5)
    self.assertIn(value1, [0, 1])
    _, value2 = random_sample.sample_bernoulli(self.key, 0.0)
    self.assertEqual(value2, 1)
    _, value3 = random_sample.sample_bernoulli(self.key, 1.0)
    self.assertEqual(value3, 0)
    _, value4 = random_sample.sample_bernoulli(self.key, 0.5, 'A', 'B')
    self.assertIn(value4, ['A', 'B'])

  def test_sample_integer(self):
    _, value1 = random_sample.sample_integer(self.key, 5, 5)
    self.assertEqual(value1, 5)
    _, value2 = random_sample.sample_integer(self.key, 0, 10)
    self.assertIn(value2, list(range(0, 11)))

  def test_sample_categorical(self):
    _, value1 = random_sample.sample_categorical(self.key, ['A'])
    self.assertEqual(value1, 'A')
    _, value2 = random_sample.sample_categorical(self.key, ['A', 'B', 'C'])
    self.assertIn(value2, ['A', 'B', 'C'])

  def test_sample_discrete(self):
    _, value1 = random_sample.sample_discrete(self.key, [2.5])
    self.assertEqual(value1, 2.5)
    _, value2 = random_sample.sample_discrete(self.key, [0.0, 1.0, 10.0])
    self.assertIn(value2, [0.0, 1.0, 10.0])

  def test_sample_value_integer(self):
    int_param = vz.ParameterConfig.factory('int', bounds=(0, 10))
    _, value = random_sample._sample_value(self.key, int_param)
    self.assertIn(value, list(range(0, 11)))

  def test_sample_value_discrete(self):
    discrete_param = vz.ParameterConfig.factory(
        'discrete', feasible_values=[1.0, 2.0, 10.0])
    _, value = random_sample._sample_value(self.key, discrete_param)
    self.assertIn(value, [1.0, 2.0, 10.0])

  def test_sample_value_float(self):
    float_param = vz.ParameterConfig.factory('float', bounds=(0.0, 1.0))
    _, value = random_sample._sample_value(self.key, float_param)
    self.assertTrue(0.0 <= value <= 1.0)

  def test_sample_value_categorical(self):
    categorical_param = vz.ParameterConfig.factory(
        'categorical', feasible_values=['a', 'b', 'c'])
    _, value = random_sample._sample_value(self.key, categorical_param)
    self.assertIn(value, ['a', 'b', 'c'])

  def test_sample_input_parameters(self):
    space = vz.SearchSpace()
    root = space.root
    root.add_bool_param('b1')
    root.add_discrete_param('d1', [1.0, 2.0, 10.0])
    root.add_float_param('f1', 0.0, 15.0, scale_type=vz.ScaleType.LINEAR)
    root.add_float_param('f2', 100.0, 200.0, scale_type=vz.ScaleType.LINEAR)
    root.add_int_param('i1', 0, 10, scale_type=vz.ScaleType.LINEAR)
    root.add_categorical_param('c1', ['a', 'b', 'c'])
    _, parameter_dict = random_sample.sample_input_parameters(self.key, space)
    self.assertIn(parameter_dict['b1'].value, ['True', 'False'])
    self.assertIn(parameter_dict['d1'].value, [1.0, 2.0, 10.0])
    self.assertTrue(0.0 <= parameter_dict['f1'].value <= 15.0)
    self.assertTrue(100.0 <= parameter_dict['f2'].value <= 200.0)
    self.assertIn(parameter_dict['i1'].value, list(range(0, 11)))
    self.assertIn(parameter_dict['c1'].value, ['a', 'b', 'c'])


if __name__ == '__main__':
  absltest.main()
