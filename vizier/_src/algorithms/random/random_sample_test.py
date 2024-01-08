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

"""Tests for random_sample."""

import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.random import random_sample

from absl.testing import absltest
from absl.testing import parameterized


class RandomSampleTest(parameterized.TestCase):

  def setUp(self):
    super(RandomSampleTest, self).setUp()
    self.rng = np.random.default_rng(0)

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
    shuffled_items = random_sample.shuffle_list(self.rng, items)
    # Check that all items appear once
    items_set = set(items)
    for item in shuffled_items:
      self.assertIn(item, items_set)
      items_set.remove(item)

  def test_sample_uniform(self):
    value1 = random_sample.sample_uniform(self.rng)
    self.assertTrue(0.0 <= value1 <= 1.0)
    value2 = random_sample.sample_uniform(self.rng)
    self.assertNotEqual(value1, value2)
    value3 = random_sample.sample_uniform(self.rng, 10.0, 20.0)
    self.assertTrue(10.0 <= value3 <= 20.0)

  def test_sample_bernoulli(self):
    value1 = random_sample.sample_bernoulli(self.rng, 0.5)
    self.assertIn(value1, [0, 1])
    value2 = random_sample.sample_bernoulli(self.rng, 0.0)
    self.assertEqual(value2, 1)
    value3 = random_sample.sample_bernoulli(self.rng, 1.0)
    self.assertEqual(value3, 0)
    value4 = random_sample.sample_bernoulli(self.rng, 0.5, 'A', 'B')
    self.assertIn(value4, ['A', 'B'])

  def test_sample_integer(self):
    value1 = random_sample.sample_integer(self.rng, 5, 5)
    self.assertEqual(value1, 5)
    value2 = random_sample.sample_integer(self.rng, 0, 10)
    self.assertIn(value2, list(range(0, 11)))

  def test_sample_categorical(self):
    value1 = random_sample.sample_categorical(self.rng, ['A'])
    self.assertEqual(value1, 'A')
    value2 = random_sample.sample_categorical(self.rng, ['A', 'B', 'C'])
    self.assertIn(value2, ['A', 'B', 'C'])

  def test_sample_discrete(self):
    value1 = random_sample.sample_discrete(self.rng, [2.5])
    self.assertEqual(value1, 2.5)
    value2 = random_sample.sample_discrete(self.rng, [0.0, 1.0, 10.0])
    self.assertIn(value2, [0.0, 1.0, 10.0])

  def test_sample_value_integer(self):
    int_param = vz.ParameterConfig.factory('int', bounds=(0, 10))
    value = random_sample._sample_value(self.rng, int_param)
    self.assertIn(value, list(range(0, 11)))

  def test_sample_value_discrete(self):
    discrete_param = vz.ParameterConfig.factory(
        'discrete', feasible_values=[1.0, 2.0, 10.0])
    value = random_sample._sample_value(self.rng, discrete_param)
    self.assertIn(value, [1.0, 2.0, 10.0])

  def test_sample_value_float(self):
    float_param = vz.ParameterConfig.factory('float', bounds=(0.0, 1.0))
    value = random_sample._sample_value(self.rng, float_param)
    self.assertTrue(0.0 <= value <= 1.0)

  def test_sample_value_categorical(self):
    categorical_param = vz.ParameterConfig.factory(
        'categorical', feasible_values=['a', 'b', 'c'])
    value = random_sample._sample_value(self.rng, categorical_param)
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
    parameter_dict = random_sample.sample_parameters(self.rng, space)
    self.assertIn(parameter_dict['b1'].value, ['True', 'False'])
    self.assertIn(parameter_dict['d1'].value, [1.0, 2.0, 10.0])
    self.assertTrue(0.0 <= parameter_dict['f1'].value <= 15.0)
    self.assertTrue(100.0 <= parameter_dict['f2'].value <= 200.0)
    self.assertIn(parameter_dict['i1'].value, list(range(0, 11)))
    self.assertIn(parameter_dict['c1'].value, ['a', 'b', 'c'])


if __name__ == '__main__':
  absltest.main()
