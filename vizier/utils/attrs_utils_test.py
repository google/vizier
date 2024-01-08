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

"""Tests for attrs_utils."""

from typing import Any

import attrs
import numpy as np

from vizier.utils import attrs_utils
from absl.testing import absltest
from absl.testing import parameterized


class ValidatorsTest(parameterized.TestCase):

  @parameterized.parameters(
      (attrs_utils.assert_not_empty, [], False),
      (attrs_utils.assert_not_empty, [1], True),
      (attrs_utils.assert_not_negative, 0, -1),
      (attrs_utils.assert_not_negative, 0, True),
      (attrs_utils.assert_not_negative, 1, True),
      (attrs_utils.assert_not_none, None, False),
      (attrs_utils.assert_not_none, 0, True),
  )
  def test_validator(self, validator, value: Any, result: bool):
    self.assertValidatorWorksAsIntended(validator, value, result)

  def assertValidatorWorksAsIntended(self, validator, value: Any, result: bool):

    @attrs.define
    class Test:
      x = attrs.field(validator=validator)

    if result:
      Test(value)
    else:
      with self.assertRaises(ValueError):
        Test(value)

  @parameterized.parameters(
      (0.0, 1.0, 0.5, True),
      (0.0, 1.0, 1.0, True),
      (0.0, 1.0, 0.0, True),
      (0.0, 1.0, 5.0, False),
  )
  def test_between_validator(
      self, low: float, high: float, value: float, result: bool
  ):
    self.assertValidatorWorksAsIntended(
        attrs_utils.assert_between(low, high), value, result
    )

  @parameterized.parameters(
      (r'[^\/]+', 'good', True),
      (r'[^\/]+', 'bad/', False),
      (r'[^\/]+', '', False),
  )
  def test_regex_validator(self, regex: str, value: str, result: bool):
    self.assertValidatorWorksAsIntended(
        attrs_utils.assert_re_fullmatch(regex), value, result)

  def test_good_shape_none(self):
    _ShapeEqualsTestAttr(np.zeros([3, 5]), None)
    _ShapeEqualsTestAttr(np.zeros([3, 0]), None)

  def test_bad_shape(self):
    with self.assertRaises(ValueError):
      _ShapeEqualsTestAttr(np.zeros([3, 2]), 4)


@attrs.define
class _ShapeEqualsTestAttr:
  x = attrs.field(validator=attrs_utils.shape_equals(lambda v: (3, v.d)))
  d = attrs.field()


class ShapeEqualsTest(absltest.TestCase):

  def test_good_shape(self):
    _ShapeEqualsTestAttr(np.zeros([3, 2]), 2)

  def test_good_shape_none(self):
    _ShapeEqualsTestAttr(np.zeros([3, 5]), None)
    _ShapeEqualsTestAttr(np.zeros([3, 0]), None)

  def test_bad_shape(self):
    with self.assertRaises(ValueError):
      _ShapeEqualsTestAttr(np.zeros([3, 2]), 4)


if __name__ == '__main__':
  absltest.main()
