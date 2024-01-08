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

"""Utils for attrs."""

import re
from typing import Any, Callable, Collection, Optional

import attr

_Validator = Callable[[Any, attr.Attribute, Any], None]


def assert_not_empty(instance: Any, attribute: attr.Attribute,
                     value: Any) -> None:
  if not value:
    raise ValueError(f'{attribute.name} must not be empty in {type(instance)}.')


def assert_not_negative(instance: Any, attribute: attr.Attribute,
                        value: int) -> None:
  if value < 0:
    raise ValueError(
        f'{attribute.name} must not be negative in {type(instance)}.')


def assert_not_none(instance: Any, attribute: attr.Attribute,
                    value: Any) -> None:
  if value is None:
    raise ValueError(f'{attribute.name} must not be None in {type(instance)}.')


def assert_between(
    low: float, high: float
) -> Callable[[Any, attr.Attribute, str], None]:
  def validator(instance, attribute, value):
    del instance
    if value < low or value > high:
      raise ValueError(
          f'{attribute.name} ({value}) must be between {low} and {high}'
      )

  return validator


def assert_re_fullmatch(
    regex: str) -> Callable[[Any, attr.Attribute, str], None]:

  def validator(instance: Any, attribute: attr.Attribute, value: str) -> None:
    if not re.fullmatch(regex, value):
      raise ValueError(
          f'{attribute.name} must match the regex {regex} in {type(instance)}.')

  return validator


def shape_equals(instance_to_shape: Callable[[Any], Collection[Optional[int]]]):
  """Creates a shape validator for attrs.

  For example, _shape_equals(lambda s : [3, None]) validates that the shape has
  length 2 and its first element is 3.

  Code Example:
  @attrs.define
  class TestAttr:
    x = attrs.field(validator=attrs_utils.shape_equals(lambda v: (3, v.d)))
    d = attrs.field()

  _TestAttr(np.zeros([3, 2]), 2)  # OK
  _TestAttr(np.zeros([3, 5]), None) # OK
  _TestAttr(np.zeros([3, 2]), 4)  # Raises ValueError

  Args:
    instance_to_shape: Takes instance as input and returns the desired shape for
      the instance. `None` is treated as "any number".

  Returns:
    A validator that can be passed into attrs.ib or attrs.field.
  """

  def validator(instance, attribute, value) -> None:
    shape = instance_to_shape(instance)

    def _validator_boolean():
      if len(value.shape) != len(shape):
        return False
      for s1, s2 in zip(value.shape, shape):
        if (s2 is not None) and (s1 != s2):
          return False
      return True

    if not _validator_boolean():
      raise ValueError(f'{attribute.name} has shape {value.shape} '
                       f'which does not match the expected shape {shape}')

  return validator
