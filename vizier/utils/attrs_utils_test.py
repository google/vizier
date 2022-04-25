"""Tests for attrs_utils."""

import attrs
import numpy as np

from vizier.utils import attrs_utils
from absl.testing import absltest


@attrs.define
class _TestAttr:
  x = attrs.field(validator=attrs_utils.shape_equals(lambda v: (3, v.d)))
  d = attrs.field()


class ShapeEqualsTest(absltest.TestCase):

  def test_good_shape(self):
    _TestAttr(np.zeros([3, 2]), 2)

  def test_good_shape_none(self):
    _TestAttr(np.zeros([3, 5]), None)
    _TestAttr(np.zeros([3, 0]), None)

  def test_bad_shape(self):
    with self.assertRaises(ValueError):
      _TestAttr(np.zeros([3, 2]), 4)


if __name__ == '__main__':
  absltest.main()
