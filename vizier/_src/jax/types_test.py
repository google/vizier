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

from absl.testing import parameterized
import equinox as eqx
import jax
from jax import numpy as jnp
import numpy as np
from vizier._src.jax import types as vt
from absl.testing import absltest


# pylint: disable=g-long-lambda
class PaddedArrayTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          factory=lambda arr: vt.PaddedArray.from_array(
              arr, arr.shape, fill_value=-1
          )
      ),
      dict(factory=vt.PaddedArray.as_padded),
  )
  def test_padding_to_the_same_shape(self, factory) -> None:
    """Tests the case where no padding dimensions are added.

    Args:
      factory: Takes an array and returns a padded array.
    """
    arr = jnp.ones([3, 2], dtype=jnp.int32)
    parr = factory(arr)
    # The padded shape matches the original shape, which means we can unpad
    # within a jit scope.
    self.assertSequenceEqual(
        eqx.filter_jit(parr.unpad)().shape,
        arr.shape,
    )

    np.testing.assert_array_equal(parr.padded_array, arr)
    np.testing.assert_array_equal(parr.is_missing[0], [False, False, False])
    np.testing.assert_array_equal(parr.is_missing[1], [False, False])
    np.testing.assert_array_equal(
        parr.replace_fill_value(-999).padded_array == -999, ~parr._mask
    )

  @parameterized.parameters(dict(fill_value=0), dict(fill_value=-1))
  def test_padding_to_a_different_shape(self, fill_value):
    arr = jnp.ones([3, 2], dtype=jnp.int32)
    # If the padded shape does not match the original shape, then we can't unpad
    # within a jit scope.
    target_shape = (5, 3)
    parr = vt.PaddedArray.from_array(arr, target_shape, fill_value=fill_value)
    with self.assertRaises(jax.errors.ConcretizationTypeError):
      eqx.filter_jit(parr.unpad)()

    expected = np.asarray(
        [
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=arr.dtype,
    )
    expected[expected == 0] = fill_value
    np.testing.assert_array_equal(parr.padded_array, expected)
    np.testing.assert_array_equal(parr.unpad(), arr)
    np.testing.assert_array_equal(
        parr.is_missing[0], [False, False, False, True, True]
    )
    np.testing.assert_array_equal(parr.is_missing[1], [False, False, True])
    np.testing.assert_array_equal(
        parr.replace_fill_value(-999).padded_array == -999, ~parr._mask
    )


if __name__ == "__main__":
  absltest.main()
