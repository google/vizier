# Copyright 2023 Google LLC.
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

"""Tests for MaskFeatures."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax.models import mask_features

from absl.testing import absltest


class MaskFeaturesTest(parameterized.TestCase):

  @parameterized.parameters(2, 4, 6, 8)
  def testMaskingAndGradient(self, dims):
    half_dims = dims // 2
    xs = np.random.randn(10, half_dims)
    xs = np.concatenate([xs, np.full([10, half_dims], np.nan)], axis=-1)
    kernel = tfp.math.psd_kernels.MaternThreeHalves()
    kernel = mask_features.MaskFeatures(
        kernel,
        dimension_is_missing=np.array([False] * half_dims + [True] * half_dims),
    )
    f = lambda x: jnp.sum(kernel.matrix(x, x))
    value, grad = jax.value_and_grad(f)(xs)
    self.assertTrue(np.all(~np.isnan(value)))
    self.assertTrue(np.all(~np.isnan(grad)))


if __name__ == '__main__':
  absltest.main()
