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

"""Tests for MaskFeatures."""

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax.models import mask_features

from absl.testing import absltest

tfpke = tfp.experimental.psd_kernels


class MaskFeaturesTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, 4)
  def testMaskingAndGradient(self, half_dims: int):
    xs_cont = np.random.randn(10, half_dims)
    xs_cont = np.concatenate(
        [xs_cont, np.full([10, half_dims], np.nan)], axis=-1
    )
    # Generate some integer-encoded categorical data with values between 0 and
    # 5.
    xs_cat = np.broadcast_to(
        np.arange(10)[:, jnp.newaxis] // 2,
        shape=(10, half_dims),
    )
    xs_cat = np.concatenate([xs_cat, np.full([10, half_dims], -1)], axis=-1)
    missing_dim = np.array([False] * half_dims + [True] * half_dims)
    x = tfpke.ContinuousAndCategoricalValues(xs_cont, xs_cat)

    def f(s):
      kernel = tfp.math.psd_kernels.MaternThreeHalves()
      kernel = tfpke.FeatureScaledWithCategorical(
          kernel,
          scale_diag=s,
      )

      kernel = mask_features.MaskFeatures(
          kernel,
          dimension_is_missing=tfpke.ContinuousAndCategoricalValues(
              continuous=missing_dim, categorical=missing_dim
          ),
      )
      return jnp.sum(kernel.matrix(x, x))

    scale_diag = tfpke.ContinuousAndCategoricalValues(
        np.random.uniform(low=1.0, high=10.0, size=(half_dims * 2,)),
        np.random.uniform(low=1.0, high=10.0, size=(half_dims * 2,)),
    )
    value, grad = jax.value_and_grad(f)(scale_diag)
    self.assertTrue(np.all(~np.isnan(value)))
    self.assertTrue(np.all(~np.isnan(grad.continuous)))
    self.assertTrue(np.all(grad.continuous[~missing_dim] != 0))
    self.assertTrue(np.all(grad.continuous[missing_dim] == 0))
    self.assertTrue(np.all(~np.isnan(grad.categorical)))
    self.assertTrue(np.all(grad.categorical[~missing_dim] != 0))
    self.assertTrue(np.all(grad.categorical[missing_dim] == 0))


if __name__ == '__main__':
  absltest.main()
