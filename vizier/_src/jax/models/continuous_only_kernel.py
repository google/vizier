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

"""PSD Kernel for convenience with continuous-only data."""

import jax
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import types

tfpk = tfp.math.psd_kernels
tfpke = tfp.experimental.psd_kernels


class ContinuousOnly(tfpk.PositiveSemidefiniteKernel):
  """Kernel for using only continuous data and discarding categorical.

  """

  def __init__(
      self,
      kernel: tfpk.PositiveSemidefiniteKernel,
  ):
    parameters = dict(locals())
    self._kernel = kernel

    super(ContinuousOnly, self).__init__(
        feature_ndims=tfpke.ContinuousAndCategoricalValues(
            kernel.feature_ndims, 1),
        dtype=tfpke.ContinuousAndCategoricalValues(
            kernel.dtype, types.INT_DTYPE),
        name='MaskFeatures',
        validate_args=False,
        parameters=parameters,
    )

  @property
  def kernel(self):
    return self._kernel

  def __getattr__(self, name):
    return getattr(self.kernel, name)

  def _apply(
      self, x1: types.ModelInput, x2: types.ModelInput, example_ndims=0
  ) -> jax.Array:
    return self.kernel._apply(x1.continuous, x2.continuous, example_ndims)  # pylint: disable=protected-access

  def _matrix(self, x1: types.ModelInput, x2: types.ModelInput) -> jax.Array:
    return self.kernel._matrix(x1.continuous, x2.continuous)  # pylint: disable=protected-access

  def matrix_over_all_tasks(
      self, x1: types.ModelInput, x2: types.ModelInput
  ) -> jax.Array:
    return self.kernel.matrix_over_all_tasks(x1.continuous, x2.continuous)  # pylint: disable=protected-access

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        kernel=tfp.internal.parameter_properties.BatchedComponentProperties(),
    )


def _continuous_to_cacv(x: jax.Array) -> tfpke.ContinuousAndCategoricalValues:
  return tfpke.ContinuousAndCategoricalValues(
      continuous=x, categorical=jnp.zeros(x.shape[:-1] + (0,))
  )


class EmptyCategoricalKernel(tfpk.PositiveSemidefiniteKernel):
  """Kernel that packs continuous data to ModelInput with empty categorical.

  If `k` is a PSDKernel that operates on `ModelInput` structures, then
  `EmptyCategoricalKernel(k)` is a kernel that operates on arrays of continuous
  data.

  Example:

  ```python
  tfpke = tfp.experimental.psd_kernels
  tfpk = tfp.math.psd_kernels

  # Define a kernel that operates on continuous and categorical values, with the
  # `categorical` field empty.
  kernel = tfpke.FeatureScaledWithCategorical(
      tfpk.ExponentiatedQuadratic(),
      scale_diag=tfpke.ContinuousAndCategoricalValues(
          continuous=jnp.ones([3]),
          categorical=jnp.ones([0])
      )
  )

  # Define an `EmptyCategoricalKernel` that operates on arrays of continuous
  # data.
  empty_cat_kernel = EmptyCategoricalKernel(k)
  xs = np.random.normal([10, 3])
  mat = empty_cat_kernel.matrix(xs, xs)  # Returns a [10, 10] matrix.
  ```
  """

  def __init__(
      self,
      kernel: tfpk.PositiveSemidefiniteKernel,
  ):
    parameters = dict(locals())
    self._kernel = kernel

    super(EmptyCategoricalKernel, self).__init__(
        feature_ndims=kernel.feature_ndims.continuous,
        dtype=kernel.dtype.continuous,
        name='EmptyCategoricalKernel',
        validate_args=False,
        parameters=parameters,
    )

  @property
  def kernel(self):
    return self._kernel

  def __getattr__(self, name):
    return getattr(self.kernel, name)

  def _apply(self, x1: jax.Array, x2: jax.Array, example_ndims: int = 0):
    return self.kernel._apply(  # pylint: disable=protected-access
        _continuous_to_cacv(x1), _continuous_to_cacv(x2), example_ndims
    )

  def _matrix(self, x1: jax.Array, x2: jax.Array):
    return self.kernel._matrix(  # pylint: disable=protected-access
        _continuous_to_cacv(x1), _continuous_to_cacv(x2)
    )

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        kernel=tfp.internal.parameter_properties.BatchedComponentProperties(),
    )


def _flatten(v):
  children = (v.kernel,)
  return (children, None)


def _unflatten(_, children):
  return ContinuousOnly(*children)


jax.tree_util.register_pytree_node(ContinuousOnly, _flatten, _unflatten)
jax.tree_util.register_pytree_node(EmptyCategoricalKernel, _flatten, _unflatten)

tfpke.MultiTaskKernel.register(ContinuousOnly)
