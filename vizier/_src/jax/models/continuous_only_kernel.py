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

"""PSD Kernel for convenience with continuous-only data."""

import jax
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

  def _apply(self, x1, x2, example_ndims=0):
    return self.kernel._apply(x1.continuous, x2.continuous, example_ndims)  # pylint: disable=protected-access

  def _matrix(self, x1, x2):
    return self.kernel._matrix(x1.continuous, x2.continuous)  # pylint: disable=protected-access

  def matrix_over_all_tasks(self, x1, x2):
    return self.kernel.matrix_over_all_tasks(x1.continuous, x2.continuous)  # pylint: disable=protected-access

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        kernel=tfp.internal.parameter_properties.BatchedComponentProperties(),
    )


def _continuous_only_flatten(v):
  children = (v.kernel,)
  return (children, None)


def _continuous_only_unflatten(_, children):
  return ContinuousOnly(*children)


jax.tree_util.register_pytree_node(
    ContinuousOnly, _continuous_only_flatten, _continuous_only_unflatten
)

tfpke.MultiTaskKernel.register(ContinuousOnly)
