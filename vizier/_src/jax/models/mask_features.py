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

"""PSD Kernel for masking out dimensions."""

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfpk = tfp.math.psd_kernels
tfpke = tfp.experimental.psd_kernels


class MaskFeatures(tfpk.PositiveSemidefiniteKernel):
  """Kernel for masking out features by setting them to zero.

  By masking out feature dimensions to zero (or to any other finite value), we
  ensure that when the Stochastic Process Model aware of this masking ingests
  these features (which may be NaN), we will get finite gradients. See
  https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
  for more details.
  """

  def __init__(
      self,
      kernel: tfpk.PositiveSemidefiniteKernel,
      dimension_is_missing: tfpke.ContinuousAndCategoricalValues,
  ):
    parameters = dict(locals())
    self._kernel = kernel
    self._dimension_is_missing = dimension_is_missing

    def _transformation_fn(x, *_):
      return jax.tree_util.tree_map(
          lambda x_, d: jnp.where(d, jnp.zeros_like(x_), x_),
          x,
          dimension_is_missing,
      )

    self._mask_kernel = tfpk.FeatureTransformed(kernel, _transformation_fn)

    super(MaskFeatures, self).__init__(
        feature_ndims=kernel.feature_ndims,
        dtype=kernel.dtype,
        name='MaskFeatures',
        validate_args=False,
        parameters=parameters,
    )

  @property
  def kernel(self):
    return self._kernel

  @property
  def dimension_is_missing(self):
    return self._dimension_is_missing

  def _apply(self, x1, x2, example_ndims=0):
    return self._mask_kernel._apply(x1, x2, example_ndims)  # pylint: disable=protected-access

  def _matrix(self, x1, x2):
    return self._mask_kernel._matrix(x1, x2)  # pylint: disable=protected-access

  @classmethod
  def _parameter_properties(cls, dtype):
    return dict(
        kernel=tfp.internal.parameter_properties.BatchedComponentProperties(),
        dimension_is_missing=(
            tfp.internal.parameter_properties.ParameterProperties(
                event_ndims=lambda self: self.kernel.feature_ndims
            )
        ),
    )


def _mask_features_flatten(v):
  children = (v.kernel, v.dimension_is_missing)
  return (children, None)


def _mask_features_unflatten(_, children):
  return MaskFeatures(*children)


jax.tree_util.register_pytree_node(
    MaskFeatures, _mask_features_flatten, _mask_features_unflatten
)
