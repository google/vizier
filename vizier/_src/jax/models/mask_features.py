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

"""PSD Kernel for masking out dimensions."""

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import types


class MaskFeatures(tfp.math.psd_kernels.FeatureTransformed):
  """Kernel for masking out features by setting them to zero.

  By masking out feature dimensions to zero (or to any other finite value), we
  ensure that when the Stochastic Process Model aware of this masking ingests
  these features (which may be NaN), we will get finite gradients. See
  https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
  for more details.
  """

  def __init__(
      self,
      kernel: tfp.math.psd_kernels.PositiveSemidefiniteKernel,
      dimension_is_missing: types.Array,
  ):
    self._kernel = kernel
    self._dimension_is_missing = dimension_is_missing

    super(MaskFeatures, self).__init__(
        kernel,
        transformation_fn=lambda x, *_: jnp.where(  # pylint:disable=g-long-lambda
            dimension_is_missing, jnp.zeros_like(x), x
        ),
    )

  @property
  def kernel(self):
    return self._kernel

  @property
  def dimension_is_missing(self):
    return self._dimension_is_missing

  def _batch_shape(self):
    return self.kernel.batch_shape

  def _batch_shape_tensor(self):
    return self.kernel.batch_shape_tensor()
