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

"""Temporary file for finding the optimal Yeo-Johnson transformation."""

from typing import Literal

from absl import logging
import numpy as np
from sklearn import preprocessing
from tensorflow_probability.substrates import jax as tfp

tfsb = tfp.staging.bijectors
tfb = tfp.bijectors
tfd = tfp.distributions


# TODO: Use Jax or numpy to find the optimum, and remove
# dependency on sklearn.
def optimal_transformation(
    data: np.ndarray,
    method: Literal['yeo-johnson', 'box-cox'] = 'yeo-johnson',
    *,
    standardize: bool = True) -> tfb.AutoCompositeTensorBijector:
  """Returns the power transformation with optimal parameterization.

  The optimal parameterization makes the transformed data as "normal"-esque
  as possible.

  Args:
    data: 1-D or 2-D array. If 1-D, then the bijector has batch_shape =[]. If
      2-D, then the bijector has batch shape equal to the last dimension
    method: 'yeo-johnson' or 'box-cox'. Boxcox can only be used for
      positive-only data.
    standardize: If True, returns a bijector that applies a power transform and
      then normalize so that the data maps to zero mean unit stddev normal. 1e-6
      is added to the stddev so that division by zero never happens.

  Returns:
    Bijector that maps data such that it follows a normal distribution.
    (standard normal if standardize=True).
  """
  dtype = data.dtype
  dimension = len(data.shape)
  if dimension not in {1, 2}:
    raise ValueError('Data must be 1-D or 2-D array')

  if dimension == 1:
    # PowerTransformer.fit() expects 2D array.
    data = data[:, np.newaxis]
    reduce_axis = None
  else:
    reduce_axis = 0

  if method == 'yeo-johnson':
    # For yeo-johnson, center the median to zero.
    # In the long run, we should consider identifying outliers that are very
    # far away from the optimum, and softclipping them to reasonable numbers.
    # This will help prevent them from having too much influence in deciding
    # the warping parameters.
    medians = np.median(data, axis=reduce_axis)
    shift1 = tfb.Shift(-medians)
    data = shift1(data)
  else:
    shift1 = tfb.Identity()
  lambdas = preprocessing.PowerTransformer(
      method, standardize=False).fit(data).lambdas_.astype(dtype)

  logging.info('Optimal lambda was: %s', lambdas)

  if dimension == 1:
    # Make it a scalar, so we don't end up with batch_shape = [1] in the
    # bijector.
    lambdas = lambdas.item()
  if method == 'yeo-johnson':
    warp = tfsb.YeoJohnson(lambdas)
  elif method == 'box-cox':
    warp = tfsb.YeoJohnson(lambdas, shift=.0)
  else:
    raise ValueError(f'Unknown method: {method}')

  if standardize:
    transformed = warp(data)  # 2-D array.
    shift2 = tfb.Shift(-np.mean(transformed, axis=reduce_axis))
    scale = tfb.Scale(1.0 / (np.std(transformed, axis=reduce_axis) + 1e-6))
    return tfb.Chain([scale, shift2, warp, shift1])
  else:
    return tfb.Chain([warp, shift1])
