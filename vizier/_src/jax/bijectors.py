"""Bijectors for output warping."""

from typing import Any

from jax import numpy as jnp
import numpy as np
from sklearn import preprocessing
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
tfsb = tfp.staging.bijectors

Array = Any


def optimal_power_transformation(
    data: np.ndarray,
    method: str = 'yeo-johnson',
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
      then normalize so that the data maps to zero mean unit stddev normal.

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
    data = data[:, jnp.newaxis]
    reduce_axis = None
  else:
    reduce_axis = 0

  if method == 'yeo-johnson':
    # For yeo-johnson, center the median to zero.
    # In the long run, we should consider identifying outliers that are very
    # far away from the optimum, and softclipping them to reasonable numbers.
    # This will help prevent them from having too much influence in deciding
    # the warping parameters.
    medians = tfp.stats.percentile(data, 50, axis=reduce_axis)
    shift1 = tfb.Shift(-medians)
    data = shift1(data)
  else:
    shift1 = tfb.Identity()
  lambdas = preprocessing.PowerTransformer(
      method, standardize=False).fit(data).lambdas_.astype(dtype)

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
    shift2 = tfb.Shift(-jnp.mean(transformed, axis=reduce_axis))
    scale = tfb.Scale(
        np.array(1.0).astype(dtype) /
        (jnp.std(transformed, axis=reduce_axis) +
         jnp.finfo(dtype).eps))
    return tfb.Chain([scale, shift2, warp, shift1])
  else:
    return tfb.Chain([warp, shift1])


def flip_sign(should_flip: np.ndarray,
              dtype: Any = 'float32') -> tfb.AutoCompositeTensorBijector:
  """Optionally flips the sign.

  Args:
    should_flip: Boolean array.
    dtype: Numpy dtype.

  Returns:
    Bijector that flips the sign.
  """
  if np.all(np.logical_not(should_flip)):
    # No sign flips.
    return tfb.Identity()

  return tfb.Scale((should_flip * -2 + 1).astype(dtype))
