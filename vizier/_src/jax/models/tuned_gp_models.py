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

"""Collection of well-tuned GP models."""

# TODO: Add Ax/BoTorch GP.

import functools
from typing import Any, Generator, Optional, Union

from flax import struct
import jax
from jax import numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier._src.jax.models import continuous_only_kernel
from vizier._src.jax.models import mask_features

tfb = tfp.bijectors
tfd = tfp.distributions
tfpk = tfp.math.psd_kernels
tfpke = tfp.experimental.psd_kernels


def _log_uniform_init(
    low: Union[float, np.floating],
    high: Union[float, np.floating],
    shape: tuple[int, ...] = tuple(),
) -> sp.InitFn:
  r"""Take log-uniform sample in the constraint and map it back to \R.

  Args:
    low: Parameter lower bound.
    high: Parameter upper bound.
    shape: Returned array has this shape. Each entry in the returned array is an
      i.i.d sample.

  Returns:
    Randomly sampled array.
  """

  def sample(key: Any) -> jnp.ndarray:
    unif = jax.random.uniform(key, shape, dtype=jnp.float64)
    return jnp.exp(unif * jnp.log(high / low) + jnp.log(low))

  return sample


@struct.dataclass
class VizierGaussianProcess(sp.ModelCoroutine[tfd.GaussianProcess]):
  """Vizier's tuned GP with categorical parameters.

  See __call__ method documentation.

  Attributes:
    _boundary_epsilon: We expand the constraints by this number so that the
      values exactly at the boundary can be mapped to unconstrained space. i.e.
      we are trying to avoid SoftClip(low=1e-2, high=1.).inverse(1e-2) giving
      NaN.
  """

  _dim: types.ContinuousAndCategorical[int] = struct.field(pytree_node=False)
  _use_retrying_cholesky: bool = struct.field(
      pytree_node=False, default=True, kw_only=True
  )
  _boundary_epsilon: float = struct.field(default=1e-12, kw_only=True)
  _linear_coef: float = struct.field(default=1e-12, kw_only=True)

  @classmethod
  def build_model(
      cls,
      features: types.ModelInput,
      *,
      use_retrying_cholesky: bool = True,
      linear_coef: float = 1e-12,
  ) -> sp.StochasticProcessModel:
    """Returns the model and loss function."""
    gp_coroutine = VizierGaussianProcess(
        _dim=types.ContinuousAndCategorical[int](
            features.continuous.padded_array.shape[-1],
            features.categorical.padded_array.shape[-1],
        ),
        _use_retrying_cholesky=use_retrying_cholesky,
        _linear_coef=linear_coef,
    )
    return sp.StochasticProcessModel(gp_coroutine)

  def __call__(
      self,
      inputs: Optional[types.ModelInput] = None,
  ) -> Generator[sp.ModelParameter, jax.Array, tfd.GaussianProcess]:
    """Creates a generator.

    Args:
      inputs: ContinuousAndCategoricalArray with array shapes of: (num_examples,
        continuous_feature_dim), (num_examples, categorical_feature_dim).

    Yields:
      GaussianProcess whose event shape is `num_examples`.
    """
    eps = self._boundary_epsilon
    observation_noise_bounds = (np.float64(1e-10 - eps), 1.0 + eps)
    amplitude_bounds = (np.float64(1e-3 - eps), 10.0 + eps)
    continuous_ones = np.ones((self._dim.continuous), dtype=np.float64)
    continuous_length_scale_bounds = (
        continuous_ones * (1e-2 - eps),
        continuous_ones * 1e2 + eps,
    )
    categorical_ones = np.ones((self._dim.categorical), dtype=np.float64)
    categorical_length_scale_bounds = (
        categorical_ones * (1e-2 - eps),
        categorical_ones * 1e2 + eps,
    )

    signal_variance = yield sp.ModelParameter(
        init_fn=_log_uniform_init(*amplitude_bounds),
        constraint=sp.Constraint(
            amplitude_bounds,
            tfb.SoftClip(*amplitude_bounds, hinge_softness=1e-2),
        ),
        regularizer=lambda x: 0.01 * jnp.log(x / 0.039) ** 2,
        name='signal_variance',
    )
    kernel = tfpk.MaternFiveHalves(amplitude=jnp.sqrt(signal_variance))

    continuous_length_scale_squared = yield sp.ModelParameter(
        init_fn=_log_uniform_init(
            *continuous_length_scale_bounds, shape=(self._dim.continuous,)
        ),
        constraint=sp.Constraint(
            continuous_length_scale_bounds,
            tfb.SoftClip(*continuous_length_scale_bounds, hinge_softness=1e-2),
        ),
        regularizer=lambda x: jnp.sum(0.01 * jnp.log(x / 0.5) ** 2),
        name='continuous_length_scale_squared',
    )
    categorical_length_scale_squared = yield sp.ModelParameter(
        init_fn=_log_uniform_init(
            *categorical_length_scale_bounds,
            shape=(self._dim.categorical,),
        ),
        constraint=sp.Constraint(
            categorical_length_scale_bounds,
            tfb.SoftClip(*categorical_length_scale_bounds, hinge_softness=1e-2),
        ),
        regularizer=lambda x: jnp.sum(0.01 * jnp.log(x / 0.5) ** 2),
        name='categorical_length_scale_squared',
    )
    kernel = tfpke.FeatureScaledWithCategorical(
        kernel,
        scale_diag=tfpke.ContinuousAndCategoricalValues(
            jnp.sqrt(continuous_length_scale_squared),
            jnp.sqrt(categorical_length_scale_squared),
        ),
    )
    if inputs is not None:
      # Ensure features are zero for this kernel. This will also ensure the
      # length scales are not trainable, since there will be no signal from
      # these dimensions.
      kernel = mask_features.MaskFeatures(
          kernel,
          dimension_is_missing=tfpke.ContinuousAndCategoricalValues(
              continuous=inputs.continuous.is_missing[1],
              categorical=inputs.categorical.is_missing[1],
          ),
      )
      inputs = tfpke.ContinuousAndCategoricalValues(
          continuous=inputs.continuous.padded_array,
          categorical=inputs.categorical.padded_array,
      )

    observation_noise_variance = yield sp.ModelParameter(
        init_fn=_log_uniform_init(*observation_noise_bounds),
        constraint=sp.Constraint(
            observation_noise_bounds,
            tfb.SoftClip(*observation_noise_bounds, hinge_softness=1e-2),
        ),
        regularizer=lambda x: 0.01 * jnp.log(x / 0.0039) ** 2,
        name='observation_noise_variance',
    )
    cholesky_fn = None
    # When cholesky fails, increase jitters and retry.
    if self._use_retrying_cholesky:
      retrying_cholesky = functools.partial(
          tfp.experimental.distributions.marginal_fns.retrying_cholesky,
          jitter=np.float64(1e-4),
          max_iters=5,
      )
      cholesky_fn = lambda matrix: retrying_cholesky(matrix)[0]

    return tfd.GaussianProcess(
        kernel,
        index_points=inputs,
        observation_noise_variance=observation_noise_variance,
        cholesky_fn=cholesky_fn,
    )


@struct.dataclass
class ConstantMeanFn:
  """Implements a constant GP mean function."""

  constant: jax.Array

  def __call__(self, x: types.ModelInput) -> jax.Array:
    del x
    return self.constant


# TODO: Consider renaming this to reflect that it uses a
# Matern + Linear kernel, not just a Linear kernel (or merging this class with
# VizierGaussianProcess).
@struct.dataclass
class VizierLinearGaussianProcess(sp.ModelCoroutine[tfd.GaussianProcess]):
  """Vizier's tuned GP with linear kernel.

  See __call__ method documentation.

  Attributes:
    _boundary_epsilon: We expand the constraints by this number so that the
      values exactly at the boundary can be mapped to unconstrained space. i.e.
      we are trying to avoid SoftClip(low=1e-2, high=1.).inverse(1e-2) giving
      NaN.
  """

  _dim: types.ContinuousAndCategorical[int] = struct.field(pytree_node=False)
  _use_retrying_cholesky: bool = struct.field(
      pytree_node=False, default=True, kw_only=True
  )
  _boundary_epsilon: float = struct.field(default=1e-12, kw_only=True)
  _linear_coef: float = struct.field(default=1e-12, kw_only=True)

  # TODO: Redesign or remove this method given the updated
  # StochasticProcessModelCoroutine class.
  @classmethod
  def build_model(
      cls,
      features: types.ModelInput,
      *,
      use_retrying_cholesky: bool = True,
      linear_coef: float = 1e-12,
  ) -> sp.StochasticProcessModel:
    """Returns the model and loss function."""
    gp_coroutine = VizierLinearGaussianProcess(
        _dim=types.ContinuousAndCategorical[int](
            features.continuous.padded_array.shape[-1],
            features.categorical.padded_array.shape[-1],
        ),
        _use_retrying_cholesky=use_retrying_cholesky,
        _linear_coef=linear_coef,
    )
    return sp.StochasticProcessModel(gp_coroutine)

  def __call__(
      self,
      inputs: Optional[types.ModelInput] = None,
  ) -> Generator[sp.ModelParameter, jax.Array, tfd.GaussianProcess]:
    """Creates a generator.

    Args:
      inputs: ContinuousAndCategoricalArray with array shapes of: (num_examples,
        continuous_feature_dim), (num_examples, categorical_feature_dim).

    Yields:
      GaussianProcess whose event shape is `num_examples`.
    """
    eps = self._boundary_epsilon
    observation_noise_bounds = (np.float64(1e-10 - eps), 1.0 + eps)
    amplitude_bounds = (np.float64(1e-3 - eps), 10.0 + eps)
    continuous_ones = np.ones((self._dim.continuous), dtype=np.float64)
    continuous_length_scale_bounds = (
        continuous_ones * (1e-2 - eps),
        continuous_ones * 1e2 + eps,
    )
    categorical_ones = np.ones((self._dim.categorical), dtype=np.float64)
    categorical_length_scale_bounds = (
        categorical_ones * (1e-2 - eps),
        categorical_ones * 1e2 + eps,
    )

    signal_variance = yield sp.ModelParameter(
        init_fn=_log_uniform_init(*amplitude_bounds),
        constraint=sp.Constraint(
            amplitude_bounds,
            tfb.SoftClip(*amplitude_bounds, hinge_softness=1e-2),
        ),
        regularizer=lambda x: 0.01 * jnp.log(x / 0.039) ** 2,
        name='signal_variance',
    )
    kernel = tfpk.MaternFiveHalves(amplitude=jnp.sqrt(signal_variance))

    continuous_length_scale_squared = yield sp.ModelParameter(
        init_fn=_log_uniform_init(
            *continuous_length_scale_bounds, shape=(self._dim.continuous,)
        ),
        constraint=sp.Constraint(
            continuous_length_scale_bounds,
            tfb.SoftClip(*continuous_length_scale_bounds, hinge_softness=1e-2),
        ),
        regularizer=lambda x: jnp.sum(0.01 * jnp.log(x / 0.5) ** 2),
        name='continuous_length_scale_squared',
    )
    categorical_length_scale_squared = yield sp.ModelParameter(
        init_fn=_log_uniform_init(
            *categorical_length_scale_bounds,
            shape=(self._dim.categorical,),
        ),
        constraint=sp.Constraint(
            categorical_length_scale_bounds,
            tfb.SoftClip(*categorical_length_scale_bounds, hinge_softness=1e-2),
        ),
        regularizer=lambda x: jnp.sum(0.01 * jnp.log(x / 0.5) ** 2),
        name='categorical_length_scale_squared',
    )
    kernel = tfpke.FeatureScaledWithCategorical(
        kernel,
        scale_diag=tfpke.ContinuousAndCategoricalValues(
            jnp.sqrt(continuous_length_scale_squared),
            jnp.sqrt(categorical_length_scale_squared),
        ),
    )

    # We do not need to tune bias here because we are tuning mean_fn.
    slopes = yield sp.ModelParameter(
        init_fn=_log_uniform_init(*amplitude_bounds),
        constraint=sp.Constraint(
            amplitude_bounds,
            tfb.SoftClip(*amplitude_bounds, hinge_softness=1e-2),
        ),
        regularizer=lambda x: 0.01 * jnp.log(x / 0.039) ** 2,
        name='slope_amplitude',
    )
    shift = yield sp.ModelParameter(
        init_fn=jax.random.normal,
        regularizer=lambda x: 0.5 * x**2,
        name='shift',
    )
    kernel = kernel + continuous_only_kernel.ContinuousOnly(
        tfpk.FeatureScaled(
            tfpk.Linear(
                slope_amplitude=self._linear_coef * slopes,
                shift=self._linear_coef * shift,
            ),
            scale_diag=jnp.sqrt(continuous_length_scale_squared),
        )
    )

    if inputs is not None:
      # Ensure features are zero for this kernel. This will also ensure the
      # length scales are not trainable, since there will be no signal from
      # these dimensions.
      kernel = mask_features.MaskFeatures(
          kernel,
          dimension_is_missing=tfpke.ContinuousAndCategoricalValues(
              continuous=inputs.continuous.is_missing[1],
              categorical=inputs.categorical.is_missing[1],
          ),
      )
      inputs = tfpke.ContinuousAndCategoricalValues(
          continuous=inputs.continuous.padded_array,
          categorical=inputs.categorical.padded_array,
      )

    observation_noise_variance = yield sp.ModelParameter(
        init_fn=_log_uniform_init(*observation_noise_bounds),
        constraint=sp.Constraint(
            observation_noise_bounds,
            tfb.SoftClip(*observation_noise_bounds, hinge_softness=1e-2),
        ),
        regularizer=lambda x: 0.01 * jnp.log(x / 0.0039) ** 2,
        name='observation_noise_variance',
    )
    cholesky_fn = None
    # When cholesky fails, increase jitters and retry.
    if self._use_retrying_cholesky:
      retrying_cholesky = functools.partial(
          tfp.experimental.distributions.marginal_fns.retrying_cholesky,
          jitter=np.float64(1e-4),
          max_iters=5,
      )
      cholesky_fn = lambda matrix: retrying_cholesky(matrix)[0]

    # The mean function output must broadcast to
    # `[batch_shape, num_observations]`, where `batch_shape` is the
    # (possibly empty) batch shape of the other hyperparameters. Initializing
    # `mean_fn_constant` with an array of shape `[1]` gives the mean function
    # output a shape of `[batch_shape, 1]`, ensuring that batch dimensions line
    # up properly.
    mean_fn_constant = yield sp.ModelParameter(
        init_fn=lambda k: jax.random.normal(key=k, shape=[1]),
        regularizer=lambda x: 0.5 * jnp.squeeze(x, axis=-1) ** 2,
        name='mean_fn',
    )

    mean_fn = ConstantMeanFn(mean_fn_constant * self._linear_coef)

    return tfd.GaussianProcess(
        kernel,
        index_points=inputs,
        observation_noise_variance=observation_noise_variance,
        cholesky_fn=cholesky_fn,
        mean_fn=mean_fn,
    )
