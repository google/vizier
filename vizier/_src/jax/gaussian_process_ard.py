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

"""Coroutine for a trainable Gaussian process with an ARD kernel."""

from typing import Any, Generator, Optional, Type, Union

from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import stochastic_process_model as sp_model

Array = Any
tfd = tfp.distributions
tfpk = tfp.math.psd_kernels
tfpke = tfp.experimental.psd_kernels


class GaussianProcessARD(sp_model.ModelCoroutine):
  """Specifies a Gaussian process model with an ARD kernel.

  `GaussianProcessARD` satisfies the `ModelCoroutine` Protocol. By default,
  `GaussianProcessARD` uses a Matern Five Halves kernel. Hyperparameter priors
  are LogNormal(0., 1.).

  #### Examples

  Build a Gaussian Process model with an ARD Exponentiated Quadratic kernel and
  LogNormal(0., 1.) hyperparameter priors for 5-dimensional data.

  ```python
  from tensorflow_probability.substrates import jax as tfp

  tfpk = tfp.math.psd_kernels

  dim = 5
  gen = GaussianProcessARD(
      dimension=dim,
      kernel_class=tfpk.ExponentiatedQuadratic)
  ```
  """

  def __init__(self,
               dimension: int,
               kernel_class: Union[
                   Type[tfpk.MaternFiveHalves], Type[tfpk.MaternThreeHalves],
                   Type[tfpk.MaternOneHalf], Type[tfpk.Parabolic],
                   Type[tfpk.ExponentiatedQuadratic]] = tfpk.MaternFiveHalves,
               *,
               use_tfp_runtime_validation: bool = False):
    """Initializes a `GaussianProcessARD`.

    Args:
      dimension: The data dimensionality.
      kernel_class: The Gaussian process' kernel type.
      use_tfp_runtime_validation: If True, run additional runtime checks on the
        validity of the parameters and input for TFP objects (e.g. verify that
        amplitude and length scale parameters are positive). Runtime checks may
        be expensive and should be used only during development/debugging.
    """
    self.dimension = dimension
    self._kernel_class = kernel_class
    self._use_tfp_runtime_validation = use_tfp_runtime_validation

  def __call__(
      self, inputs: Optional[Array] = None
  ) -> Generator[sp_model.ModelParameter, Array, tfd.GaussianProcess]:
    # TODO: Determine why pylint doesn't allow both Returns and
    # Yields sections.
    # pylint: disable=g-doc-return-or-yield
    """The coroutine that specifies the GP model.

    Args:
      inputs: index_points to be provided to the GP.

    Yields:
      `ModelParameter`s describing the parameters to be declared in the Flax
        model.

    Returns:
      A tfd.GaussianProcess with the given index points.
    """
    amplitude = yield sp_model.ModelParameter.from_prior(
        tfd.LogNormal(0.0, 1.0, name='amplitude'),
        constraint=sp_model.Constraint(bounds=(jnp.array(0.0), None)),
    )
    kernel = self._kernel_class(
        amplitude=amplitude,
        length_scale=1.,
        validate_args=self._use_tfp_runtime_validation)
    inverse_length_scale = yield sp_model.ModelParameter.from_prior(
        tfd.Sample(
            tfd.LogNormal(0.0, 1.0),
            sample_shape=(self.dimension,),
            name='inverse_length_scale',
        ),
        constraint=sp_model.Constraint(bounds=(jnp.array(0.0), None)),
    )
    kernel = tfpk.FeatureScaled(
        kernel,
        inverse_scale_diag=inverse_length_scale,
        validate_args=self._use_tfp_runtime_validation)
    observation_noise_variance = yield sp_model.ModelParameter.from_prior(
        tfd.LogNormal(0.0, 1.0, name='observation_noise_variance'),
        constraint=sp_model.Constraint(bounds=(jnp.array(0.0), None)),
    )
    return tfd.GaussianProcess(
        kernel,
        index_points=inputs,
        observation_noise_variance=observation_noise_variance,
        validate_args=self._use_tfp_runtime_validation,
        always_yield_multivariate_normal=True)


class GaussianProcessARDWithCategorical(sp_model.ModelCoroutine):
  """Specifies an ARD Gaussian process model over continuous/categorical data.

  While `GaussianProcessARD` operates on a continuous domain,
  `GaussianProcessARDWithCategorical` operates on a domain that contains both
  continuous and categorical features. The GP's kernel, a
  `FeatureScaledWithCategorical` instance, operates on inputs of this type as
  well.

  `GaussianProcessARDWithCategorical` satisfies the `ModelCoroutine`
  Protocol. By default, `GaussianProcessARDWithCategorical` uses a Matern
  Five Halves kernel. Hyperparameter priors are LogNormal(0., 1.).

  #### Examples

  Build a Gaussian Process model with an ARD Exponentiated Quadratic kernel and
  LogNormal(0., 1.) hyperparameter priors for 5 continuous features and 3
  categorical features.

  ```python
  from tensorflow_probability.substrates import jax as tfp

  tfpk = tfp.math.psd_kernels
  tfpke = tfp.experimental.psd_kernels

  gen = GaussianProcessARDWithCategorical(
      dimension=tfpke.ContinuousAndCategoricalValues(
          continuous=5, categorical=3),
      kernel_class=tfpk.ExponentiatedQuadratic)
  ```
  """

  def __init__(
      self,
      dimension: tfpke.ContinuousAndCategoricalValues,
      kernel_class: Union[
          Type[tfpk.MaternFiveHalves],
          Type[tfpk.MaternThreeHalves],
          Type[tfpk.MaternOneHalf],
          Type[tfpk.Parabolic],
          Type[tfpk.ExponentiatedQuadratic],
      ] = tfpk.MaternFiveHalves,
      *,
      use_tfp_runtime_validation: bool = False
  ):
    """Initializes a `GaussianProcessARDWithCategorical`.

    Args:
      dimension: The dimensionality of the continuous/categorical features.
      kernel_class: The Gaussian process' kernel type.
      use_tfp_runtime_validation: If True, run additional runtime checks on the
        validity of the parameters and input for TFP objects (e.g. verify that
        amplitude and length scale parameters are positive). Runtime checks may
        be expensive and should be used only during development/debugging.
    """
    self.dimension = dimension
    self._kernel_class = kernel_class
    self._use_tfp_runtime_validation = use_tfp_runtime_validation

  def __call__(
      self, inputs: Optional[tfpke.ContinuousAndCategoricalValues] = None
  ) -> Generator[sp_model.ModelParameter, Array, tfd.GaussianProcess]:
    # pylint: disable=g-doc-return-or-yield
    """The coroutine that specifies the GP model.

    The inputs (and the index points of the returned Gaussian process) are
    instances of `tfpke.ContinuousAndCategoricalValues`, a `namedtuple` with
    a `"continuous"` field that contains continuous (float) data and a
    `"categorical"` field that contains categorical (int) data. The
    categorical data is encoded as integers (not one-hot encoded).

    Args:
      inputs: index_points to be provided to the GP.

    Yields:
      `ModelParameter`s describing the parameters to be declared in the Flax
        model.

    Returns:
      A `tfd.GaussianProcess` with the given index points.
    """
    amplitude = yield sp_model.ModelParameter.from_prior(
        tfd.LogNormal(0.0, 1.0, name='amplitude'),
        constraint=sp_model.Constraint(bounds=(jnp.array(0.0), None)),
    )
    kernel = self._kernel_class(
        amplitude=amplitude,
        length_scale=1.0,
        validate_args=self._use_tfp_runtime_validation,
    )
    inverse_length_scale_continuous = yield sp_model.ModelParameter.from_prior(
        tfd.Sample(
            tfd.LogNormal(0.0, 1.0),
            sample_shape=(self.dimension.continuous,),
            name='inverse_length_scale_continuous',
        ),
        constraint=sp_model.Constraint(bounds=(jnp.array(0.0), None)),
    )
    inverse_length_scale_categorical = yield sp_model.ModelParameter.from_prior(
        tfd.Sample(
            tfd.LogNormal(0.0, 1.0),
            sample_shape=(self.dimension.categorical,),
            name='inverse_length_scale_categorical',
        ),
        constraint=sp_model.Constraint(
            bounds=(jnp.zeros(self.dimension.categorical), None)
        ),
    )
    kernel = tfpke.FeatureScaledWithCategorical(
        kernel,
        inverse_scale_diag=tfpke.ContinuousAndCategoricalValues(
            inverse_length_scale_continuous, inverse_length_scale_categorical
        ),
        validate_args=self._use_tfp_runtime_validation,
    )
    observation_noise_variance = yield sp_model.ModelParameter.from_prior(
        tfd.LogNormal(0.0, 1.0, name='observation_noise_variance'),
        constraint=sp_model.Constraint(bounds=(jnp.array(0.0), None)),
    )
    return tfd.GaussianProcess(
        kernel,
        index_points=inputs,
        observation_noise_variance=observation_noise_variance,
        validate_args=self._use_tfp_runtime_validation,
        always_yield_multivariate_normal=True,
    )
