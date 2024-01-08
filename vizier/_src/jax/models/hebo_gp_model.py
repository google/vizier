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

"""HEBO GP models.


Faithful reimplementation of HEBO model based off of:
Paper: https://arxiv.org/abs/2012.03826
Repo: https://github.com/huawei-noah/HEBO.
"""

from typing import Generator, Optional
from flax import struct
import jax
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier._src.jax.models import continuous_only_kernel

tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels
tfpke = tfp.experimental.psd_kernels


@struct.dataclass
class VizierHeboGaussianProcess(sp.ModelCoroutine[tfd.GaussianProcess]):
  """Hebo Gaussian process model."""

  @classmethod
  def build_model(
      cls,
      features: types.ModelInput,
  ) -> sp.StochasticProcessModel:
    """Returns the model and loss function."""
    del features
    gp_coroutine = VizierHeboGaussianProcess()
    return sp.StochasticProcessModel(gp_coroutine)

  def __call__(
      self, inputs: Optional[types.ModelInput] = None
  ) -> Generator[sp.ModelParameter, jax.Array, tfd.GaussianProcess]:
    """Creates a generator.

    Args:
      inputs: tuple of (train_features, train_labels) train_features - array of
        dimension (num_examples, _feature_dim) train_labels - array of dimension
        (num_examples,)

    Yields:
      GaussianProcess whose event shape is `num_examples`.
    """
    epsilon = jnp.finfo(jnp.float64).resolution
    lim_val = jnp.float64(36.0)

    def _constraint_fn(x):
      return jnp.where(x > lim_val, x,
                       jnp.log1p(jnp.exp(jnp.clip(x, -lim_val,
                                                  lim_val)))) + epsilon

    def _inverse_constraint_fn(f):
      return jnp.where(f > lim_val, f, jnp.log(jnp.exp(f + 1e-20) - 1.))

    constraint_bijector = tfb.Inline(
        forward_fn=_constraint_fn,
        inverse_fn=_inverse_constraint_fn,
        forward_min_event_ndims=0)
    sigmoid = tfb.Sigmoid(jnp.float64(0.0), jnp.float64(10.0))

    # Signal variance
    gamma = tfd.Gamma(
        concentration=jnp.float64(0.5),
        rate=jnp.float64(1.0),
        name='signal_variance')
    signal_variance = yield sp.ModelParameter.from_prior(
        gamma,
        constraint=sp.Constraint((epsilon, None), bijector=constraint_bijector))

    # Observation noise variance
    noise_log_normal = tfd.LogNormal(
        loc=jnp.float64(-4.63), scale=0.5, name='observation_noise_variance')
    observation_noise_variance = yield sp.ModelParameter.from_prior(
        noise_log_normal,
        constraint=sp.Constraint((epsilon, None), bijector=constraint_bijector))

    # Kernel
    kernel = tfpk.MaternThreeHalves(
        amplitude=jnp.sqrt(signal_variance)) + tfpk.Linear()

    # Length scale
    length_scale = yield sp.ModelParameter.from_prior(
        tfd.LogNormal(
            loc=jnp.float64(0.0), scale=jnp.float64(1.0), name='length_scale'
        ),
        constraint=sp.Constraint((0.0, None), bijector=tfb.Exp()),
    )
    kernel = tfpk.FeatureScaled(kernel, scale_diag=length_scale)

    # Kumaraswamy input warping
    kumar_log_normal = tfd.LogNormal(loc=jnp.float64(0.0), scale=0.75)

    concentration0 = yield sp.ModelParameter.from_prior(
        kumar_log_normal.copy(name='concentration0'),
        constraint=sp.Constraint((0.0, 10.0), bijector=sigmoid),
    )

    concentration1 = yield sp.ModelParameter.from_prior(
        kumar_log_normal.copy(name='concentration1'),
        constraint=sp.Constraint((0.0, 10.0), bijector=sigmoid),
    )

    kernel = tfpk.KumaraswamyTransformed(kernel, concentration1, concentration0)

    kernel = continuous_only_kernel.ContinuousOnly(kernel)
    if inputs is not None:
      inputs = tfpke.ContinuousAndCategoricalValues(
          inputs.continuous.padded_array, inputs.categorical.padded_array
      )
    return tfd.GaussianProcess(
        kernel,
        index_points=inputs,
        observation_noise_variance=observation_noise_variance,
        cholesky_fn=None,
    )
