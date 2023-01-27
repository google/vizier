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

"""HEBO GP models.


Faithful reimplementation of HEBO model based off of:
Paper: https://arxiv.org/abs/2012.03826
Repo: https://github.com/huawei-noah/HEBO.
"""

from typing import Generator
import attr
import chex
import jax
from jax import numpy as jnp
from jax.config import config
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax.optimizers import optimizers

config.update('jax_enable_x64', True)

tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels


@attr.define
class VizierHeboGaussianProcess(sp.ModelCoroutine[chex.Array,
                                                  tfd.GaussianProcess]):
  """Hebo Gaussian process model."""

  @classmethod
  def model_and_loss_fn(
      cls,
      features: chex.Array,
      labels: chex.Array,
  ) -> tuple[sp.StochasticProcessModel, optimizers.LossFunction]:
    """Returns the model and loss function."""
    gp_coroutine = VizierHeboGaussianProcess()
    model = sp.StochasticProcessModel(gp_coroutine)

    # Define the ARD loss function.
    def loss_fn(params):
      gp, mutables = model.apply({'params': params},
                                 features,
                                 mutable=['losses', 'predictive'])
      # Normalize so we can use the same learning rate regardless of
      # how many examples we have.
      loss = (-gp.log_prob(labels) + jax.tree_util.tree_reduce(
          jax.numpy.add, mutables['losses'])) / features.shape[0]
      return loss, dict()

    return model, loss_fn

  def __call__(
      self,
      inputs: chex.Array = None
  ) -> Generator[sp.ModelParameter, chex.Array, tfd.GaussianProcess]:
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
            loc=jnp.float64(0.0), scale=jnp.float64(1.0), name='length_scale'),
        constraint=sp.Constraint((0., None), bijector=tfb.Exp()))
    kernel = tfpk.FeatureScaled(kernel, scale_diag=length_scale)

    # Kumaraswamy input warping
    kumar_log_normal = tfd.LogNormal(loc=jnp.float64(0.0), scale=0.75)

    concentration0 = yield sp.ModelParameter.from_prior(
        kumar_log_normal.copy(name='concentration0'),
        constraint=sp.Constraint((0., 10.), bijector=sigmoid))

    concentration1 = yield sp.ModelParameter.from_prior(
        kumar_log_normal.copy(name='concentration1'),
        constraint=sp.Constraint((0., 10.), bijector=sigmoid))

    kernel = tfpk.KumaraswamyTransformed(kernel, concentration1, concentration0)

    return tfd.GaussianProcess(
        kernel,
        index_points=inputs,
        observation_noise_variance=observation_noise_variance,
        cholesky_fn=None,
        always_yield_multivariate_normal=True)
