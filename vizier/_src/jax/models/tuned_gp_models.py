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

"""Collection of well-tuned GP models."""

# TODO: Add Ax/BoTorch GP.

import functools
from typing import Any, Generator

import attr
import chex
import jax
from jax import numpy as jnp
from jax.config import config
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax.optimizers import optimizers

# Jax disables float64 computations by default and will silently convert
# float64s to float32s. We must explicitly enable float64.
config.update('jax_enable_x64', True)

tfb = tfp.bijectors
tfd = tfp.distributions
Array = Any
tfpk = tfp.math.psd_kernels


@attr.define
class VizierGaussianProcess(sp.ModelCoroutine[chex.Array, tfd.GaussianProcess]):
  """Vizier's tuned GP.

  See __call__ method documentation.

  Attributes:
    _boundary_epsilon: We expand the constraints by this number so that the
      values exactly at the boundary can be mapped to unconstrained space. i.e.
      we are trying to avoid SoftClip(low=1e-2, high=1.).inverse(1e-2) giving
      NaN.
  """

  _feature_dim: int
  _use_retrying_cholesky: bool = attr.field(default=True, kw_only=True)
  _boundary_epsilon: float = attr.field(default=1e-12, kw_only=True)

  @classmethod
  def model_and_loss_fn(
      cls,
      features: chex.Array,
      labels: chex.Array,
      *,
      use_retrying_cholesky: bool = True,
  ) -> tuple[sp.StochasticProcessModel, optimizers.LossFunction]:
    """Returns the model and loss function."""
    gp_coroutine = VizierGaussianProcess(
        features.shape[-1], use_retrying_cholesky=use_retrying_cholesky
    )
    model = sp.StochasticProcessModel(gp_coroutine)

    # Run ARD.
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

  def _log_uniform_init(
      self, low: float, high: float, shape: tuple[int,
                                                  ...] = tuple()) -> sp.InitFn:
    r"""Take log-uniform sample in the constraint and map it back to \R.

    Args:
      low: Parameter lower bound.
      high: Parameter upper bound.
      shape: Returned array has this shape. Each entry in the returned array is
        an i.i.d sample.

    Returns:
      Randomly sampled array.
    """

    def sample(key: Any) -> jnp.ndarray:
      unif = jnp.array(jax.random.uniform(key, shape, dtype=jnp.float64))
      return jnp.exp(unif * jnp.log(high / low) + jnp.log(low))

    return sample

  def __call__(
      self,
      inputs: Array = None
  ) -> Generator[sp.ModelParameter, Array, tfd.GaussianProcess]:
    """Creates a generator.

    Args:
      inputs: Floating array of dimension (num_examples, _feature_dim).

    Yields:
      GaussianProcess whose event shape is `num_examples`.
    """

    eps = self._boundary_epsilon
    observation_noise_bounds = (np.float64(1e-10 - eps), 1.0 + eps)
    amplitude_bounds = (np.float64(1e-3 - eps), 10.0 + eps)
    ones = np.ones((self._feature_dim,), dtype=np.float64)
    length_scale_bounds = (ones * (1e-2 - eps), ones * 1e2 + eps)

    signal_variance = yield sp.ModelParameter(
        init_fn=self._log_uniform_init(*amplitude_bounds),
        constraint=sp.Constraint(
            amplitude_bounds,
            tfb.SoftClip(*amplitude_bounds, hinge_softness=1e-2),
        ),
        regularizer=lambda x: 0.01 * jnp.log(x / 0.039) ** 2,
        name='signal_variance',
    )
    kernel = tfpk.MaternFiveHalves(amplitude=jnp.sqrt(signal_variance))

    length_scale = yield sp.ModelParameter(
        init_fn=self._log_uniform_init(
            *length_scale_bounds, shape=(self._feature_dim,)
        ),
        constraint=sp.Constraint(
            length_scale_bounds,
            tfb.SoftClip(*length_scale_bounds, hinge_softness=1e-2),
        ),
        regularizer=lambda x: jnp.sum(0.01 * jnp.log(x / 0.5) ** 2),
        name='length_scale_squared',
    )
    kernel = tfpk.FeatureScaled(kernel, scale_diag=jnp.sqrt(length_scale))

    observation_noise_variance = yield sp.ModelParameter(
        init_fn=self._log_uniform_init(*observation_noise_bounds),
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
        always_yield_multivariate_normal=True,
    )
