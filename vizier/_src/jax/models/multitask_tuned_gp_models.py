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

"""Collection of multitask GP models."""

import enum
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
tfde = tfp.experimental.distributions
tfpk = tfp.math.psd_kernels
tfpke = tfp.experimental.psd_kernels


class MultiTaskType(enum.Enum):
  """The form of the MultiTask GP."""

  # Use a TFP MultiTask GP with an Independent kernel. A base (prior) kernel
  # matrix is shared across all tasks, and the overall kernel matrix is the
  # Kronecker product of the base kernel matrix and the identity.
  INDEPENDENT = 'independent'

  # Use a TFP MultiTask GP with a Separable kernel. The prior kernel matrix
  # is the Kronecker product of a base kernel matrix and a task kernel matrix
  # (assumed to be full-covariance).
  SEPARABLE_NORMAL_TASK_KERNEL_PRIOR = 'separable_normal_task_kernel_prior'

  # Use an LKJ-distributed prior for the (full-covariance) Separable task
  # kernel.
  SEPARABLE_LKJ_TASK_KERNEL_PRIOR = 'separable_lkj_task_kernel_prior'

  # Use a diagonal matrix for the Separable task kernel.
  SEPARABLE_DIAG_TASK_KERNEL_PRIOR = 'separable_diag_task_kernel_prior'


@struct.dataclass
class VizierMultitaskGaussianProcess(
    sp.ModelCoroutine[Union[tfd.GaussianProcess, tfde.MultiTaskGaussianProcess]]
):
  """Multitask GP model using priors from Vizier's tuned GP."""

  _feature_dim: types.ContinuousAndCategorical[int] = struct.field(
      pytree_node=False
  )
  _num_tasks: int = struct.field(pytree_node=False)
  _multitask_type: MultiTaskType = struct.field(
      default=MultiTaskType.INDEPENDENT, kw_only=True, pytree_node=False
  )
  _use_retrying_cholesky: bool = struct.field(
      default=True, kw_only=True, pytree_node=False
  )
  _boundary_epsilon: float = struct.field(default=1e-12, kw_only=True)

  def _log_uniform_init(
      self,
      low: Union[float, np.floating],
      high: Union[float, np.floating],
      shape: tuple[int, ...] = tuple(),
  ) -> sp.InitFn:
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
      unif = jax.random.uniform(key, shape, dtype=jnp.float64)
      return jnp.exp(unif * jnp.log(high / low) + jnp.log(low))

    return sample

  def _build_task_kernel_scale_linop(
      self,
  ) -> Generator[
      sp.ModelParameter, jax.Array, tfp.tf2jax.linalg.LinearOperator
  ]:
    if self._multitask_type == MultiTaskType.SEPARABLE_DIAG_TASK_KERNEL_PRIOR:
      correlation_diag = yield sp.ModelParameter.from_prior(
          tfd.Sample(
              tfd.Uniform(low=jnp.float64(1e-6), high=1.0),
              sample_shape=self._num_tasks,
              name='correlation_diag',
          ),
          constraint=sp.Constraint(
              bounds=(1e-6, 1.0),
              bijector=tfb.Sigmoid(low=jnp.float64(1e-6), high=1.0),
          ),
      )
      task_kernel_scale_linop = tfp.tf2jax.linalg.LinearOperatorDiag(
          correlation_diag
      )
    elif self._multitask_type == MultiTaskType.SEPARABLE_LKJ_TASK_KERNEL_PRIOR:
      # Generate parameters for the Cholesky of the task kernel matrix,
      # which accounts for correlations between tasks.
      num_task_kernel_entries = tfb.CorrelationCholesky().inverse_event_shape(
          [self._num_tasks, self._num_tasks]
      )
      correlation_cholesky_vec = yield sp.ModelParameter(
          init_fn=lambda key: tfd.Sample(  # pylint: disable=g-long-lambda
              tfd.Normal(jnp.float64(0.0), 1.0), num_task_kernel_entries
          ).sample(seed=key),
          # Use `jnp.copy` to prevent tracers leaking from bijector cache.
          regularizer=lambda x: -tfd.CholeskyLKJ(  # pylint: disable=g-long-lambda
              dimension=self._num_tasks, concentration=1.0
          ).log_prob(tfb.CorrelationCholesky()(jnp.copy(x))),
          name='task_kernel_correlation_cholesky_vec',
      )

      task_kernel_correlation_cholesky = tfb.CorrelationCholesky()(
          jnp.copy(correlation_cholesky_vec)
      )

      task_kernel_scale_vec = yield sp.ModelParameter(
          init_fn=functools.partial(
              jax.random.uniform,
              shape=(self._num_tasks,),
              dtype=jnp.float64,
              minval=1e-6,
              maxval=1.0,
          ),
          constraint=sp.Constraint(
              bounds=(1e-6, 1.0),
              bijector=tfb.Sigmoid(low=jnp.float64(1e-6), high=1.0),
          ),
          name='task_kernel_sqrt_diagonal',
      )
      task_kernel_cholesky = (
          task_kernel_correlation_cholesky
          * task_kernel_scale_vec[:, jnp.newaxis]
      )

      # Build the `LinearOperator` object representing the task kernel matrix,
      # to parameterize the Separable kernel.
      task_kernel_scale_linop = tfp.tf2jax.linalg.LinearOperatorLowerTriangular(
          task_kernel_cholesky
      )
    elif (
        self._multitask_type == MultiTaskType.SEPARABLE_NORMAL_TASK_KERNEL_PRIOR
    ):
      # Generate parameters for the Cholesky of the task kernel matrix;
      # accounts for correlations between tasks. The task kernel matrix must
      # be positive definite, so we construct it via a Cholesky factor.
      # Define the prior of the kernel task matrix to be centered at the
      # identity.
      prior_mean = jnp.eye(self._num_tasks, dtype=jnp.float64)
      prior_mean_vec = tfb.FillTriangular().inverse(prior_mean)
      prior_mean_batched = jnp.broadcast_to(
          prior_mean_vec, prior_mean_vec.shape
      )

      task_kernel_cholesky_entries = yield sp.ModelParameter.from_prior(
          tfd.Independent(
              tfd.Normal(prior_mean_batched, 1.0),
              reinterpreted_batch_ndims=1,
              name='task_kernel_cholesky_entries',
          )
      )

      # Apply a bijector to pack the task kernel entries into a lower
      # triangular matrix and ensure the diagonal is positive.
      task_kernel_bijector = tfb.Chain([
          tfb.TransformDiagonal(
              tfb.Chain([tfb.Shift(jnp.float64(1e-6)), tfb.Softplus()])
          ),
          tfb.FillTriangular(),
      ])
      task_kernel_cholesky = task_kernel_bijector(
          jnp.copy(task_kernel_cholesky_entries)
      )

      # Build the `LinearOperator` object representing the task kernel
      # matrix, to parameterize the Separable kernel.
      task_kernel_scale_linop = tfp.tf2jax.linalg.LinearOperatorLowerTriangular(
          task_kernel_cholesky
      )
    else:
      raise ValueError(f'Unsupported multitask type: {self._multitask_type}')

    return task_kernel_scale_linop

  def __call__(
      self, inputs: Optional[types.ModelInput] = None
  ) -> Generator[
      sp.ModelParameter,
      jax.Array,
      Union[tfd.GaussianProcess, tfde.MultiTaskGaussianProcess],
  ]:
    """Creates a generator.

    Args:
      inputs: Floating array of dimension (num_examples, _feature_dim).

    Yields:
      Multitask GP hyperparameters.
    """

    eps = self._boundary_epsilon
    observation_noise_bounds = (np.float64(1e-10 - eps), 1.0 + eps)
    amplitude_bounds = (np.float64(1e-3 - eps), 10.0 + eps)
    continuous_ones = np.ones((self._feature_dim.continuous), dtype=np.float64)
    continuous_length_scale_bounds = (
        continuous_ones * (1e-2 - eps),
        continuous_ones * 1e2 + eps,
    )
    categorical_ones = np.ones(
        (self._feature_dim.categorical), dtype=np.float64
    )
    categorical_length_scale_bounds = (
        categorical_ones * (1e-2 - eps),
        categorical_ones * 1e2 + eps,
    )

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
    continuous_length_scale_squared = yield sp.ModelParameter(
        init_fn=self._log_uniform_init(
            *continuous_length_scale_bounds,
            shape=(self._feature_dim.continuous,),
        ),
        constraint=sp.Constraint(
            continuous_length_scale_bounds,
            tfb.SoftClip(*continuous_length_scale_bounds, hinge_softness=1e-2),
        ),
        regularizer=lambda x: jnp.sum(0.01 * jnp.log(x / 0.5) ** 2),
        name='continuous_length_scale_squared',
    )
    categorical_length_scale_squared = yield sp.ModelParameter(
        init_fn=self._log_uniform_init(
            *categorical_length_scale_bounds,
            shape=(self._feature_dim.categorical,),
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

    bias_amplitude = yield sp.ModelParameter.from_prior(
        tfd.Normal(
            jnp.zeros(shape=tuple(), dtype=jnp.float64),
            1.0,
            name='bias_amplitude',
        ),
    )
    slope_amplitude = yield sp.ModelParameter.from_prior(
        tfd.Normal(
            jnp.zeros(shape=tuple(), dtype=jnp.float64),
            1.0,
            name='slope_amplitude',
        ),
    )
    shift = yield sp.ModelParameter.from_prior(
        tfd.Normal(
            jnp.zeros(shape=tuple(), dtype=jnp.float64),
            1.0,
            name='shift',
        )
    )
    kernel = kernel + continuous_only_kernel.ContinuousOnly(
        tfpk.FeatureScaled(
            tfpk.Linear(
                slope_amplitude=slope_amplitude,
                bias_amplitude=bias_amplitude,
                shift=shift,
            ),
            scale_diag=jnp.sqrt(continuous_length_scale_squared),
        )
    )

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

    if inputs is not None:
      kernel = mask_features.MaskFeatures(
          kernel,
          dimension_is_missing=tfpke.ContinuousAndCategoricalValues(
              continuous=inputs.continuous.is_missing[1],
              categorical=inputs.categorical.is_missing[1],
          ),
      )

      inputs = tfpke.ContinuousAndCategoricalValues(
          inputs.continuous.padded_array,
          inputs.categorical.padded_array,
      )

    # Creates multitask kernel from single-task kernel. Defaults to INDEPENDENT
    # for multitask kernel.
    if self._multitask_type == MultiTaskType.INDEPENDENT:
      multitask_kernel = tfpke.Independent(self._num_tasks, kernel)
    else:
      task_kernel_scale_linop = yield from self._build_task_kernel_scale_linop()
      multitask_kernel = tfpke.Separable(
          self._num_tasks,
          base_kernel=kernel,
          task_kernel_scale_linop=task_kernel_scale_linop,
      )

    return tfde.MultiTaskGaussianProcess(
        multitask_kernel,
        index_points=inputs,
        observation_noise_variance=observation_noise_variance,
        cholesky_fn=cholesky_fn,
    )
