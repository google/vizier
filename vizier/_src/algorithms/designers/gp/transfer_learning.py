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

"""Algorithms for transfer learning."""

import chex
import equinox as eqx
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions


class TransferPredictionState(eqx.Module):
  """State and metadata for combining this prediction in transfer learning.

  Attributes:
    pred: The prediction to combine
    aux: Auxiliary information about the prediction
    training_data_count: Number of samples to train the `Predictive` that made
      the prediction.
    num_hyperparameters: Hyperparameters of the `Predictive` that made the
      prediction.
  """

  pred: tfd.Distribution
  aux: chex.ArrayTree
  training_data_count: int
  num_hyperparameters: int


def _compute_dof(training_data_count: int, num_hyperparameters: int) -> float:
  """Computes the DOF of a `Predictive`.

  This is a maximum of two measures of DOF.

  The first represents the DOF associated with a log likelihood computation,
  after optimizing the hyperparameters of the kernel, i.e. the
  degrees-of-freedom (dof) of a finite linear regression problem.

  The second represents the fact we know something about the standard
  deviation even when there are more hyperpameters than training data.

  Args:
    training_data_count: Number of samples used to train the `Predictive`.
    num_hyperparameters: Number of hyperparameters in the `Predictive`.

  Returns:
    Returns DOF of the predictive
  """
  return max(
      training_data_count - num_hyperparameters,
      training_data_count / (1 + num_hyperparameters),
  )


def combine_predictions_with_aux(
    top_pred: TransferPredictionState,
    base_pred: TransferPredictionState,
    *,
    expected_base_stddev_mismatch: float = 1.0
) -> tuple[tfd.Distribution, chex.ArrayTree]:
  """Combines two predictions from transfer learning.

  The means are combined as a simple sum.

  The standard deviations are combined using a geometric mean, with a
  weighting coefficient `alpha` that sets their relative importance.

  See the below code for the exact computation of `alpha`, which is a function
  of the uncertainty of the base to the uncertainty of the top-level
  prediction.
  Args:
    top_pred: Prediction from the top model (trained on the base's residuals)
    base_pred: Prediction from the base model (trained on the full data)
    expected_base_stddev_mismatch: Used for combining a base standard deviation
      with the top-level model's standard deviation estimate. Formally, it is
      the expected RMS fractional mismatch in standard deviation between a
      typical base and a typical top-level model (averaged over the feasible
      region). Allowable values are [0, 1] with (0.1, 0.8) being more likely.
      Assumes that the value is allowable, due to compatibility with `jax` and
      avoiding `jax.checkify`. Unexpected results may occur if value is set
      out-of-bounds.

  Returns:
    The combined distribution, assumed to be Normal, and auxiliary information.
  """

  dof_base = _compute_dof(
      training_data_count=base_pred.training_data_count,
      num_hyperparameters=base_pred.num_hyperparameters,
  )
  dof_top = _compute_dof(
      training_data_count=top_pred.training_data_count,
      num_hyperparameters=top_pred.num_hyperparameters,
  )

  # `beta_squared` is the ratio of uncertainty of the base to the uncertainty
  # in the top-level model.  More precisely, it is the
  # variance{ log { stddev returned by the base}} /
  # variance{ log { stddev returned by the top model}}.
  # This is a large number when the top-level stddev is more trustworthy, and
  # small when the base stddev is relatively trustworthy.
  beta_squared = (dof_top / dof_base) * (
      1 + dof_base + (expected_base_stddev_mismatch**2)
  )

  # Finally, compute the geometric mean weight, `alpha`.
  alpha = beta_squared / (1 + beta_squared)

  # Combine the means.
  comb_mean = top_pred.pred.mean() + base_pred.pred.mean()

  # Use `alpha` to combine the stddevs in a weighted geometric mean.
  comb_stddev = jnp.power(top_pred.pred.stddev(), alpha) * jnp.power(
      base_pred.pred.stddev(), (1 - alpha)
  )

  prev_aux = {
      'base_aux': base_pred.aux,
      'top_aux': top_pred.aux,
  }

  # Entries in `aux` must have the same batch shape as the predictions.
  batch_shape = comb_mean.shape[0]
  aux = {
      'prev_aux': prev_aux,
      'mean': comb_mean,
      'stddev': comb_stddev,
      'alpha': jnp.ones(batch_shape) * alpha,
      'expected_base_stddev_mismatch': (
          jnp.ones(batch_shape) * expected_base_stddev_mismatch
      ),
      'beta_squared': jnp.ones(batch_shape) * beta_squared,
  }

  # Assume a multivariate normal distribution with diagonal covariance as output
  return tfd.MultivariateNormalDiag(loc=comb_mean, scale_diag=comb_stddev), aux
