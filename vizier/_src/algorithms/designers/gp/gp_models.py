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

"""Gaussian Process models."""

import logging
from typing import Iterable, Optional, Union

import chex
import equinox as eqx
import jax
from jax import numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.algorithms.designers.gp import transfer_learning as vtl
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier._src.jax.models import multitask_tuned_gp_models
from vizier._src.jax.models import tuned_gp_models
from vizier.jax import optimizers
from vizier.utils import profiler

tfd = tfp.distributions


class GPTrainingSpec(eqx.Module):
  """Encapsulates all the information needed to train a singular GP model.

  Attributes:
    ard_optimizer: An `Optimizer` which should return a batch of hyperparameters
      to be ensembled.
    ard_rng: PRNG key for the ARD optimization.
    coroutine: The model coroutine.
    ensemble_size: If set, ensembles `ensemble_size` GP models together.
    ard_random_restarts: The number of random restarts.
  """

  ard_optimizer: optimizers.Optimizer[types.ParameterDict]
  ard_rng: jax.Array
  coroutine: sp.ModelCoroutine
  ensemble_size: int = eqx.field(static=True, default=1)
  ard_random_restarts: int = eqx.field(
      static=True, default=optimizers.DEFAULT_RANDOM_RESTARTS
  )


class GPState(eqx.Module):
  """A GP model and its training data. Implements `Predictive`.

  The data is kept around to deduce degrees of freedom and other related
  metrics. This implements `Predictive`, so that it and any of its dervied
  classes like `StackedResidualGP` can be used as a `Predictive` in
  `acquisitions.py`.
  """

  predictive: sp.UniformEnsemblePredictive
  data: types.ModelData

  def predict_with_aux(
      self, features: types.ModelInput
  ) -> tuple[tfd.Distribution, chex.ArrayTree]:
    return self.predictive.predict_with_aux(features)

  def num_hyperparameters(self) -> int:
    """Returns the number of hyperparameters optimized on `data`."""

    # For a GP model, this is feature dimensionality + 2
    # (length scales, amplitude, observation noise)
    # TODO: Compute this from the params returned by the ard
    # optimizer
    return (
        self.data.features.continuous.shape[1]  # (num_samples, num_features)
        + self.data.features.categorical.shape[1]  # (num_samples, num_features)
        + 2
    )


class StackedResidualGP(GPState):
  """GP that implements the `predictive` interface and contains stacked GPs.

  This GP handles sequential transfer learning. This holds one or no base
  (prior) GPs, along with a current top-level GP. The training process is such
  that the 'top' GP is trained on the residuals of the predictions from the
  base GP. The inference process is such that the predictions of the base
  GP and the 'top' GP are combined together.

  The base GP may also have its own base GPs and be a `StackedResidualGP`.
  """

  # `base_gp` refers to a GP trained and conditioned on previous data for
  # transfer learning. The top level GP is trained on the residuals from
  # `base_gp` on `data`.
  # If `None`, no transfer learning is used and all predictions happen through
  # `predictive`.
  base_gp: Optional[GPState] = None

  def predict_with_aux(
      self, features: types.ModelInput
  ) -> tuple[tfd.Distribution, chex.ArrayTree]:
    # Override the existing implementation of `predict_with_aux` to handle
    # combining `predictive` with `base_gp`.
    if not self.base_gp:
      return self.predictive.predict_with_aux(features)

    base_pred_dist, base_aux = self.base_gp.predict_with_aux(features)
    top_pred_dist, top_aux = self.predictive.predict_with_aux(features)

    base_pred = vtl.TransferPredictionState(
        pred=base_pred_dist,
        aux=base_aux,
        training_data_count=self.base_gp.data.labels.shape[0],
        num_hyperparameters=self.num_hyperparameters(),
    )
    top_pred = vtl.TransferPredictionState(
        pred=top_pred_dist,
        aux=top_aux,
        training_data_count=self.data.labels.shape[0],
        num_hyperparameters=self.num_hyperparameters(),
    )

    # TODO: Decide what to do with
    # `expected_base_stddev_mismatch` - currently set to default.
    comb_dist, aux = vtl.combine_predictions_with_aux(
        top_pred=top_pred, base_pred=base_pred
    )

    return comb_dist, aux


def get_vizier_gp_coroutine(
    data: types.ModelData,
    *,
    linear_coef: float = 0.0,
) -> sp.ModelCoroutine:
  """Gets a GP model coroutine.

  Args:
    data: The data used to the train the GP model
    linear_coef: If non-zero, uses a linear kernel with `linear_coef`
      hyperparameter.

  Returns:
    The model coroutine.
  """
  # Construct the multi-task GP.
  labels_shape = data.labels.shape
  if labels_shape[-1] > 1:
    gp_coroutine = multitask_tuned_gp_models.VizierMultitaskGaussianProcess(
        _feature_dim=types.ContinuousAndCategorical[int](
            data.features.continuous.padded_array.shape[-1],
            data.features.categorical.padded_array.shape[-1],
        ),
        _num_tasks=labels_shape[-1],
    )
    return sp.StochasticProcessModel(gp_coroutine).coroutine

  if linear_coef:
    return tuned_gp_models.VizierLinearGaussianProcess.build_model(
        features=data.features, linear_coef=linear_coef
    ).coroutine

  return tuned_gp_models.VizierGaussianProcess.build_model(
      data.features
  ).coroutine


def _train_gp(spec: GPTrainingSpec, data: types.ModelData) -> GPState:
  """Trains a Gaussian Process model.

  1. Performs ARD to find the best model parameters.
  2. Pre-computes the Cholesky decomposition for the model.

  Args:
    spec: Spec required to train the GP. See `GPTrainingSpec` for more info.
    data: Data on which to train the GP.

  Returns:
    The trained GP model.
  """
  jax.monitoring.record_event(
      '/vizier/jax/designers/gp_bandit/train_gp', scope=profiler.current_scope()
  )

  jax.monitoring.record_event(
      '/vizier/jax/train_gp_with_data_shapes',
      **{
          'num_rows': data.features.categorical.shape[0],
          'num_categoricals': data.features.categorical.shape[1],
          'num_continuous': data.features.continuous.shape[1],
          'num_labels': (
              data.labels.shape[1] if data.labels.padded_array.ndim == 2 else 1
          ),
      },
  )
  model = sp.CoroutineWithData(spec.coroutine, data)

  # Optimize the parameters
  ard_rngs = jax.random.split(spec.ard_rng, spec.ard_random_restarts + 1)
  best_n = spec.ensemble_size or 1
  best_params, _ = spec.ard_optimizer(
      eqx.filter_jit(eqx.filter_vmap(model.setup))(ard_rngs[1:]),
      model.loss_with_aux,
      ard_rngs[0],
      constraints=model.constraints(),
      best_n=best_n,
  )
  if best_n == 1 and all(x.shape[0] == 1 for x in best_params.values()):
    best_params = jax.tree_util.tree_map(
        lambda x: jnp.squeeze(x, axis=0), best_params
    )
  best_models = sp.StochasticProcessWithCoroutine(
      coroutine=spec.coroutine, params=best_params
  )
  # Logging for debugging purposes.
  logging.info(
      'Best models: %s', eqx.tree_pformat(best_models, short_arrays=False)
  )
  predictive = sp.UniformEnsemblePredictive(
      eqx.filter_jit(best_models.precompute_predictive)(data)
  )
  return GPState(predictive=predictive, data=data)


@jax.jit
def _pred_mean(
    pred: acquisitions.Predictive, features: types.ModelInput
) -> types.Array:
  """Returns the mean of the predictions from `pred` on `features`.

  Workaround while `eqx.filter_jit(pred.pred_with_aux)(features)` is broken
  due to a bug in tensorflow probability.

  Args:
    pred: `Predictive` to predict with.
    features: Xs to predict on.

  Returns:
    Means of the predictions from `pred` on `features`.
  """
  return pred.predict_with_aux(features)[0].mean()


def train_stacked_residual_gp(
    base_gp: GPState,
    spec: GPTrainingSpec,
    data: types.ModelData,
) -> StackedResidualGP:
  """Trains a `StackedResidualGP`.

  Completes the following steps in order:
    1. Uses `base_gp` to predict on the `data`
    2. Computes the residuals from the above predictions
    3. Trains a top-level GP on the above residuals
    4. Returns a `StackedResidualGP` combining the base GP and newly-trained
    GP.

  Args:
    base_gp: The predictive to use as the base GP for the `StackedResidualGP`
      training.
    spec: Training spec for the top level GP.
    data: Training data for the top level GP.

  Returns:
    The trained `StackedResidualGP`.
  """
  # Compute the residuals of `data` as predicted by `base_gp`
  pred_means = _pred_mean(base_gp, data.features)

  has_no_padding = ~(
      data.features.continuous.is_missing[0]
      | data.features.categorical.is_missing[0]
      | data.labels.is_missing[0]
  )

  # Scope this to non-padded predictions only.
  pred_means_no_padding = pred_means[has_no_padding]
  residuals = (
      data.labels.unpad().reshape(pred_means_no_padding.shape)
      - pred_means_no_padding
  )

  # Train on the re-padded residuals
  residual_labels = types.PaddedArray.from_array(
      array=residuals,
      target_shape=data.labels.shape,
      fill_value=data.labels.fill_value,
  )
  data_with_residuals = types.ModelData(
      features=data.features, labels=residual_labels
  )

  top_gp = _train_gp(spec=spec, data=data_with_residuals)
  return StackedResidualGP(
      predictive=top_gp.predictive,
      data=top_gp.data,
      base_gp=base_gp,
  )


def train_gp(
    spec: Union[GPTrainingSpec, Iterable[GPTrainingSpec]],
    data: Union[types.ModelData, Iterable[types.ModelData]],
) -> GPState:
  """Trains a Gaussian Process model.

  If `spec` contains multiple elements, each will be used to train a
  `StackedResidualGP`, sequentially. The first entry will be used to train the
  first GP, and then subsequent GPs will be trained on the residuals from the
  previous GP. This process completes in the order that `spec` and `data are
  provided, such that `spec[0]` is the first GP trained and `spec[-1]` is the
  last GP trained.

  spec[-1] and data[-1] make up the top-level GP, and spec[:-1] and data[:-1]
  define the priors in context of transfer learning.

  Args:
    spec: Specification for how to train a GP model. If multiple specs are
      provided, transfer learning will train multiple models and combine into a
      single GP.
    data: Data on which to train GPs. NOTE: `spec` and `data` must be of the
      same shape. Trains a GP on `data[i]` with `spec[i]`.

  Returns:
    The trained GP model.
  """
  is_singleton_spec = isinstance(spec, GPTrainingSpec)
  is_singleton_data = isinstance(data, types.ModelData)
  if is_singleton_spec != is_singleton_data:
    raise ValueError(
        '`train_gp` expected the shapes of `spec` and `data`  to be identical.'
        f' Instead got `data` {data} but `spec` {spec}.'
    )

  if is_singleton_spec and is_singleton_data:
    return _train_gp(spec=spec, data=data)

  if len(spec) != len(data):
    raise ValueError(
        '`train_gp` expected the shapes of `spec` and `data` to be identical.'
        f' Instead got `spec` of length {len(spec)} but `data` of length'
        f' {len(data)}. `spec` was {spec} and `data` was {data}.'
    )

  curr_gp: Optional[GPState] = None
  for curr_spec, curr_data in zip(spec, data):
    if curr_gp is None:
      # We are on the first iteration.
      curr_gp = _train_gp(spec=curr_spec, data=curr_data)
    else:
      # Otherwise, we have a base GP to use - the GP trained on the last
      # iteration.
      curr_gp = train_stacked_residual_gp(
          base_gp=curr_gp,
          spec=curr_spec,
          data=curr_data,
      )

  if curr_gp is None:
    raise ValueError(
        f'Failed to train a GP with provided training spec: {spec} and'
        f' data: {data}. `curr_gp` was never updated. This should never happen.'
    )
  return curr_gp
