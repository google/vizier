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

"""Gaussian Process models."""

import logging
import equinox as eqx
import jax
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier._src.jax.models import tuned_gp_models
from vizier.jax import optimizers

tfd = tfp.distributions


class GPState(eqx.Module):
  """Stores a GP model `predictive` and the data used during training.

  The data is kept around to deduce degrees of freedom and other related
  metrics.
  """

  predictive: sp.UniformEnsemblePredictive
  data: types.ModelData


def get_vizier_gp_coroutine(
    features: types.ModelInput, *, linear_coef: float = 0.0
) -> sp.ModelCoroutine:
  """Gets a GP model coroutine.

  Args:
    features: The features used to the train the GP model
    linear_coef: If non-zero, uses a linear kernel with `linear_coef`
      hyperparameter.

  Returns:
    The model coroutine.
  """
  if linear_coef:
    return tuned_gp_models.VizierLinearGaussianProcess.build_model(
        features=features, linear_coef=linear_coef
    ).coroutine

  return tuned_gp_models.VizierGaussianProcess.build_model(features).coroutine


def train_gp(
    data: types.ModelData,
    ard_optimizer: optimizers.Optimizer[types.ParameterDict],
    ard_rng: jax.random.KeyArray,
    *,
    coroutine: sp.ModelCoroutine,
    ensemble_size: int = 1,
    ard_random_restarts: int = optimizers.DEFAULT_RANDOM_RESTARTS,
) -> GPState:
  """Trains a Gaussian Process model.

  1. Performs ARD to find the best model parameters.
  2. Pre-computes the Cholesky decomposition for the model.

  Args:
    data: Data to train the GP model(s) on.
    ard_optimizer: An `Optimizer` which should return a batch of hyperparameters
      to be ensembled.
    ard_rng: PRNG key for the ARD optimization.
    coroutine: The model coroutine.
    ensemble_size: If set, ensembles `ensemble_size` GP models together.
    ard_random_restarts: The number of random restarts.

  Returns:
    The trained GP model.
  """
  model = sp.CoroutineWithData(coroutine, data)

  # Optimize the parameters
  ard_rngs = jax.random.split(ard_rng, ard_random_restarts + 1)
  best_params, _ = ard_optimizer(
      eqx.filter_jit(eqx.filter_vmap(model.setup))(ard_rngs[1:]),
      model.loss_with_aux,
      ard_rngs[0],
      constraints=model.constraints(),
      best_n=ensemble_size or 1,
  )
  best_models = sp.StochasticProcessWithCoroutine(
      coroutine=coroutine, params=best_params
  )
  # Logging for debugging purposes.
  logging.info(
      'Best models: %s', eqx.tree_pformat(best_models, short_arrays=False)
  )
  predictive = sp.UniformEnsemblePredictive(
      eqx.filter_jit(best_models.precompute_predictive)(data)
  )

  return GPState(predictive=predictive, data=data)
