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

"""Utilities for GP Bandit."""

import functools
from typing import Optional

import jax
from jax import numpy as jnp
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import predictive_fns
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier.pyvizier.converters import feature_mapper


def precompute_cholesky(model, data, params):
  return model.apply(
      {'params': params},
      data.features,
      data.labels,
      method=model.precompute_predictive,
      mutable='predictive',
      observations_is_missing=data.label_is_missing,
  )


def optimize_acquisition(
    count: int,
    model: sp.StochasticProcessModel,
    acquisition_fn: acquisitions.AcquisitionFunction,
    optimizer: vb.VectorizedOptimizer,
    prior_features: types.Array,
    state: types.GPState,
    seed: jax.random.KeyArray,
    use_vmap: bool,
    trust_region: Optional[acquisitions.TrustRegion] = None,
    # TODO: Remove this when Eagle optimizer takes
    # ContinuousAndCategoricalFeatures.
    mapper: Optional[feature_mapper.ContinuousCategoricalFeatureMapper] = None,
    num_parallel_candidates: Optional[int] = None,
) -> vb.VectorizedStrategyResults:
  """Optimize the acquisition function."""
  base_score_fn = functools.partial(
      predictive_fns.acquisition_on_array,
      model=model,
      acquisition_fn=acquisition_fn,
      state=state,
      trust_region=trust_region,
      use_vmap=use_vmap,
  )
  if mapper is not None:
    mapped_score_fn = lambda xs: base_score_fn(mapper.map(xs))
  else:
    mapped_score_fn = base_score_fn
  if num_parallel_candidates:
    # Unflatten suggested features before passing them to the score_fn.
    score_fn = lambda x: mapped_score_fn(  # pylint: disable=g-long-lambda
        jax.tree_util.tree_map(
            lambda x_: jnp.reshape(  # pylint: disable=g-long-lambda
                x_,
                [
                    -1,
                    num_parallel_candidates,
                    x_.shape[-1] // num_parallel_candidates,
                ],
            ),
            x,
        )
    )
  else:
    score_fn = mapped_score_fn
  return optimizer(
      score_fn=score_fn, count=count, prior_features=prior_features, seed=seed
  )


def stochastic_process_model_loss_fn(
    params: types.ParameterDict,
    model: sp.StochasticProcessModel,
    data: types.StochasticProcessModelData,
    normalize: bool = False,
):
  """Loss function for a stochastic process model."""
  kwargs = {}
  if data.dimension_is_missing is not None:
    kwargs['dimension_is_missing'] = data.dimension_is_missing
  gp, mutables = model.apply(
      {'params': params},
      data.features,
      **kwargs,
      mutable=['losses', 'predictive'],
  )
  loss = -gp.log_prob(
      data.labels,
      is_missing=data.label_is_missing,
  ) + jax.tree_util.tree_reduce(jnp.add, mutables['losses'])
  if normalize:
    loss /= data.labels.shape[0]
  return loss, dict()


def stochastic_process_model_setup(
    key: jax.random.KeyArray,
    model: sp.StochasticProcessModel,
    data: types.StochasticProcessModelData,
):
  """Setup function for a stochastic process model."""
  kwargs = {}
  if data.dimension_is_missing is not None:
    kwargs['dimension_is_missing'] = data.dimension_is_missing
  return model.init(key, data.features, **kwargs)['params']
