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

import jax
from jax import numpy as jnp
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types


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
