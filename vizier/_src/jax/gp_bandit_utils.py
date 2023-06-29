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
    data: types.ModelData,
    normalize: bool = False,
):
  """Loss function for a stochastic process model."""
  gp, mutables = model.apply(
      {'params': params},
      data.features,
      mutable=['losses', 'predictive'],
  )
  loss = -gp.log_prob(
      data.labels.padded_array,
      is_missing=data.labels.is_missing[0],
  ) + jax.tree_util.tree_reduce(jnp.add, mutables['losses'])
  if normalize:
    loss /= data.labels.original_shape[0]
  return loss, dict()


def stochastic_process_model_setup(
    key: jax.random.KeyArray,
    model: sp.StochasticProcessModel,
    data: types.ModelData,
):
  """Setup function for a stochastic process model."""
  return model.init(key, data.features)['params']


# TODO: Remove this when Vectorized Optimizer works on CACV.
def make_one_hot_to_modelinput_fn(seed_features_unpad, mapper, cacpa):
  """Temporary utility fn for converting one hot to ModelInput."""

  def _one_hot_to_cacpa(x_):
    if seed_features_unpad is not None:
      x_unpad = x_[..., : seed_features_unpad.shape[1]]
    else:
      x_unpad = x_
    cacv = mapper.map(x_unpad)
    return types.ModelInput(
        continuous=types.PaddedArray.from_array(
            cacv.continuous,
            (
                cacv.continuous.shape[:-1]
                + (cacpa.continuous.padded_array.shape[1],)
            ),
            fill_value=cacpa.continuous.fill_value,
        ),
        categorical=types.PaddedArray.from_array(
            cacv.categorical,
            (
                cacv.categorical.shape[:-1]
                + (cacpa.categorical.padded_array.shape[1],)
            ),
            fill_value=cacpa.categorical.fill_value,
        ),
    )

  return _one_hot_to_cacpa
