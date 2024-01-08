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

"""Random vectorized optimizer to be used in convergence test."""

from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import types
from vizier.pyvizier import converters

tfd = tfp.distributions


class RandomVectorizedStrategy(vb.VectorizedStrategy[None]):
  """Random vectorized strategy."""

  def __init__(
      self,
      converter: converters.TrialToModelInputConverter,
      suggestion_batch_size: int,
  ):
    empty_features = converter.to_features([])
    n_feature_dimensions_with_padding = types.ContinuousAndCategorical(
        empty_features.continuous.shape[-1],
        empty_features.categorical.shape[-1],
    )

    categorical_sizes = []
    for spec in converter.output_specs.categorical:
      categorical_sizes.append(spec.bounds[1])

    self._suggestion_batch_size = suggestion_batch_size
    self.n_feature_dimensions_with_padding = n_feature_dimensions_with_padding
    self.n_feature_dimensions = n_feature_dimensions_with_padding
    self.dtype = types.ContinuousAndCategorical(jnp.float64, types.INT_DTYPE)

    self._categorical_logits = None
    if categorical_sizes:
      categorical_logits = np.zeros(
          [len(categorical_sizes), max(categorical_sizes)]
      )
      for i, s in enumerate(categorical_sizes):
        categorical_logits[i, s:] = -np.inf
      self._categorical_logits = categorical_logits

  def init_state(
      self,
      seed: jax.Array,
      n_parallel: int = 1,
      *,
      prior_features: Optional[vb.VectorizedOptimizerInput] = None,
      prior_rewards: Optional[types.Array] = None,
  ) -> None:
    del seed
    return

  def suggest(
      self,
      seed: jax.Array,
      state: None,
      n_parallel: int = 1,
  ) -> vb.VectorizedOptimizerInput:
    del state
    cont_seed, cat_seed = jax.random.split(seed)
    cont_data = jax.random.uniform(
        cont_seed,
        shape=(
            self._suggestion_batch_size,
            n_parallel,
            self.n_feature_dimensions_with_padding.continuous,
        ),
    )
    if self._categorical_logits is None:
      cat_data = jnp.zeros(
          [self._suggestion_batch_size, n_parallel, 0], dtype=jnp.int32
      )
    else:
      cat_data = tfd.Categorical(logits=self._categorical_logits).sample(
          (self._suggestion_batch_size, n_parallel), seed=cat_seed
      )
    return vb.VectorizedOptimizerInput(cont_data, cat_data)

  def suggestion_batch_size(self) -> int:
    return self._suggestion_batch_size

  def update(
      self,
      seed: jax.Array,
      state: None,
      batch_features: vb.VectorizedOptimizerInput,
      batch_rewards: types.Array,
  ) -> None:
    return


def random_strategy_factory(
    converter: converters.TrialToModelInputConverter,
    suggestion_batch_size: int,
) -> vb.VectorizedStrategy:
  """Creates a new vectorized strategy based on the Protocol."""
  return RandomVectorizedStrategy(
      converter=converter,
      suggestion_batch_size=suggestion_batch_size,
  )


def create_random_optimizer(
    converter: converters.TrialToModelInputConverter,
    max_evaluations: int,
    suggestion_batch_size: int,
) -> vb.VectorizedOptimizer:
  """Creates a random optimizer."""
  return vb.VectorizedOptimizerFactory(
      strategy_factory=random_strategy_factory,
      max_evaluations=max_evaluations,
      suggestion_batch_size=suggestion_batch_size,
  )(converter=converter)


def create_random_optimizer_factory(
    max_evaluations: int, suggestion_batch_size: int
) -> vb.VectorizedOptimizerFactory:
  """Creates a random optimizer factory."""
  return vb.VectorizedOptimizerFactory(
      strategy_factory=random_strategy_factory,
      max_evaluations=max_evaluations,
      suggestion_batch_size=suggestion_batch_size,
  )
