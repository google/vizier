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

"""Random vectorized optimizer to be used in convergence test."""

from typing import Optional

import jax
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import types
from vizier.pyvizier import converters


class RandomVectorizedStrategy(vb.VectorizedStrategy):
  """Random vectorized strategy."""

  def __init__(
      self,
      converter: converters.TrialToArrayConverter,
      suggestion_batch_size: int,
  ):
    self._converter = converter
    self._suggestion_batch_size = suggestion_batch_size
    self._n_features = sum(
        [spec.num_dimensions for spec in self._converter.output_specs]
    )

  def init_state(
      self,
      seed: jax.random.KeyArray,
      prior_features: Optional[types.Array] = None,
      prior_rewards: Optional[types.Array] = None,
  ) -> None:
    del seed
    return

  def suggest(self, state: None, seed: jax.random.KeyArray) -> jax.Array:
    del state
    return jax.random.uniform(
        seed,
        shape=(
            self._suggestion_batch_size,
            self._n_features,
        ),
    )

  def suggestion_batch_size(self) -> int:
    return self._suggestion_batch_size

  def update(
      self,
      state: None,
      batch_features: types.Array,
      batch_rewards: types.Array,
      seed: jax.random.KeyArray,
  ) -> None:
    return


def _random_strategy_factory(
    converter: converters.TrialToArrayConverter,
    suggestion_batch_size: int,
) -> vb.VectorizedStrategy:
  """Creates a new vectorized strategy based on the Protocol."""
  return RandomVectorizedStrategy(
      converter=converter,
      suggestion_batch_size=suggestion_batch_size,
  )


def create_random_optimizer(
    max_evaluations: int, suggestion_batch_size: int
) -> vb.VectorizedOptimizer:
  """Creates a random optimizer."""
  return vb.VectorizedOptimizer(
      strategy_factory=_random_strategy_factory,
      max_evaluations=max_evaluations,
      suggestion_batch_size=suggestion_batch_size,
  )


def create_random_optimizer_factory() -> vb.VectorizedOptimizerFactory:
  """Creates a random optimizer factory."""
  return vb.VectorizedOptimizerFactory(
      strategy_factory=_random_strategy_factory
  )
