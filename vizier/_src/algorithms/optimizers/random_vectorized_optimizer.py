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

import numpy as np
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier.pyvizier import converters


class RandomVectorizedStrategy(vb.VectorizedStrategy):
  """Random vectorized strategy."""

  def __init__(
      self,
      converter: converters.TrialToArrayConverter,
      suggestion_batch_size: int,
      seed: Optional[int] = None,
      prior_features: Optional[np.ndarray] = None,
      prior_rewards: Optional[np.ndarray] = None,
  ):
    self._converter = converter
    self._suggestion_batch_size = suggestion_batch_size
    self._rng = np.random.default_rng(seed)
    self._n_features = sum(
        [spec.num_dimensions for spec in self._converter.output_specs]
    )

  def suggest(self) -> vb.Array:
    return self._rng.uniform(
        low=0,
        high=1,
        size=(
            self._suggestion_batch_size,
            self._n_features,
        ),
    )

  def suggestion_batch_size(self) -> int:
    return self._suggestion_batch_size

  def update(self, rewards: vb.Array) -> None:
    pass


def _random_strategy_factory(
    converter: converters.TrialToArrayConverter,
    suggestion_batch_size: int,
    seed: Optional[int] = None,
    prior_features: Optional[np.ndarray] = None,
    prior_rewards: Optional[np.ndarray] = None,
) -> vb.VectorizedStrategy:
  """Creates a new vectorized strategy based on the Protocol."""
  return RandomVectorizedStrategy(
      converter=converter,
      suggestion_batch_size=suggestion_batch_size,
      seed=seed,
      prior_features=prior_features,
      prior_rewards=prior_rewards,
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
