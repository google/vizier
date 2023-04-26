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

"""L-BFGS-B Strategy optimizer."""

from typing import Optional

import attr
import jax
import jax.numpy as jnp
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import vectorized_base
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax.optimizers import optimizers
from vizier.pyvizier import converters


@attr.define(kw_only=True)
class LBFGSBOptimizer:
  """L-BFGS-B optimizer."""

  # Number of parallel runs of L-BFGS-B.
  random_restarts: int = attr.field(init=True, repr=False, default=25)
  # Number of features to consider at a time. The score function is assumed to
  # score this many features at a time.
  parallel_batch_size: Optional[int] = attr.field(default=None)

  def optimize(
      self,
      converter: converters.TrialToArrayConverter,
      score_fn: vectorized_base.BatchArrayScoreFunction,
      *,
      count: int = 1,
      seed: Optional[int] = None,
  ) -> list[vz.Trial]:
    """Optimize a continuous objective function using L-BFGS-B.

    Arguments:
      converter: Converter to map between trials and arrays.
      score_fn: `BatchArrayScoreFunction`. Converts (batches of) features to
        scores.
      count: The number of best results to store.
      seed: Optional seed.

    Returns:
      The best trials found in the optimization.
    """
    optimize = optimizers.JaxoptLbfgsB(
        random_restarts=self.random_restarts, best_n=count
    )
    num_features = sum(spec.num_dimensions for spec in converter.output_specs)

    feature_shape = [num_features]
    if self.parallel_batch_size is not None:
      feature_shape = [self.parallel_batch_size, num_features]

    def setup(rng):
      return jax.random.uniform(rng, shape=feature_shape)

    rng = jax.random.PRNGKey(seed or 0)

    # Constraints are [0, 1].
    constraints = sp.Constraint(
        bounds=(np.zeros(feature_shape), np.ones(feature_shape))
    )

    # We'll assume score_fn is differentiable.
    # We also wrap the score function because the convention for optimizers is
    # to maximize, while L-BFGS-B always minimizes.
    def wrapped_score_fn(x):
      # Add a batch axis since `score_fn` is assumed to work on a batch of
      # trials.
      return -score_fn(x[jnp.newaxis, ...])[0], dict()

    new_features, _ = optimize(
        setup, wrapped_score_fn, rng, constraints=constraints
    )
    new_rewards = np.asarray(score_fn(new_features[jnp.newaxis, ...]))[0]

    if self.parallel_batch_size is None:
      parameters = converter.to_parameters(new_features[jnp.newaxis, ...])
    else:
      parameters = converter.to_parameters(new_features)
    trials = []
    for i in range(len(parameters)):
      trial = vz.Trial(parameters=parameters[i])
      trial.complete(vz.Measurement({'acquisition': new_rewards}))
      trials.append(trial)
    return trials


@attr.define
class LBFGSBOptimizerFactory:
  """LBFGSB strategy optimizer factory."""

  def __call__(
      self, random_restarts: int, parallel_batch_size: Optional[int] = None
  ) -> LBFGSBOptimizer:
    """Generates a new LBFGSBOptimizer object."""
    return LBFGSBOptimizer(
        parallel_batch_size=parallel_batch_size, random_restarts=random_restarts
    )
