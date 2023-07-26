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

from typing import Callable, Optional

import attr
import jax
import jax.numpy as jnp
import numpy as np
from vizier import pyvizier as vz
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier.jax import optimizers
from vizier.pyvizier import converters


@attr.define(kw_only=True)
class LBFGSBOptimizer:
  """L-BFGS-B optimizer."""

  # Number of parallel runs of L-BFGS-B.
  random_restarts: int = attr.field(init=True, repr=False, default=25)
  # Number of iterations for each L-BFGS-B run.
  maxiter: int = attr.field(init=True, repr=False, default=50)
  # In parallel optimization (suggesting multiple candidates at once), this is
  # the number of candidates to consider at once. The score function is assumed
  # to score this many candidates together and output a scalar.
  num_parallel_candidates: Optional[int] = attr.field(default=None)

  def optimize(
      self,
      converter: converters.TrialToModelInputConverter,
      # TODO: Update the type to (Parallel)ArrayScoreFunction.
      score_fn: Callable[[jax.Array], jax.Array],
      *,
      count: Optional[int] = None,
      seed: Optional[int] = None,
  ) -> list[vz.Trial]:
    """Optimize a continuous objective function using L-BFGS-B.

    Arguments:
      converter: Converter to map between trials and arrays.
      score_fn: Converts (batches of) features to scores.
      count: The number of best results to return. None means squeezing out the
        dimension.
      seed: Optional seed.

    Returns:
      The best trials found in the optimization.
    """
    if self.num_parallel_candidates and (count or 0) > 1:
      # Note that we can't distinguish between 'BatchArrayScoreFunction' and
      # 'ParallelArrayScoreFunction' using 'isinstance' as they both have the
      # same function names.
      raise ValueError(
          "LBFGSBOptimizer doesn't support batch of batches (count > 1 is"
          " disallowed when num_parallel_candidates is set)."
      )
    if self.num_parallel_candidates:
      count = None
    optimize = optimizers.JaxoptScipyLbfgsB(
        optimizers.LbfgsBOptions(maxiter=self.maxiter)
    )
    num_features = sum(
        spec.num_dimensions for spec in converter.output_specs.continuous
    )

    feature_shape = [num_features]
    if self.num_parallel_candidates is not None:
      feature_shape = [self.num_parallel_candidates, num_features]

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

    rng, init_rng = jax.random.split(rng, 2)
    new_features, _ = optimize(
        jax.vmap(setup)(jax.random.split(init_rng, self.random_restarts)),
        wrapped_score_fn,
        rng,
        constraints=constraints,
        best_n=count,
    )
    new_rewards = np.asarray(score_fn(new_features[jnp.newaxis, ...]))[0]
    if self.num_parallel_candidates is None:
      new_features = new_features[jnp.newaxis, ...]
    parameters = converter.to_parameters(
        types.ModelInput(
            continuous=types.PaddedArray.as_padded(new_features),
            categorical=types.PaddedArray.as_padded(
                jnp.zeros(new_features.shape[:-1] + (0,), dtype=types.INT_DTYPE)
            ),
        )
    )
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
      self, random_restarts: int, num_parallel_candidates: Optional[int] = None
  ) -> LBFGSBOptimizer:
    """Generates a new LBFGSBOptimizer object."""
    return LBFGSBOptimizer(
        num_parallel_candidates=num_parallel_candidates,
        random_restarts=random_restarts,
    )
