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

"""L-BFGS-B Strategy optimizer."""

from typing import Callable, Optional, Union

import attr
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier.jax import optimizers
from vizier.pyvizier import converters


def _optimizer_to_model_input(
    x: jax.Array,
    n_features: jax.Array,
) -> types.ModelInput:
  x_cat = jnp.zeros(x.shape[:-1] + (0,), dtype=types.INT_DTYPE)
  return types.ModelInput(
      continuous=vb.optimizer_to_model_input_single_array(x, n_features),
      categorical=vb.optimizer_to_model_input_single_array(
          x_cat, jnp.zeros([])
      ),
  )


@struct.dataclass
class LBFGSBOptimizer:
  """L-BFGS-B optimizer."""

  n_feature_dimensions: types.ContinuousAndCategorical[jax.Array]
  n_feature_dimensions_with_padding: types.ContinuousAndCategorical[int] = (
      struct.field(pytree_node=False)
  )
  # Number of parallel runs of L-BFGS-B.
  random_restarts: int = struct.field(pytree_node=False, default=25)
  # Number of iterations for each L-BFGS-B run.
  maxiter: int = struct.field(pytree_node=False, default=50)

  def __post_init__(self):
    if self.n_feature_dimensions_with_padding.categorical > 0:
      raise ValueError("LBFGSBOptimizer doesn't support categorical features.")

  # TODO: Remove score_fn argument.
  # pylint: disable=g-bare-generic
  def __call__(
      self,
      score_fn: Union[vb.ArrayScoreFunction, vb.ParallelArrayScoreFunction],
      *,
      score_with_aux_fn: Optional[Callable] = None,
      count: Optional[int] = 1,
      prior_features: Optional[types.ModelInput] = None,
      n_parallel: Optional[int] = None,
      seed: Optional[int] = None,
  ) -> vb.VectorizedStrategyResults:
    """Optimize a continuous objective function using L-BFGS-B.

    Arguments:
      score_fn: A callback that expects 2D Array with dimensions (batch_size,
        features_count) or a 3D array with dimensions (batch_size, n_parallel,
        features_count), and a random seed, and returns a 1D Array
        (batch_size,).
      score_with_aux_fn: A callback similar to score_fn but additionally returns
        an array tree.
      count: The number of suggestions to generate.
      prior_features: Completed trials to be used for knowledge transfer.
        Currently ignored.
      n_parallel: This arg should be specified if a parallel acquisition
        function (e.g. qEI, qUCB) is used, which evaluates a set of points of
        this size to a single value. Pass `None` when optimizing acquisition
        functions that evaluate a single point. (Note that `num_parallel=1` and
        `num_parallel=None` will each return a single suggestion, though they
        are different in that `num_parallel=1` indicates that the acquisition
        function consumes the two rightmost dimensions of the input while
        `num_parallel=None` indicates that only the rightmost dimension is
        consumed.)
      seed: The seed to use in the random generator.

    Returns:
      The best trials found in the optimization.
    """
    del prior_features
    seed = jax.random.PRNGKey(0) if seed is None else seed
    if n_parallel and (count or 0) > 1:
      # Note that we can't distinguish between 'BatchArrayScoreFunction' and
      # 'ParallelArrayScoreFunction' using 'isinstance' as they both have the
      # same function names.
      raise ValueError(
          "LBFGSBOptimizer doesn't support batch of batches (count > 1 is"
          " disallowed when num_parallel_candidates is set)."
      )

    parallel_dim = n_parallel or 1
    optimize = optimizers.JaxoptScipyLbfgsB(
        optimizers.LbfgsBOptions(maxiter=self.maxiter)
    )

    score_rng, init_seed, optim_seed = jax.random.split(seed, num=3)

    score_with_aux = score_with_aux_fn
    if score_with_aux is None:
      score_with_aux = lambda *args: (score_fn(*args), dict())

    # We'll assume score_fn is differentiable and use the same acquisition
    # function seed at every call.
    # We also wrap the score function because the convention for optimizers is
    # to maximize, while L-BFGS-B always minimizes.
    def _opt_score_fn(x):
      score, aux = score_with_aux(
          _optimizer_to_model_input(
              x,
              self.n_feature_dimensions.continuous,
          ),
          score_rng,
      )
      if n_parallel is None:
        score = jnp.squeeze(score, axis=-1)
      return -score, aux

    feature_shape = (
        parallel_dim,
        self.n_feature_dimensions_with_padding.continuous,
    )

    def setup(rng):
      return jax.random.uniform(rng, shape=feature_shape)

    # Constraints are [0, 1].
    constraints = sp.Constraint(
        bounds=(np.zeros(feature_shape), np.ones(feature_shape))
    )

    new_features, _ = optimize(
        jax.vmap(setup)(jax.random.split(init_seed, self.random_restarts)),
        _opt_score_fn,
        optim_seed,
        constraints=constraints,
        best_n=count,
    )
    loss_val, aux = _opt_score_fn(new_features)
    new_rewards = -loss_val

    dimension_is_missing = (
        jnp.arange(self.n_feature_dimensions_with_padding.continuous)
        >= self.n_feature_dimensions.continuous
    )
    new_features_model_input = vb.VectorizedOptimizerInput(
        continuous=jnp.where(
            dimension_is_missing, jnp.zeros_like(new_features), new_features
        ),
        categorical=jnp.zeros(
            new_features.shape[:-1] + (0,), dtype=types.INT_DTYPE
        ),
    )
    return vb.VectorizedStrategyResults(
        new_features_model_input,
        new_rewards,
        aux,
    )


@attr.define
class LBFGSBOptimizerFactory:
  """LBFGSB strategy optimizer factory."""

  random_restarts: int = 25
  maxiter: int = 100

  def __call__(
      self,
      converter: converters.TrialToModelInputConverter,
  ) -> LBFGSBOptimizer:
    """Generates a new LBFGSBOptimizer object."""
    n_feature_dimensions = types.ContinuousAndCategorical(
        jnp.array(len(converter.output_specs.continuous)),
        jnp.array(len(converter.output_specs.categorical)),
    )
    empty_features = converter.to_features([])
    n_feature_dimensions_with_padding = types.ContinuousAndCategorical[int](
        empty_features.continuous.shape[-1],
        empty_features.categorical.shape[-1],
    )
    return LBFGSBOptimizer(
        n_feature_dimensions=n_feature_dimensions,
        n_feature_dimensions_with_padding=n_feature_dimensions_with_padding,
        random_restarts=self.random_restarts,
        maxiter=self.maxiter,
    )
