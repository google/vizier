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

"""High-level wrappers for stochastic process hyperparameter optimizers."""

import functools
from typing import Generic, Optional, Protocol, TypeVar

from absl import logging
import chex
import jax
from jax import numpy as jnp
from vizier._src.jax import stochastic_process_model as sp


Params = TypeVar('Params', bound=chex.ArrayTree)


class LossFunction(Protocol, Generic[Params]):
  """Evaluates model params and returns (loss, dict of auxiliary metrics)."""

  def __call__(self, params: Params) -> tuple[jax.Array, chex.ArrayTree]:
    """Evaluates model params and returns (loss, dict of auxiliary metrics)."""
    pass


class LossAndGradFunction(Protocol, Generic[Params]):
  """Computes loss and gradient."""

  def __call__(
      self, params: Params
  ) -> tuple[tuple[jax.Array, chex.ArrayTree], jax.Array]:
    """Returns (loss, tree of auxiliary metrics), grad."""


class Optimizer(Protocol[Params]):
  """Optimizes the LossFunction.

  Example:

  ```python
  setup: Setup = lambda rng: jax.random.uniform(rng, minval=-5, maxval=5)

  def loss_fn(xs):  # satisfies `LossFunction` Protocol
    xs = jax.nn.sigmoid(xs)
    return jnp.cos(xs * 5 * 2 * jnp.pi) * (2 - 5 * (xs - 0.5)**2), dict()

  optimize = optimizers.OptaxTrain(optax.adam(5e-3), epochs=500, verbose=True)
  rngs = jax.random.split(jax.random.PRNGKey(0), 51)
  optimal_params, metrics = optimize(
      init_params=jax.vmap(setup)(rngs[1:]),
      loss_fn=loss_fn,
      rng=rngs[0])
  ```
  """

  def __call__(
      self,
      init_params: Params,
      loss_fn: LossFunction[Params],
      rng: jax.Array,
      *,
      constraints: Optional[sp.Constraint] = None,
      best_n: Optional[int] = None,
  ) -> tuple[Params, chex.ArrayTree]:
    """Optimizes a LossFunction expecting Params as input.

    When constraint bijectors are applied, note that the returned parameters are
    in the constrained space (the parameter domain), not the unconstrained space
    over which the optimization takes place.

    If the Optimizer uses lower and upper bounds, then it is responsible for
    converting `None` bounds to `+inf` or `-inf` as necessary.

    Args:
      init_params: A set of initial points represented as a batched Pytree. All
        leaf nodes are expected to have the same leading dimension, which is the
        batch size.
      loss_fn: Evaluates a point.
      rng: JAX PRNGKey.
      constraints: Parameter constraints.
      best_n: If not None, returns a pytree with a batch dimension [best_n].

    Returns:
      Tuple containing optimal input in the constrained space and optimization
      metrics.
    """


def get_best_params(
    losses: jax.Array,
    all_params: chex.ArrayTree,
    *,
    best_n: Optional[int] = None,
) -> chex.ArrayTree:
  """Returns the top `best_n` parameters that minimize the losses.

  Args:
    losses: Shape (N,) array
    all_params: ArrayTree whose leaves have shape (N, ...)
    best_n: Integer greater than or equal to 1. If None, squeezes the leading
      dimension.

  Returns:
    Top `best_n` parameters.
  """
  argsorted = jnp.argsort(losses)
  if not best_n:
    best_idx = argsorted[:1]
  else:
    best_idx = argsorted[:best_n]

  logging.info('Best loss(es): %s at retry %s', losses[best_idx], best_idx)
  optimal_params = jax.tree_util.tree_map(lambda p: p[best_idx], all_params)
  if best_n is None:
    optimal_params = jax.tree.map(
        functools.partial(jnp.squeeze, axis=0), optimal_params
    )
  return optimal_params
