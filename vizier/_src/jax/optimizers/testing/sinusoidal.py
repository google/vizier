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

"""Sinusodial function for testing."""

from typing import Optional
import unittest

from absl import logging
import chex
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
import tree
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier._src.jax.optimizers import core
from vizier._src.jax.types import Bounds


tfb = tfp.bijectors


@chex.assert_max_traces(1)
def loss_fn(inputs):
  # Sinusoidal function minimized at all inputs being zero. (min value=-2)
  # There are lots of local optima between [-3, 3]
  xs = jnp.concatenate(jax.tree_util.tree_flatten(inputs)[0], axis=-1)
  xs = jax.nn.sigmoid(xs)
  return (
      jnp.mean(jnp.cos(xs * 5 * 2 * jnp.pi) * (2 - 5 * (xs - 0.5) ** 2)) + 2,
      dict(),
  )


def setup(rng):
  rng1, rng2 = jax.random.split(rng, 2)
  return {
      'x1': jax.random.uniform(rng1, shape=(1,), minval=-3, maxval=3),
      'x2': jax.random.uniform(rng2, shape=(2,), minval=-3.0, maxval=3.0),
  }


def _make_constraints(b: Optional[types.Bounds]) -> Optional[chex.ArrayTree]:
  if b is None:
    return None
  return {'x1': jnp.array([b]), 'x2': jnp.array([b, b])}


def bounds_to_constraints(
    bounds: Optional[Bounds],
    nest_constraint: bool = True,
) -> Optional[sp.Constraint]:
  if bounds is None:
    return None
  if nest_constraint:
    bounds = tree.map_structure(_make_constraints, bounds)
  return sp.Constraint.create(bounds, tfb.SoftClip)


class ConvergenceTestMixin(unittest.TestCase):
  """Mixin for subclasses."""

  def assert_converges(
      self,
      optimize: core.Optimizer,
      *,
      constraints: sp.Constraint,
      threshold: float = 5e-3,
      random_restarts: int,
  ) -> None:
    """Assert the optimizer converges."""
    rngs = jax.random.split(jax.random.PRNGKey(1), random_restarts + 1)
    optimal_params, metrics = optimize(
        init_params=jax.vmap(setup)(rngs[1:]),
        loss_fn=jax.jit(loss_fn),
        rng=rngs[0],
        constraints=constraints,
    )
    logging.info('Optimal: %s', optimal_params)

    self.assertLessEqual(loss_fn(optimal_params)[0], threshold)
    if metrics['loss'].shape[0] > 1:
      np.testing.assert_array_less(
          metrics['loss'][-1, :], metrics['loss'][0, :]
      )

    if (constraints is not None) and (constraints.bounds is not None):
      bounds = constraints.bounds
      for y_, b_ in zip(tree.flatten(optimal_params), tree.flatten(bounds[0])):
        if b_ is not None:
          np.testing.assert_array_less(b_, y_)
      for y_, b_ in zip(tree.flatten(optimal_params), tree.flatten(bounds[1])):
        if b_ is not None:
          np.testing.assert_array_less(y_, b_)
