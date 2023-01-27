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

"""Tests for optimizers."""
from absl import logging
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import optax
from tensorflow_probability.substrates import jax as tfp
import tree
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax.optimizers import optimizers

from absl.testing import absltest

tfb = tfp.bijectors


def loss_fn(inputs):
  # Sinusoidal function minimized at all inputs being zero. (min value=-2)
  # There are lots of local optima between [-3, 3]
  xs = jnp.concatenate(jax.tree_util.tree_flatten(inputs)[0], axis=-1)
  xs = jax.nn.sigmoid(xs)
  return jnp.mean(jnp.cos(xs * 5 * 2 * jnp.pi) * (2 - 5 *
                                                  (xs - 0.5)**2)), dict()


def optimizer_setup(rng):
  rng1, rng2 = jax.random.split(rng, 2)
  return {
      'x1': jax.random.uniform(rng1, shape=(1,), minval=-3, maxval=3),
      'x2': jax.random.uniform(rng2, shape=(2,), minval=-3., maxval=3.)
  }


def _make_constraint_array(b):
  if b is None:
    return None
  return {'x1': jnp.array([b]), 'x2': jnp.array([b, b])}


class OptimizersTest(parameterized.TestCase):

  @parameterized.parameters(
      (None,),
      ((-4.0, None),),
      ((None, 5.0),),
      ((-3.0, 3.0),),
  )
  def test_sinusoidal(self, bounds):
    optimize = optimizers.OptaxTrainWithRandomRestarts(
        optax.adam(5e-2), epochs=100, verbose=True, random_restarts=200)
    if bounds is None:
      constraints = None
    else:
      lb, ub = jax.tree_util.tree_map(_make_constraint_array, bounds)
      constraints = sp.Constraint.create((lb, ub), tfb.SoftClip)

    optimal_params, metrics = optimize(
        optimizer_setup, loss_fn, jax.random.PRNGKey(1), constraints=constraints
    )
    logging.info('Optimal: %s', optimal_params)
    # Thanks to restarts, we can find the optimum.
    self.assertLessEqual(jnp.abs(loss_fn(optimal_params)[0] + 2), 5e-3)
    self.assertTrue((metrics['loss'][-1] < metrics['loss'][0]).all())
    if bounds is not None:
      for y_, b_ in zip(tree.flatten(optimal_params), tree.flatten(lb)):
        if b_ is not None:
          self.assertTrue((y_ > b_).all())
      for y_, b_ in zip(tree.flatten(optimal_params), tree.flatten(ub)):
        if b_ is not None:
          self.assertTrue((y_ < b_).all())

  def test_sinusodial_bestn_optax(self):
    optimize = optimizers.OptaxTrainWithRandomRestarts(
        optax.adam(5e-3),
        epochs=100,
        verbose=True,
        random_restarts=100,
        best_n=5)
    constraint_fn = tfb.JointMap({'x1': tfb.Exp(), 'x2': tfb.Softplus()})
    optimal_params, _ = optimize(
        optimizer_setup,
        loss_fn,
        jax.random.PRNGKey(0),
        # Optax uses the bijector and not the bounds, so it is safe to pass only
        # the bijector.
        constraints=sp.Constraint(bijector=constraint_fn),
    )
    logging.info('Optimal: %s', optimal_params)

    self.assertSequenceEqual(optimal_params['x2'].shape, (5, 2))
    self.assertSequenceEqual(optimal_params['x1'].shape, (5, 1))

  @absltest.skip("Test breaks externally due to JaxOpt.")
  @parameterized.parameters(
      (None,),
      ((-4.0, None),),
      ((None, 5.0), False),
      ((-3.0, 3.0),),
  )
  def test_sinusodial_bestn_l_bfgs_b(self, bounds, nest_constraint=True):
    if bounds is None:
      constraints = None
    else:
      if nest_constraint:
        bounds = jax.tree_util.tree_map(_make_constraint_array, bounds)
      constraints = sp.Constraint.create(bounds, tfb.SoftClip)
    optimize = optimizers.JaxoptLbfgsB(random_restarts=10, best_n=5)
    optimal_params, _ = optimize(
        optimizer_setup, loss_fn, jax.random.PRNGKey(0), constraints=constraints
    )
    logging.info('Optimal: %s', optimal_params)

    self.assertSequenceEqual(optimal_params['x2'].shape, (5, 2))
    self.assertSequenceEqual(optimal_params['x1'].shape, (5, 1))
    if bounds is not None:
      for y_, b_ in zip(tree.flatten(optimal_params), tree.flatten(bounds[0])):
        if b_ is not None:
          self.assertTrue((y_ > b_).all())
      for y_, b_ in zip(tree.flatten(optimal_params), tree.flatten(bounds[1])):
        if b_ is not None:
          self.assertTrue((y_ < b_).all())


if __name__ == '__main__':
  absltest.main()
