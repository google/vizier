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

"""Tests for hebo_gp_model."""

from absl import logging
import jax
from jax import numpy as jnp
import optax
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax.models import hebo_gp_model
from vizier._src.jax.optimizers import optimizers

from absl.testing import absltest

VizierHeboGaussianProcess = hebo_gp_model.VizierHeboGaussianProcess


class VizierHeboGaussianProcessTest(absltest.TestCase):

  def setUp(self):
    super(VizierHeboGaussianProcessTest, self).setUp()
    self.x_obs = jnp.array([[
        0.2941264, 0.29313548, 0.68817519, 0.37502566, 0.48356813, 0.34127283
    ], [
        0.66218224, 0.70770083, 0.6901334, 0.66787973, 0.5400858, 0.52721233
    ], [
        0.88469647, 0.50593371, 0.83160862, 0.58674892, 0.42145673, 0.31749428
    ], [
        0.39976682, 0.59517741, 0.73295106, 0.6084903, 0.54891015, 0.44338632
    ], [
        0.8354305, 0.87605574, 0.47855956, 0.48174861, 0.37685449, 0.38348768
    ], [
        0.55608455, 0.72781129, 0.52432913, 0.44291417, 0.3816395, 0.326599
    ], [
        0.24689187, 0.50979672, 0.67604857, 0.45172594, 0.34994392, 0.75239792
    ], [
        0.71007257, 0.60896354, 0.29270877, 0.74683367, 0.50169051, 0.74480515
    ], [
        0.9193235, 0.24393112, 0.63868591, 0.43271524, 0.43339578, 0.59413154
    ], [0.51850627, 0.62689204, 0.76134879, 0.65990021, 0.82350868, 0.7429215]],
                           dtype=jnp.float64)

    self.y_obs = jnp.array([
        0.55552674, -0.29054829, -0.04703586, 0.0217839, 0.15445438, 0.46654119,
        0.12255823, -0.19540335, -0.11772564, -0.44447326
    ],
                           dtype=jnp.float64)

  def test_log_prob_and_loss(self):
    model, loss_fn = VizierHeboGaussianProcess.model_and_loss_fn(
        features=self.x_obs, labels=self.y_obs)
    setup = lambda rng: model.init(rng, self.x_obs)['params']
    key = jax.random.PRNGKey(2)
    init_params = setup(key)
    optimize = optimizers.OptaxTrainWithRandomRestarts(
        optax.adam(5e-3), epochs=500, verbose=True, random_restarts=20
    )
    constraints = sp.get_constraints(model.coroutine)
    params, metrics = optimize(setup, loss_fn, key, constraints=constraints)

    init_gp = model.apply({'params': init_params}, self.x_obs)
    gp = model.apply({'params': params}, self.x_obs)
    self.assertGreater(gp.log_prob(self.y_obs), init_gp.log_prob(self.y_obs))
    losses_every_50 = metrics['loss'][::50]
    self.assertTrue((losses_every_50[1:] < losses_every_50[:-1]).all())

    logging.info('Optimal parameters: %s', params)
    logging.info('Optimal loss fn: %s', loss_fn(params)[0])
    self.assertLess(loss_fn(params)[0], 0.3)


if __name__ == '__main__':
  absltest.main()
