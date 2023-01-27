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

"""Tests for tuned_gp_models."""

from absl import logging
import jax
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax.models import tuned_gp_models
from vizier._src.jax.optimizers import optimizers

from absl.testing import absltest

tfb = tfp.bijectors


class VizierGpTest(absltest.TestCase):

  def test_good_log_likelihood(self):
    x_obs = np.array([[
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
                     dtype=np.float64)
    y_obs = np.array([
        0.55552674, -0.29054829, -0.04703586, 0.0217839, 0.15445438, 0.46654119,
        0.12255823, -0.19540335, -0.11772564, -0.44447326
    ],
                     dtype=np.float64)

    target_loss = -0.2
    model, loss_fn = tuned_gp_models.VizierGaussianProcess.model_and_loss_fn(
        x_obs, y_obs
    )
    setup = lambda rng: model.init(rng, x_obs)['params']

    optimize = optimizers.JaxoptLbfgsB(random_restarts=50)
    constraints = sp.get_constraints(model.coroutine)
    optimal_params, metrics = optimize(
        setup, loss_fn, jax.random.PRNGKey(2), constraints=constraints
    )
    logging.info('Optimal: %s', optimal_params)
    logging.info('Loss: %s', loss_fn(optimal_params)[0])
    self.assertLess(metrics['loss'].min(), target_loss)


if __name__ == '__main__':
  absltest.main()
