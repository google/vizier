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

  def _generate_xys(self):
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
    return x_obs, y_obs

  def test_masking_works(self):
    # Mask three dimensions and four observations.
    observation_is_missing = np.array(
        [False, False, True, False, True, False, True, True, False, False]
    )
    dimension_is_missing = np.array([False, True, True, False, True, False])
    x_obs, y_obs = self._generate_xys()

    # Change these to nans to ensure that they are ignored.
    modified_x_obs = np.where(dimension_is_missing, np.nan, x_obs)
    modified_y_obs = np.where(observation_is_missing, np.nan, y_obs)

    model1, loss_fn1 = tuned_gp_models.VizierGaussianProcess.model_and_loss_fn(
        x_obs,
        y_obs,
        observation_is_missing=observation_is_missing,
        dimension_is_missing=dimension_is_missing,
    )

    model2, loss_fn2 = tuned_gp_models.VizierGaussianProcess.model_and_loss_fn(
        modified_x_obs,
        modified_y_obs,
        observation_is_missing=observation_is_missing,
        dimension_is_missing=dimension_is_missing,
    )

    # Check that the model loss and optimal parameters are independent of those
    # dimensions and observations.
    optimize = optimizers.JaxoptLbfgsB(random_restarts=1)
    optimal_params1, _ = optimize(
        lambda rng: model1.init(rng, x_obs)['params'],
        loss_fn1,
        jax.random.PRNGKey(2),
        constraints=sp.get_constraints(model1),
    )
    optimal_params2, _ = optimize(
        lambda rng: model2.init(rng, modified_x_obs)['params'],
        loss_fn2,
        jax.random.PRNGKey(2),
        constraints=sp.get_constraints(model2),
    )

    for key in optimal_params1:
      self.assertTrue(
          np.all(np.equal(optimal_params1[key], optimal_params2[key])),
          msg=f'{key} parameters were not equal.',
      )
    self.assertEqual(loss_fn1(optimal_params1)[0], loss_fn2(optimal_params2)[0])

  def test_good_log_likelihood(self):
    x_obs, y_obs = self._generate_xys()
    target_loss = -0.2
    model, loss_fn = tuned_gp_models.VizierGaussianProcess.model_and_loss_fn(
        x_obs, y_obs
    )
    setup = lambda rng: model.init(rng, x_obs)['params']

    optimize = optimizers.JaxoptLbfgsB(random_restarts=50)
    constraints = sp.get_constraints(model)
    optimal_params, metrics = optimize(
        setup, loss_fn, jax.random.PRNGKey(2), constraints=constraints
    )
    logging.info('Optimal: %s', optimal_params)
    logging.info('Loss: %s', loss_fn(optimal_params)[0])
    self.assertLess(metrics['loss'].min(), target_loss)

  def test_good_log_likelihood_with_masks(self):
    x_obs, y_obs = self._generate_xys()
    # Pad x_s and y_s and generate masks.
    x_obs = np.pad(x_obs, ((0, 0), (0, 3)), constant_values=np.nan)
    x_obs = np.pad(x_obs, ((0, 2), (0, 0)), constant_values=np.nan)
    y_obs = np.pad(y_obs, (0, 2), constant_values=-np.inf)

    observation_is_missing = np.array(
        [False] * (y_obs.shape[0] - 2) + [True] * 2
    )
    dimension_is_missing = np.array(
        [False] * (x_obs.shape[-1] - 3) + [True] * 3
    )

    target_loss = -0.2
    model, loss_fn = tuned_gp_models.VizierGaussianProcess.model_and_loss_fn(
        x_obs,
        y_obs,
        observation_is_missing=observation_is_missing,
        dimension_is_missing=dimension_is_missing,
    )
    setup = lambda rng: model.init(rng, x_obs)['params']

    optimize = optimizers.JaxoptLbfgsB(random_restarts=50)
    constraints = sp.get_constraints(model)
    optimal_params, metrics = optimize(
        setup, loss_fn, jax.random.PRNGKey(2), constraints=constraints
    )
    logging.info('Optimal: %s', optimal_params)
    logging.info('Loss: %s', loss_fn(optimal_params)[0])
    self.assertLess(metrics['loss'].min(), target_loss)


if __name__ == '__main__':
  absltest.main()
