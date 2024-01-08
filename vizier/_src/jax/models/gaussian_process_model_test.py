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

"""Tests for gaussian_process_model."""

import jax
from jax import random
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import types
from vizier._src.jax.models import gaussian_process_model as gp_model
from absl.testing import absltest

tfpk = tfp.math.psd_kernels
tfpke = tfp.experimental.psd_kernels


class GaussianProcessARDTest(absltest.TestCase):

  def test_gp_model_with_categorical(self):
    cont_dim = 5
    cat_dim = 3
    num_obs = 10
    coro = gp_model.GaussianProcessARD(
        dimension=types.ContinuousAndCategorical[int](cont_dim, cat_dim),
        kernel_class=tfpk.ExponentiatedQuadratic,
        use_tfp_runtime_validation=True,
    )

    x_cont_key, x_cat_key, coro_key, sample_key = random.split(
        random.PRNGKey(0), num=4
    )
    x_cont = random.uniform(x_cont_key, shape=(num_obs, cont_dim),
                            dtype=np.float64)
    x_cat = random.randint(
        x_cat_key, shape=(num_obs, cat_dim), minval=0, maxval=5
    )
    x = types.ModelInput(
        continuous=types.PaddedArray.from_array(
            x_cont, x_cont.shape, fill_value=np.nan
        ),
        categorical=types.PaddedArray.from_array(
            x_cat, x_cat.shape, fill_value=-1
        ),
    )
    gp, param_vals = _run_coroutine(coro(x), seed=coro_key)
    samples = gp.sample(100, seed=sample_key)
    self.assertSequenceEqual(gp.event_shape, [num_obs])
    self.assertEmpty(gp.batch_shape)
    self.assertTrue(np.isfinite(gp.log_prob(samples)).all())
    self.assertSameElements(
        param_vals.keys(),
        (
            'amplitude',
            'inverse_length_scale_continuous',
            'inverse_length_scale_categorical',
            'observation_noise_variance',
        ),
    )
    self.assertEmpty(param_vals['amplitude'].shape)
    self.assertEmpty(param_vals['observation_noise_variance'].shape)
    self.assertSequenceEqual(
        param_vals['inverse_length_scale_continuous'].shape, [cont_dim]
    )
    self.assertSequenceEqual(
        param_vals['inverse_length_scale_categorical'].shape, [cat_dim]
    )


def _run_coroutine(g, seed):
  param = next(g)
  param_vals = {}
  try:
    while True:
      seed, current_seed = random.split(seed)
      v = param.init_fn(current_seed)
      param_vals[param.name] = v
      param = g.send(v)
  except StopIteration as e:
    return e.value, param_vals


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
