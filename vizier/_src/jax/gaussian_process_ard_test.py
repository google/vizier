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

"""Tests for gaussian_process_ard."""

from jax import random
import numpy as np

from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax import gaussian_process_ard as gp_ard
from absl.testing import absltest

tfpk = tfp.math.psd_kernels


class GaussianProcessARDTest(absltest.TestCase):

  def test_gaussian_process_ard(self):
    dim = 5
    num_obs = 10
    coro = gp_ard.GaussianProcessARD(
        dimension=dim,
        kernel_class=tfpk.ExponentiatedQuadratic,
        use_tfp_runtime_validation=True)

    obs_key, coro_key, sample_key = random.split(random.PRNGKey(0), num=3)
    x = random.uniform(obs_key, shape=(num_obs, dim), dtype=np.float32)
    gp, param_vals = _run_coroutine(coro(x), seed=coro_key)
    samples = gp.sample(100, seed=sample_key)
    self.assertSequenceEqual(gp.event_shape, [num_obs])
    self.assertEmpty(gp.batch_shape)
    self.assertTrue(np.isfinite(gp.log_prob(samples)).all())
    self.assertSameElements(
        param_vals.keys(),
        ('amplitude', 'inverse_length_scale', 'observation_noise_variance'))
    self.assertEmpty(param_vals['amplitude'].shape)
    self.assertEmpty(param_vals['observation_noise_variance'].shape)
    self.assertSequenceEqual(param_vals['inverse_length_scale'].shape, [dim])


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
  absltest.main()
