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

"""Tests for optimizers."""

import datetime

from absl.testing import parameterized
import jax
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax.optimizers import jaxopt_wrappers as jw
from vizier._src.jax.optimizers.testing import sinusoidal

from absl.testing import absltest

tfb = tfp.bijectors


class JaxoptWrappersTest(
    parameterized.TestCase, sinusoidal.ConvergenceTestMixin
):

  @parameterized.product(
      (
          dict(bounds=None, nest_constraint=True),
          dict(bounds=((-4.0, None)), nest_constraint=True),
          dict(bounds=((None, 5.0)), nest_constraint=False),
          dict(bounds=((-3.0, 3.0)), nest_constraint=True),
      ),
      cls=(jw.JaxoptScipyLbfgsB, jw.JaxoptLbfgsB),
  )
  def test_sinusoidal(self, cls, bounds, nest_constraint):
    constraints = sinusoidal.bounds_to_constraints(bounds, nest_constraint)
    self.assert_converges(
        cls(jw.LbfgsBOptions()),
        constraints=constraints,
        threshold=1.0,
        random_restarts=20,
    )

  def test_max_duration(self):
    optimizer = jw.JaxoptScipyLbfgsB(
        max_duration=datetime.timedelta(seconds=0),
        speed_test=True,
    )
    random_restarts = 3
    rngs = jax.random.split(jax.random.PRNGKey(1), random_restarts + 1)
    _, metrics = optimizer(
        init_params=jax.vmap(sinusoidal.setup)(rngs[1:]),
        loss_fn=jax.jit(sinusoidal.loss_fn),
        rng=rngs[0],
    )
    # Check that there's only one training instead of 3.
    self.assertLen(metrics['train_time'], 1)


if __name__ == '__main__':
  absltest.main()
