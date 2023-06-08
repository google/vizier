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

from absl.testing import parameterized
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
        cls(jw.LbfgsBOptions(random_restarts=20)),
        constraints=constraints,
        threshold=1.0,
    )


if __name__ == '__main__':
  absltest.main()
