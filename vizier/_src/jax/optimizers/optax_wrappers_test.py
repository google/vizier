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

from absl.testing import parameterized
import optax
from tensorflow_probability.substrates import jax as tfp
from vizier._src.jax.optimizers.testing import sinusoidal
from vizier.jax import optimizers

from absl.testing import absltest


tfb = tfp.bijectors


class OptaxWrapperTest(parameterized.TestCase, sinusoidal.ConvergenceTestMixin):

  @parameterized.parameters(
      (None,),
      ((-4.0, None),),
      ((None, 5.0),),
      ((-3.0, 3.0),),
  )
  def test_sinusoidal(self, bounds):
    constraints = sinusoidal.bounds_to_constraints(bounds, nest_constraint=True)
    self.assert_converges(
        optimizers.OptaxTrain(optax.adam(5e-2), epochs=100, verbose=True),
        constraints=constraints,
        threshold=5e-3 if bounds is None else 1.0,
        random_restarts=200,
    )


if __name__ == '__main__':
  absltest.main()
