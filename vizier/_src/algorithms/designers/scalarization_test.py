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

from jax import numpy as jnp
from vizier._src.algorithms.designers import scalarization
from absl.testing import absltest


class ScalarizationsTest(absltest.TestCase):

  def test_linear_scalarizer(self):
    scalarizer = scalarization.LinearScalarization(
        weights=jnp.array([0.1, 0.2])
    )
    self.assertAlmostEqual(scalarizer(jnp.array([3.0, 4.5])), 1.2)

  def test_chebyshev_scalarizer(self):
    scalarizer = scalarization.ChebyshevScalarization(
        weights=jnp.array([0.1, 0.2])
    )
    self.assertAlmostEqual(scalarizer(jnp.array([3.0, 4.5])), 0.3)

  def test_hypervolume_scalarizer(self):
    scalarizer = scalarization.HyperVolumeScalarization(
        weights=jnp.array([0.1, 0.2])
    )
    self.assertAlmostEqual(scalarizer(jnp.array([3.0, 4.5])), 22.5)

  def test_hypervolume_scalarizer_with_reference(self):
    scalarizer = scalarization.HyperVolumeScalarization(
        weights=jnp.array([0.1, 0.2]), reference_point=jnp.array([-1])
    )
    self.assertAlmostEqual(scalarizer(jnp.array([3.0, 4.5])), 27.5)

  def test_augmented_scalarizer(self):
    scalarizer = scalarization.LinearAugmentedScalarization(
        weights=jnp.array([0.1, 0.2]),
        scalarization_factory=scalarization.HyperVolumeScalarization,
    )
    # Should be the sum of hypervolume and linear scalarizations.
    self.assertAlmostEqual(scalarizer(jnp.array([3.0, 4.5])), 23.7)

if __name__ == "__main__":
  absltest.main()
