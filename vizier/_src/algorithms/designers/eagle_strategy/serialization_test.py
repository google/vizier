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

"""Tests for serialization."""

import numpy as np
from vizier._src.algorithms.designers.eagle_strategy import serialization
from vizier._src.algorithms.designers.eagle_strategy import testing

from absl.testing import absltest
from absl.testing import parameterized


class SerializationTest(parameterized.TestCase):

  @parameterized.parameters(
      {'x_values': [1, 2, 5, -2], 'obj_values': [2, 10, -2, 5]},
      {'x_values': [1, 2, 5, -2], 'obj_values': [None, 10, -2, None]},
  )
  def test_restore_pool(self, x_values, obj_values):
    firefly_pool = testing.create_fake_populated_firefly_pool(
        capacity=20, x_values=x_values, obj_values=obj_values
    )
    encoded = serialization.partially_serialize_firefly_pool(firefly_pool)
    # Restore the firefly pool.
    utils = firefly_pool._utils
    restored_firefly_pool = serialization.restore_firefly_pool(utils, encoded)
    # Check that the restored and original firefly_pool are the same.
    self.assertEqual(restored_firefly_pool._capacity, firefly_pool._capacity)
    self.assertEqual(restored_firefly_pool._last_id, firefly_pool._last_id)
    self.assertEqual(restored_firefly_pool._max_fly_id,
                     firefly_pool._max_fly_id)
    self.assertEqual(
        set(firefly_pool._pool.keys()), set(restored_firefly_pool._pool.keys()))
    for fly_id, firefly in firefly_pool._pool.items():
      restored_firefly = restored_firefly_pool._pool[fly_id]
      self.assertEqual(restored_firefly.id_, firefly.id_)
      self.assertEqual(restored_firefly.perturbation, firefly.perturbation)
      self.assertEqual(restored_firefly.generation, firefly.generation)
      self.assertEqual(restored_firefly.trial.parameters,
                       firefly.trial.parameters)
      self.assertEqual(
          restored_firefly.trial.infeasible, firefly.trial.infeasible
      )
      self.assertEqual(
          restored_firefly.trial.infeasibility_reason,
          firefly.trial._infeasibility_reason,
      )
      self.assertEqual(
          restored_firefly.trial.final_measurement_or_die.metrics[
              'objective'
          ].value,
          firefly.trial.final_measurement_or_die.metrics['objective'].value,
      )

  def test_restore_rng(self):
    rng = np.random.default_rng(0)
    serialized_rng = serialization.serialize_rng(rng)
    rand_value = rng.normal()
    restored_rng = serialization.restore_rng(serialized_rng)
    assert restored_rng.normal() == rand_value


if __name__ == '__main__':
  absltest.main()
