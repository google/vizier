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

"""End-to-end tests for Vizier backend for PyGlove."""

import pyglove as pg
from vizier._src.pyglove import oss_vizier
from vizier._src.service import clients as pyvizier_clients
from absl.testing import absltest


pyvizier_clients.environment_variables.servicer_use_sql_ram()


class PygloveTest(absltest.TestCase):
  """Tests for using Vizier as PyGlove backend."""

  def test_sample(self):
    oss_vizier.init('my_study')

    examples = []
    for x, f in pg.sample(
        pg.oneof([1, 2, 3]), pg.geno.Random(seed=1), num_examples=3
    ):
      f(x)
      examples.append(x)

    self.assertEqual(examples, [1, 3, 1])
    result = pg.poll_result('')
    self.assertLen(result.trials, 3)


if __name__ == '__main__':
  absltest.main()
