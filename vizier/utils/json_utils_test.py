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

"""Tests for json_utils."""

import json
import numpy as np

from vizier.utils import json_utils
from absl.testing import absltest
from absl.testing import parameterized


class JsonUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(shape=[3, 0]),
      dict(shape=[3, 5]),
  )
  def test_dump_and_recover(self, shape):
    original = {'a': np.zeros(shape), 'b': 3}
    dumped = json.dumps(original, cls=json_utils.NumpyEncoder)
    loaded = json.loads(dumped, cls=json_utils.NumpyDecoder)
    np.testing.assert_array_equal(original['a'], loaded['a'])
    self.assertEqual(original['b'], loaded['b'])


if __name__ == '__main__':
  absltest.main()
