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

"""Tests for failing."""

from vizier import pyvizier as vz
from vizier._src.algorithms.testing import failing
from absl.testing import absltest


class FailingTest(absltest.TestCase):

  def test_failing_designer(self):
    failing_designer = failing.FailingDesigner()
    with self.assertRaises(failing.FailedSuggestError):
      failing_designer.suggest(1)

  def test_alternate_failing_designer(self):
    search_space = vz.SearchSpace()
    search_space.root.add_float_param("x", 0.0, 1.0)
    alt_failing_designer = failing.AlternateFailingDesigner(search_space)
    for _ in range(5):
      alt_failing_designer.suggest(1)
      with self.assertRaises(failing.FailedSuggestError):
        alt_failing_designer.suggest(1)


if __name__ == "__main__":
  absltest.main()
