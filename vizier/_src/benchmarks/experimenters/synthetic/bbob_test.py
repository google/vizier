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

"""Tests for bbob."""

import sys

import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest
from absl.testing import parameterized


class BbobTest(parameterized.TestCase):

  @parameterized.parameters(
      {'scale_type': vz.ScaleType.LINEAR}, {'scale_type': vz.ScaleType.LOG}
  )
  def test_bbob_problem_statement_scale_type(self, scale_type):
    problem = bbob.DefaultBBOBProblemStatement(10, scale_type=scale_type)
    for p in problem.search_space.parameters:
      self.assertEqual(p.scale_type, scale_type)

  def test_rotation(self):
    rotation = bbob._R(5, sys.maxsize, 'A', b'XAFDAF', 3)
    rotation2 = bbob._R(5, sys.maxsize, 'A', b'XAFDAF', 3)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(5), atol=1e-5)
    np.testing.assert_array_equal(rotation, rotation2)


if __name__ == '__main__':
  absltest.main()
