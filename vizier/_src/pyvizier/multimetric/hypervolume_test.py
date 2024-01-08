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

import numpy as np

from vizier._src.pyvizier.multimetric import hypervolume
from absl.testing import absltest


class HypervolumeTest(absltest.TestCase):

  def testParetoHypervolume(self):
    x = np.random.normal()
    y = np.random.normal()
    points = np.array([[x, y], [y, x]])
    origin = np.array([-1, -1])
    pf = hypervolume.ParetoFrontier(points, origin)
    self.assertAlmostEqual(
        pf.hypervolume(),
        max(2 * (x + 1) * (y + 1) - min(x + 1, y + 1)**2, 0.0),
        delta=0.3)

  def testParetoHypervolumeCumulative(self):
    points = np.random.normal(size=(100, 4))
    origin = np.zeros(4)

    pf = hypervolume.ParetoFrontier(points, origin)
    cumulative_vol = pf.hypervolume(is_cumulative=True)
    self.assertLen(cumulative_vol, len(points))
    for i in range(len(points)):
      if i < len(points) - 1:
        self.assertGreaterEqual(cumulative_vol[i + 1], cumulative_vol[i])
      else:
        self.assertAlmostEqual(cumulative_vol[i], pf.hypervolume())

  def testDimensionMismatchInit(self):
    points = np.array([[2, 1], [1, 2], [0, 0]])
    origin_3d = np.array([-1, -1, 1])
    # Dimension mismatch.
    with self.assertRaisesRegex(ValueError, 'Dimension mismatch'):
      hypervolume.ParetoFrontier(points, origin_3d)

  def testDimensionMismatchHypervolume(self):
    points = np.array([[2, 1], [1, 2], [0, 0]])
    origin_3d = np.array([-1, -1, 1])
    pf = hypervolume.ParetoFrontier(points, origin=np.array([0, 0]))
    # Dimension mismatch.
    with self.assertRaisesRegex(ValueError, 'Dimension mismatch'):
      pf.hypervolume(additional_points=origin_3d[..., np.newaxis])


if __name__ == '__main__':
  absltest.main()
