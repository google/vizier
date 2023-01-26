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

"""Tests for acquisitions."""
import mock
import numpy as np
from vizier._src.algorithms.designers.gp import acquisitions
from vizier.pyvizier import converters

from absl.testing import absltest


def _build_mock_continuous_array_specs(n):
  continuous_spec = mock.create_autospec(converters.NumpyArraySpec)
  continuous_spec.type = converters.NumpyArraySpecType.CONTINUOUS
  continuous_spec.num_dimensions = 1
  return [continuous_spec] * n


class TrustRegionTest(absltest.TestCase):

  def test_trust_region_small(self):
    tr = acquisitions.TrustRegion(
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]),
        _build_mock_continuous_array_specs(4),
    )

    np.testing.assert_allclose(
        tr.min_linf_distance(
            np.array([
                [0., .2, .3, 0.],
                [.9, .8, .9, .9],
                [1., 1., 1., 1.],
            ]),), np.array([0.3, 0.2, 0.]))
    self.assertAlmostEqual(tr.trust_radius, 0.224, places=3)

  def test_trust_region_bigger(self):
    tr = acquisitions.TrustRegion(
        np.vstack(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
            * 10
        ),
        _build_mock_continuous_array_specs(4),
    )
    np.testing.assert_allclose(
        tr.min_linf_distance(
            np.array([
                [0., .2, .3, 0.],
                [.9, .8, .9, .9],
                [1., 1., 1., 1.],
            ]),), np.array([0.3, 0.2, 0.]))
    self.assertAlmostEqual(tr.trust_radius, 0.44, places=3)


if __name__ == '__main__':
  absltest.main()
