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
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters.synthetic import branin
from vizier._src.benchmarks.testing import experimenter_testing

from absl.testing import absltest


class Branin2DExperimenterTest(absltest.TestCase):

  def test_branin_impl(self):
    np.testing.assert_allclose(
        branin._branin(
            np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        ),
        0.397887,
        atol=1e-5,
    )

  def test_experimenter_argmin(self):
    trial = vz.Trial(parameters={'x1': -np.pi, 'x2': 12.275})
    branin.Branin2DExperimenter().evaluate([trial])
    self.assertAlmostEqual(
        trial.final_measurement_or_die.metrics.get_value('value', np.nan),
        0.397887,
        places=5,
    )

  def test_experimenter(self):
    experimenter_testing.assert_evaluates_random_suggestions(
        self, branin.Branin2DExperimenter()
    )


if __name__ == '__main__':
  absltest.main()
