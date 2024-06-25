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

"""Tests for hartmann."""

import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters.synthetic import hartmann
from vizier._src.benchmarks.testing import experimenter_testing
from absl.testing import absltest


class Hartmann3DExperimenterTest(absltest.TestCase):

  def test_experimenter_argmin(self):
    trial = vz.Trial(
        parameters={
            f'x{i+1}': x for i, x in enumerate([0.114614, 0.555649, 0.852547])
        }
    )
    hartmann.HartmannExperimenter.from_3d().evaluate([trial])
    self.assertAlmostEqual(
        trial.final_measurement_or_die.metrics.get_value('value', np.nan),
        -3.86278,
        places=5,
    )

  def test_experimenter(self):
    experimenter_testing.assert_evaluates_random_suggestions(
        self, hartmann.HartmannExperimenter.from_3d()
    )


class Hartmann6DExperimenterTest(absltest.TestCase):

  def test_experimenter_argmin(self):
    trial = vz.Trial(
        parameters={
            f'x{i+1}': x
            for i, x in enumerate(
                [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
            )
        }
    )
    hartmann.HartmannExperimenter.from_6d().evaluate([trial])
    self.assertAlmostEqual(
        trial.final_measurement_or_die.metrics.get_value('value', np.nan),
        -3.32237,
        places=5,
    )

  def test_experimenter(self):
    experimenter_testing.assert_evaluates_random_suggestions(
        self, hartmann.HartmannExperimenter.from_6d()
    )


if __name__ == '__main__':
  absltest.main()
