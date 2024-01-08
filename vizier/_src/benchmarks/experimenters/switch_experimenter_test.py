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

from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import numpy_experimenter
from vizier._src.benchmarks.experimenters import switch_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob
from absl.testing import absltest


class SwitchExperimenterTest(absltest.TestCase):

  def test_switch_experimenter(self):
    exptr0 = numpy_experimenter.NumpyExperimenter(
        lambda x: 0.0 * x, bbob.DefaultBBOBProblemStatement(1, metric_name='0')
    )
    exptr1 = numpy_experimenter.NumpyExperimenter(
        lambda x: 1.0 * x, bbob.DefaultBBOBProblemStatement(1, metric_name='1')
    )
    switch_exptr = switch_experimenter.SwitchExperimenter([exptr0, exptr1])

    t0 = vz.Trial(parameters={'switch': 0, 'x0': 100.0})
    t1 = vz.Trial(parameters={'switch': 1, 'x0': 100.0})

    switch_exptr.evaluate([t0, t1])

    self.assertEqual(t0.final_measurement.metrics['switch_metric'].value, 0.0)  # pytype:disable=attribute-error
    self.assertEqual(t1.final_measurement.metrics['switch_metric'].value, 100.0)  # pytype:disable=attribute-error


if __name__ == '__main__':
  absltest.main()
