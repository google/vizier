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
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import multiobjective_experimenter
from vizier._src.benchmarks.experimenters import numpy_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest


class MultiobjectiveExperimenterTest(absltest.TestCase):

  def test_mulitobjective_numpy(self):
    dim = 2
    func1 = bbob.Sphere
    func2 = bbob.Rastrigin
    exptr1 = numpy_experimenter.NumpyExperimenter(
        func1, bbob.DefaultBBOBProblemStatement(dim)
    )
    exptr2 = numpy_experimenter.NumpyExperimenter(
        func2, bbob.DefaultBBOBProblemStatement(dim)
    )
    exptr = multiobjective_experimenter.MultiObjectiveExperimenter(
        {'m1': exptr1, 'm2': exptr2}
    )
    parameters = exptr1.problem_statement().search_space.parameters
    self.assertLen(parameters, dim)

    t = pyvizier.Trial(
        parameters={
            param.name: float(index) for index, param in enumerate(parameters)
        }
    )

    exptr.evaluate([t])
    self.assertAlmostEqual(
        func1(np.array([0.0, 1.0])),
        t.final_measurement_or_die.metrics['m1'].value,
    )
    self.assertAlmostEqual(
        func2(np.array([0.0, 1.0])),
        t.final_measurement_or_die.metrics['m2'].value,
    )
    self.assertEqual(t.status, pyvizier.TrialStatus.COMPLETED)

  def test_dimension_mismatch(self):
    exptr1 = numpy_experimenter.NumpyExperimenter(
        bbob.Sphere, bbob.DefaultBBOBProblemStatement(dimension=3)
    )
    exptr2 = numpy_experimenter.NumpyExperimenter(
        bbob.Rastrigin, bbob.DefaultBBOBProblemStatement(dimension=6)
    )
    with self.assertRaisesRegex(ValueError, 'space must match'):
      multiobjective_experimenter.MultiObjectiveExperimenter(
          {'m1': exptr1, 'm2': exptr2}
      )


if __name__ == '__main__':
  absltest.main()
