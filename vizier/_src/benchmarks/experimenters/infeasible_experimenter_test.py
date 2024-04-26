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
from vizier._src.benchmarks.experimenters import infeasible_experimenter
from vizier._src.benchmarks.experimenters import numpy_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob
from absl.testing import absltest


class HashingInfeasibleExperimenterTest(absltest.TestCase):

  def test_consistency(self):
    exptr = numpy_experimenter.NumpyExperimenter(
        bbob.Sphere, bbob.DefaultBBOBProblemStatement(2)
    )
    exptr = infeasible_experimenter.HashingInfeasibleExperimenter(
        exptr, infeasible_prob=0.5, seed=0
    )

    for i in range(10):
      trials = [vz.Trial(parameters={'x0': i, 'x1': -i}) for _ in range(10)]
      trials += [vz.Trial(parameters={'x1': -i, 'x0': i}) for _ in range(10)]
      exptr.evaluate(trials)

      for t in trials:
        self.assertEqual(t.infeasible, trials[0].infeasible)
        self.assertEqual(
            t.final_measurement_or_die, trials[0].final_measurement_or_die
        )


class ParamRegionInfeasibleExperimenterTest(absltest.TestCase):

  def test_e2e(self):
    exptr = numpy_experimenter.NumpyExperimenter(
        bbob.Sphere, bbob.DefaultBBOBProblemStatement(2)
    )
    exptr = infeasible_experimenter.ParamRegionInfeasibleExperimenter(
        exptr, parameter_name='x0', infeasible_interval=(0.0, 0.5)
    )

    infeasible_trial = vz.Trial(parameters={'x0': -3.5, 'x1': 0})
    feasible_trial = vz.Trial(parameters={'x0': 3.5, 'x1': 0})
    exptr.evaluate([infeasible_trial, feasible_trial])

    self.assertTrue(infeasible_trial.infeasible)
    self.assertFalse(feasible_trial.infeasible)


if __name__ == '__main__':
  absltest.main()
