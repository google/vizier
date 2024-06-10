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
from vizier._src.benchmarks.experimenters import discretizing_experimenter
from vizier._src.benchmarks.experimenters import numpy_experimenter
from vizier._src.benchmarks.experimenters import permuting_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest
from absl.testing import parameterized


class PermutingExperimenterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('Sphere', bbob.Sphere),
      ('Rastrigin', bbob.Rastrigin),
      ('BuecheRastrigin', bbob.BuecheRastrigin),
      ('LinearSlope', bbob.LinearSlope),
      ('AttractiveSector', bbob.AttractiveSector),
      ('StepEllipsoidal', bbob.StepEllipsoidal),
      ('RosenbrockRotated', bbob.RosenbrockRotated),
      ('Discus', bbob.Discus),
      ('BentCigar', bbob.BentCigar),
      ('SharpRidge', bbob.SharpRidge),
      ('DifferentPowers', bbob.DifferentPowers),
      ('Weierstrass', bbob.Weierstrass),
      ('SchaffersF7', bbob.SchaffersF7),
      ('SchaffersF7IllConditioned', bbob.SchaffersF7IllConditioned),
      ('GriewankRosenbrock', bbob.GriewankRosenbrock),
      ('Schwefel', bbob.Schwefel),
      ('Katsuura', bbob.Katsuura),
      ('Lunacek', bbob.Lunacek),
      ('Gallagher101Me', bbob.Gallagher101Me),
  )
  def testNumpyExperimenter(self, func):
    dim = 3
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim)
    )

    # Asserts parameters are the same.
    parameters = list(exptr.problem_statement().search_space.parameters)
    self.assertLen(parameters, dim)

    discretization = {
        parameters[0].name: ['0.1', '0.5', '-0.1'],
        parameters[1].name: [0.2, 1.2, 2.2],
    }

    dis_exptr = discretizing_experimenter.DiscretizingExperimenter(
        exptr, discretization
    )
    discretized_parameters = [
        p.name
        for p in dis_exptr.problem_statement().search_space.parameters
        if np.isfinite(p.num_feasible_values)
    ]
    self.assertLen(discretized_parameters, 2)
    permuted_exptr = permuting_experimenter.PermutingExperimenter(
        dis_exptr, discretized_parameters, seed=0
    )

    self.assertEqual(
        permuted_exptr.problem_statement(), dis_exptr.problem_statement()
    )

    parameters = {
        parameters[0].name: '-0.1',
        parameters[1].name: 1.2,
        parameters[2].name: 1.5,
    }
    t = pyvizier.Trial(parameters=parameters)

    permuted_exptr.evaluate([t])
    metric_name = exptr.problem_statement().metric_information.item().name
    self.assertNotAlmostEqual(
        func(np.array([-0.1, 1.2, 1.5])),
        t.final_measurement_or_die.metrics[metric_name].value,
    )
    self.assertEqual(t.status, pyvizier.TrialStatus.COMPLETED)


if __name__ == '__main__':
  absltest.main()
