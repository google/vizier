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

"""Tests for numpy_experimenter."""

import numpy as np
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import numpy_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest
from absl.testing import parameterized


class NumpyExperimenterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('Sphere', bbob.Sphere), ('Rastrigin', bbob.Rastrigin),
      ('BuecheRastrigin', bbob.BuecheRastrigin),
      ('LinearSlope', bbob.LinearSlope),
      ('AttractiveSector', bbob.AttractiveSector),
      ('StepEllipsoidal', bbob.StepEllipsoidal),
      ('RosenbrockRotated', bbob.RosenbrockRotated), ('Discus', bbob.Discus),
      ('BentCigar', bbob.BentCigar), ('SharpRidge', bbob.SharpRidge),
      ('DifferentPowers', bbob.DifferentPowers),
      ('Weierstrass', bbob.Weierstrass), ('SchaffersF7', bbob.SchaffersF7),
      ('SchaffersF7IllConditioned', bbob.SchaffersF7IllConditioned),
      ('GriewankRosenbrock', bbob.GriewankRosenbrock),
      ('Schwefel', bbob.Schwefel), ('Katsuura', bbob.Katsuura),
      ('Lunacek', bbob.Lunacek), ('Gallagher101Me', bbob.Gallagher101Me))
  def testNumpyExperimenter(self, func):
    dim = 2
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim))

    parameters = exptr.problem_statement().search_space.parameters
    self.assertLen(parameters, dim)

    t = pyvizier.Trial(parameters={
        param.name: float(index) for index, param in enumerate(parameters)
    })

    exptr.evaluate([t])
    metric_name = exptr.problem_statement().metric_information.item().name
    self.assertAlmostEqual(
        func(np.array([0.0, 1.0])),
        t.final_measurement_or_die.metrics[metric_name].value,
    )
    self.assertEqual(t.status, pyvizier.TrialStatus.COMPLETED)

  def testNonFinite(self):
    dim = 2
    exptr = numpy_experimenter.NumpyExperimenter(
        impl=lambda x: np.inf,
        problem_statement=bbob.DefaultBBOBProblemStatement(dim))

    parameters = exptr.problem_statement().search_space.parameters
    self.assertLen(parameters, dim)

    t1 = pyvizier.Trial(parameters={
        param.name: float(index) for index, param in enumerate(parameters)
    })
    t2 = pyvizier.Trial(parameters={
        param.name: -float(index) for index, param in enumerate(parameters)
    })

    trials = [t1, t2]
    exptr.evaluate(trials)
    for trial in trials:
      self.assertEmpty(trial.final_measurement_or_die.metrics)
      self.assertTrue(trial.infeasible)

  def testNotInSearchSpace(self):
    exptr = numpy_experimenter.NumpyExperimenter(
        impl=lambda x: x,
        problem_statement=bbob.DefaultBBOBProblemStatement(1),
    )

    t1 = pyvizier.Trial(parameters={'yyyy': 0.0})
    with self.assertRaises(ValueError):
      exptr.evaluate([t1])


if __name__ == '__main__':
  absltest.main()
