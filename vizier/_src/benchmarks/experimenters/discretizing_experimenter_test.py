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
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest
from absl.testing import parameterized


class DiscretizingExperimenterTest(parameterized.TestCase):

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
    dim = 3
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim))

    # Asserts parameters are the same.
    parameters = list(exptr.problem_statement().search_space.parameters)
    self.assertLen(parameters, dim)

    discretization = {
        parameters[0].name: ['-1', '0', '1'],
        parameters[1].name: [0, 1, 2]
    }

    dis_exptr = discretizing_experimenter.DiscretizingExperimenter(
        exptr, discretization)
    discretized_parameters = dis_exptr.problem_statement(
    ).search_space.parameters

    self.assertLen(discretized_parameters, dim)
    self.assertListEqual([p.type for p in discretized_parameters], [
        pyvizier.ParameterType.CATEGORICAL, pyvizier.ParameterType.DISCRETE,
        pyvizier.ParameterType.DOUBLE
    ])

    parameters = {
        parameters[0].name: '0',
        parameters[1].name: 1,
        parameters[2].name: 1.5
    }
    t = pyvizier.Trial(parameters=parameters)

    dis_exptr.evaluate([t])
    metric_name = exptr.problem_statement().metric_information.item().name
    self.assertAlmostEqual(
        func(np.array([0.0, 1.0, 1.5])),
        t.final_measurement_or_die.metrics[metric_name].value,
    )
    self.assertEqual(t.status, pyvizier.TrialStatus.COMPLETED)
    self.assertDictEqual(t.parameters.as_dict(), parameters)

  def testGridCreation(self):
    dim = 3
    func = bbob.Sphere
    problem_statement = bbob.DefaultBBOBProblemStatement(dim)
    # Mutate the last parameter.
    parameters = list(problem_statement.search_space.parameters)
    log_param = problem_statement.search_space.pop(parameters[-1].name)
    problem_statement.search_space.add(
        pyvizier.ParameterConfig.factory(
            log_param.name,
            scale_type=pyvizier.ScaleType.LOG,
            bounds=(0.01, 10.0),
        )
    )
    exptr = numpy_experimenter.NumpyExperimenter(func, problem_statement)

    parameters = list(exptr.problem_statement().search_space.parameters)
    self.assertLen(parameters, dim)

    discretization = {parameters[0].name: 3, parameters[-1].name: 4}

    dis_exptr = (
        discretizing_experimenter.DiscretizingExperimenter.create_with_grid(
            exptr, discretization
        )
    )
    search_space = dis_exptr.problem_statement().search_space
    self.assertEqual(search_space.num_parameters(), dim)
    self.assertEqual(
        search_space.num_parameters(pyvizier.ParameterType.DISCRETE), 2
    )
    self.assertLen(search_space.parameters[0].feasible_values, 3)
    self.assertSequenceEqual(
        search_space.parameters[0].feasible_values, [-5, 0, 5]
    )
    self.assertLen(search_space.parameters[-1].feasible_values, 4)
    self.assertSequenceAlmostEqual(
        search_space.parameters[-1].feasible_values,
        [0.01, 0.1, 1.0, 10.0],
        places=5,
    )

  def testGridCreationError(self):
    dim = 3
    func = bbob.Sphere
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim)
    )
    discretization = {'not_found_error': 3}

    with self.assertRaisesRegex(ValueError, 'not in search space'):
      discretizing_experimenter.DiscretizingExperimenter.create_with_grid(
          exptr, discretization
      )

  def testGridCreationErrorNonDouble(self):
    dim = 5
    func = bbob.Sphere
    problem_statement = bbob.DefaultBBOBProblemStatement(dim)
    exptr = numpy_experimenter.NumpyExperimenter(func, problem_statement)
    parameters = list(exptr.problem_statement().search_space.parameters)
    discretization = {parameters[0].name: 3, parameters[1].name: 4}

    dis_exptr = (
        discretizing_experimenter.DiscretizingExperimenter.create_with_grid(
            exptr, discretization
        )
    )

    with self.assertRaisesRegex(ValueError, 'Non-double parameters'):
      discretizing_experimenter.DiscretizingExperimenter.create_with_grid(
          dis_exptr, discretization
      )


if __name__ == '__main__':
  absltest.main()
