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

"""Tests for experimenter_factory."""

import numpy as np
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter_factory

from absl.testing import absltest
from absl.testing import parameterized


class ExperimenterFactoryTest(parameterized.TestCase):

  @parameterized.parameters(
      {'bbob_name': 'Sphere'},
      {'bbob_name': 'LinearSlope'},
      {'bbob_name': 'RosenbrockRotated'},
      {'bbob_name': 'SchaffersF7IllConditioned'},
  )
  def testBBOBFactory(self, bbob_name):
    dim = 4
    bbob_factory = experimenter_factory.BBOBExperimenterFactory(
        name=bbob_name, dim=dim)
    exptr = bbob_factory()

    parameters = exptr.problem_statement().search_space.parameters
    self.assertLen(parameters, dim)

    t = pyvizier.Trial(parameters={
        param.name: float(index) for index, param in enumerate(parameters)
    })
    exptr.evaluate([t])
    self.assertEqual(t.status, pyvizier.TrialStatus.COMPLETED)

  def testBBOBFactoryError(self):

    with self.assertRaisesRegex(ValueError, 'not a valid BBOB'):
      experimenter_factory.BBOBExperimenterFactory(name='Error', dim=3)()

  def testSingleObjectiveFactory(self):
    dim = 5
    bbob_factory = experimenter_factory.BBOBExperimenterFactory(
        name='Sphere', dim=dim)
    exptr_factory = experimenter_factory.SingleObjectiveExperimenterFactory(
        base_factory=bbob_factory,
        shift=np.asarray(1.9),
        noise_type='moderate_gaussian')
    exptr = exptr_factory()

    self.assertIn('Shifting', str(exptr))
    self.assertIn('Noisy', str(exptr))

    parameters = exptr.problem_statement().search_space.parameters
    self.assertLen(parameters, dim)

    t = pyvizier.Trial(parameters={
        param.name: float(index) for index, param in enumerate(parameters)
    })
    exptr.evaluate([t])
    self.assertEqual(t.status, pyvizier.TrialStatus.COMPLETED)

  def testCombinedFactory(self):
    dim = 5
    experimenter_factories = {
        'sphere': experimenter_factory.BBOBExperimenterFactory('Sphere', dim),
        'discus': experimenter_factory.BBOBExperimenterFactory('Discus', dim),
    }
    exptr = experimenter_factory.CombinedExperimenterFactory(
        base_factories=experimenter_factories
    )()

    parameters = exptr.problem_statement().search_space.parameters
    self.assertLen(parameters, dim)

    t = pyvizier.Trial(
        parameters={
            param.name: float(index) for index, param in enumerate(parameters)
        }
    )
    exptr.evaluate([t])
    self.assertIn('sphere', t.final_measurement_or_die.metrics)
    self.assertIn('discus', t.final_measurement_or_die.metrics)
    self.assertEqual(t.status, pyvizier.TrialStatus.COMPLETED)

  def testSingleObjectiveFactoryDiscrete(self):
    dim = 5
    bbob_factory = experimenter_factory.BBOBExperimenterFactory(
        name='Sphere', dim=dim
    )
    exptr_factory = experimenter_factory.SingleObjectiveExperimenterFactory(
        base_factory=bbob_factory,
        shift=np.asarray(1.9),
        noise_type='moderate_gaussian',
        num_normalization_samples=10,
        discrete_dict={0: 3, 1: 5},
        categorical_dict={4: 4},
    )
    exptr = exptr_factory()

    self.assertIn('Shifting', str(exptr))
    self.assertIn('Noisy', str(exptr))
    self.assertIn('Discretizing', str(exptr))
    self.assertIn('Normalizing', str(exptr))

    space = exptr.problem_statement().search_space
    self.assertEqual(space.num_parameters(pyvizier.ParameterType.DISCRETE), 2)
    self.assertEqual(
        space.num_parameters(pyvizier.ParameterType.CATEGORICAL), 1
    )

    parameters = {}
    for param in space.parameters:
      if param.type == pyvizier.ParameterType.DOUBLE:
        parameters[param.name] = param.bounds[0]
      else:
        parameters[param.name] = param.feasible_values[0]
    t = pyvizier.Trial(parameters=parameters)
    exptr.evaluate([t])
    self.assertEqual(t.status, pyvizier.TrialStatus.COMPLETED)

  def testSingleObjectiveWithPermutation(self):
    dim = 5
    bbob_factory = experimenter_factory.BBOBExperimenterFactory(
        name='Sphere', dim=dim
    )
    exptr_factory = experimenter_factory.SingleObjectiveExperimenterFactory(
        base_factory=bbob_factory,
        shift=np.asarray(1.9),
        categorical_dict={0: 3, 1: 5, 2: 4},
        permute_categoricals=True,
    )
    exptr = exptr_factory()

    self.assertIn('Shifting', str(exptr))
    self.assertIn('Discretizing', str(exptr))
    self.assertIn('Permuting', str(exptr))

    space = exptr.problem_statement().search_space
    self.assertEqual(
        space.num_parameters(pyvizier.ParameterType.CATEGORICAL), 3
    )

    parameters = {}
    for param in space.parameters:
      if param.type == pyvizier.ParameterType.DOUBLE:
        parameters[param.name] = param.bounds[0]
      else:
        parameters[param.name] = param.feasible_values[0]
    t = pyvizier.Trial(parameters=parameters)
    exptr.evaluate([t])
    self.assertEqual(t.status, pyvizier.TrialStatus.COMPLETED)

  def testSingleObjectiveFactoryError(self):
    dim = 4
    bbob_factory = experimenter_factory.BBOBExperimenterFactory(
        name='Sphere', dim=dim)

    with self.assertRaisesRegex(ValueError, 'not supported'):
      experimenter_factory.SingleObjectiveExperimenterFactory(
          base_factory=bbob_factory, shift=np.asarray(1.9),
          noise_type='ERROR')()

  def testDiscreteSingleObjectiveFactoryError(self):
    dim = 4
    bbob_factory = experimenter_factory.BBOBExperimenterFactory(
        name='Sphere', dim=dim
    )

    with self.assertRaisesRegex(ValueError, 'overlap'):
      experimenter_factory.SingleObjectiveExperimenterFactory(
          base_factory=bbob_factory,
          shift=np.asarray(1.9),
          noise_type='moderate_gaussian',
          num_normalization_samples=10,
          discrete_dict={0: 3, 1: 5},
          categorical_dict={1: 4},
      )()

  def testSingleObjectiveSerialization(self):
    dim = 5
    bbob_factory = experimenter_factory.BBOBExperimenterFactory(
        name='Sphere', dim=dim
    )
    bbob_metadata = bbob_factory.dump()
    recovered_bbob = experimenter_factory.BBOBExperimenterFactory.recover(
        bbob_metadata
    )
    self.assertEqual(recovered_bbob.dim, dim)
    self.assertEqual(recovered_bbob.name, 'Sphere')

    exptr_factory = experimenter_factory.SingleObjectiveExperimenterFactory(
        base_factory=bbob_factory,
        shift=np.asarray(1.9),
        noise_type='moderate_gaussian',
        num_normalization_samples=10,
        discrete_dict={0: 3, 1: 5},
    )
    factory_metadata = exptr_factory.dump()
    recovered_factory = (
        experimenter_factory.SingleObjectiveExperimenterFactory.recover(
            factory_metadata
        )
    )
    self.assertEqual(recovered_factory.noise_type, 'moderate_gaussian')
    self.assertEqual(recovered_factory.num_normalization_samples, 10)
    np.testing.assert_array_equal(recovered_factory.shift, np.asarray(1.9))
    self.assertDictEqual(recovered_factory.discrete_dict, {0: 3, 1: 5})


if __name__ == '__main__':
  absltest.main()
