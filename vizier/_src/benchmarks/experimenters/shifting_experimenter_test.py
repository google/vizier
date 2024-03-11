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

"""Tests for shifting_experimenter."""

import numpy as np
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import numpy_experimenter
from vizier._src.benchmarks.experimenters import shifting_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest
from absl.testing import parameterized


class ShiftingExperimenterTest(parameterized.TestCase):

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
  def test_numpy_experimenter(self, func):
    dim = 2
    shift = 1.2
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim))
    shifted_exptr = shifting_experimenter.ShiftingExperimenter(
        exptr=exptr, shift=np.asarray(shift))

    parameters = exptr.problem_statement().search_space.parameters
    self.assertLen(parameters, dim)

    t = pyvizier.Trial(parameters={
        param.name: float(index) for index, param in enumerate(parameters)
    })
    t_shifted = pyvizier.Trial(
        parameters={
            param.name: float(index) - shift
            for index, param in enumerate(parameters)
        }
    )

    exptr.evaluate([t_shifted])
    shifted_exptr.evaluate([t])
    metric_name = exptr.problem_statement().metric_information.item().name

    self.assertAlmostEqual(
        t_shifted.final_measurement_or_die.metrics[metric_name].value,
        t.final_measurement_or_die.metrics[metric_name].value,
    )
    self.assertEqual(t.status, t_shifted.status)

    # Check parameter bounds are shifted.
    shifted_parameters = shifted_exptr.problem_statement(
    ).search_space.parameters
    for param, shifted_param in zip(parameters, shifted_parameters):
      self.assertEqual(param.bounds[0] + shift, shifted_param.bounds[0])
      self.assertEqual(param.bounds[1], shifted_param.bounds[1])

  def test_evaluate_shift(self):
    dim = 2
    shift = np.array([1.2, -2.3])
    func = bbob.Sphere
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim))
    shifted_exptr = shifting_experimenter.ShiftingExperimenter(
        exptr=exptr, shift=np.asarray(shift))

    parameters = exptr.problem_statement().search_space.parameters
    self.assertLen(parameters, dim)

    t = pyvizier.Trial(parameters={
        param.name: float(index) for index, param in enumerate(parameters)
    })
    t_shifted = pyvizier.Trial(
        parameters={
            param.name: float(index) - shift[index]
            for index, param in enumerate(parameters)
        }
    )

    exptr.evaluate([t_shifted])
    shifted_exptr.evaluate([t])
    metric_name = exptr.problem_statement().metric_information.item().name

    self.assertAlmostEqual(
        t_shifted.final_measurement_or_die.metrics[metric_name].value,
        t.final_measurement_or_die.metrics[metric_name].value,
    )
    self.assertEqual(t.status, t_shifted.status)
    self.assertNotEqual(t.parameters, t_shifted.parameters)

  def test_shift_backward(self):
    dim = 2
    shift = np.array([1.2, -2.3])
    func = bbob.Sphere
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim))
    shifted_exptr = shifting_experimenter.ShiftingExperimenter(
        exptr=exptr, shift=np.asarray(shift))
    # Test shift within bounds
    trial = pyvizier.Trial(parameters={'x0': 3.0, 'x1': 1.0})
    shifted_exptr._offset([trial], shift=-shift)
    self.assertEqual(
        trial.parameters.as_dict(), {'x0': 3.0 + 1.2, 'x1': 1.0 - 2.3}
    )
    # Test shift in out of bounds
    trial = pyvizier.Trial(parameters={'x0': -5.0, 'x1': 5.0})
    shifted_exptr._offset([trial], shift=shift)
    self.assertEqual(trial.parameters.as_dict(), {'x0': -5.0, 'x1': 5.0})

  def test_shift_forward(self):
    dim = 2
    shift = np.array([1.2, -2.3])
    func = bbob.Sphere
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim))
    shifted_exptr = shifting_experimenter.ShiftingExperimenter(
        exptr=exptr, shift=np.asarray(shift))
    # Test shift within bounds
    trial = pyvizier.Trial(parameters={'x0': 3.0, 'x1': 1.0})
    shifted_exptr._offset([trial], shift=shift)
    self.assertEqual(
        trial.parameters.as_dict(), {'x0': 3.0 - 1.2, 'x1': 1.0 + 2.3}
    )
    # Test shift in out of bounds
    trial = pyvizier.Trial(parameters={'x0': 5.0, 'x1': -5.0})
    shifted_exptr._offset([trial], shift=-shift)
    self.assertEqual(trial.parameters.as_dict(), {'x0': 5.0, 'x1': -5.0})

  @parameterized.parameters((True,), (False,))
  def test_shift_forward_oob(self, should_restrict):
    dim = 2
    shift = np.array([-2.2, 2.3])
    func = bbob.Sphere
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim)
    )
    shifted_exptr = shifting_experimenter.ShiftingExperimenter(
        exptr=exptr, shift=np.asarray(shift), should_restrict=should_restrict
    )
    # Test OOB shifts does not change parameter values.
    trial = pyvizier.Trial(parameters={'x0': 3.0, 'x1': 1.0})
    shifted_exptr.evaluate([trial])
    self.assertEqual(trial.parameters.as_dict(), {'x0': 3.0, 'x1': 1.0})

    # Test OOB shifts stay within bounds.
    shifted_exptr._offset([trial], shift=shift)
    # x0 is shifted OOB, so clip at 5.0 if should_restrict=True.
    if should_restrict:
      self.assertEqual(trial.parameters.as_dict(), {'x0': 5.0, 'x1': 1.0 - 2.3})
    else:
      self.assertEqual(trial.parameters.as_dict(), {'x0': 5.2, 'x1': 1.0 - 2.3})

  def test_large_shift(self):
    dim = 2
    shift = np.array([10.2, 20.3])
    func = bbob.Sphere
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim))

    with self.assertRaisesRegex(ValueError, 'is too large'):
      shifting_experimenter.ShiftingExperimenter(
          exptr=exptr, shift=np.asarray(shift))

    shifted_exptr = shifting_experimenter.ShiftingExperimenter(
        exptr=exptr, shift=np.asarray(shift), should_restrict=False
    )
    parameters = exptr.problem_statement().search_space.parameters
    self.assertEqual(
        parameters, shifted_exptr.problem_statement().search_space.parameters
    )
    t = pyvizier.Trial(
        parameters={
            param.name: float(index) for index, param in enumerate(parameters)
        }
    )
    shifted_exptr.evaluate([t])

if __name__ == '__main__':
  absltest.main()
