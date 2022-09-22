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
  def testNumpyExperimenter(self, func):
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
    t_shifted = pyvizier.Trial(parameters={
        param.name: float(index) + shift
        for index, param in enumerate(parameters)
    })

    exptr.evaluate([t_shifted])
    shifted_exptr.evaluate([t])
    metric_name = exptr.problem_statement().metric_information.item().name

    self.assertAlmostEqual(
        t_shifted.final_measurement.metrics[metric_name].value,
        t.final_measurement.metrics[metric_name].value)
    self.assertEqual(t.status, t_shifted.status)

    # Check parameter bounds are shifted.
    shifted_parameters = shifted_exptr.problem_statement(
    ).search_space.parameters
    for param, shifted_param in zip(parameters, shifted_parameters):
      self.assertEqual(param.bounds[0], shifted_param.bounds[0])
      self.assertEqual(param.bounds[1] - shift, shifted_param.bounds[1])

  def testVectorShift(self):
    dim = 2
    shift = [1.2, -2.3]
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
            param.name: float(index) + shift[index]
            for index, param in enumerate(parameters)
        })

    exptr.evaluate([t_shifted])
    shifted_exptr.evaluate([t])
    metric_name = exptr.problem_statement().metric_information.item().name

    self.assertAlmostEqual(
        t_shifted.final_measurement.metrics[metric_name].value,
        t.final_measurement.metrics[metric_name].value)
    self.assertEqual(t.status, t_shifted.status)
    self.assertNotEqual(t.parameters, t_shifted.parameters)

  def testLargeShift(self):
    dim = 2
    shift = [10.2, 20.3]
    func = bbob.Sphere
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim))

    with self.assertRaisesRegex(ValueError, 'is too large'):
      shifting_experimenter.ShiftingExperimenter(
          exptr=exptr, shift=np.asarray(shift))

if __name__ == '__main__':
  absltest.main()
