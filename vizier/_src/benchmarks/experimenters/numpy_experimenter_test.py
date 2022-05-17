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

    completed_trial = exptr.evaluate([t])[0]
    metric_name = exptr.problem_statement().metric_information.item().name
    self.assertAlmostEqual(
        func(np.array([0.0, 1.0])),
        completed_trial.final_measurement.metrics[metric_name].value)
    self.assertEqual(completed_trial.status, pyvizier.TrialStatus.COMPLETED)

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

    completed_trials = exptr.evaluate([t1, t2])
    for trial in completed_trials:
      self.assertEmpty(trial.final_measurement.metrics)


if __name__ == '__main__':
  absltest.main()
