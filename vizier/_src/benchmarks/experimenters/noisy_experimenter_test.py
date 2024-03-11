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

"""Tests for noisy_experimenter."""

import numpy as np

from vizier import pyvizier
from vizier._src.benchmarks.experimenters import noisy_experimenter
from vizier._src.benchmarks.experimenters import numpy_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest
from absl.testing import parameterized


class NoisyExperimenterTest(parameterized.TestCase):

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
  def testDeterministicNoiseApply(self, func):
    dim = 2
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim))
    noisy_exptr = noisy_experimenter.NoisyExperimenter(
        exptr=exptr, noise_fn=lambda v: v - 1)

    parameters = exptr.problem_statement().search_space.parameters
    self.assertLen(parameters, dim)

    t = pyvizier.Trial(parameters={
        param.name: float(index) for index, param in enumerate(parameters)
    })

    exptr.evaluate([t])
    metric_name = exptr.problem_statement().metric_information.item().name
    unnoised_value = t.final_measurement_or_die.metrics[metric_name].value

    noisy_exptr.evaluate([t])
    noised_value = t.final_measurement_or_die.metrics[metric_name].value
    self.assertEqual(unnoised_value - 1, noised_value)
    self.assertEqual(
        unnoised_value,
        t.final_measurement_or_die.metrics[metric_name + '_before_noise'].value,
    )

  @parameterized.named_parameters(
      ('NO_NOISE', 'NO_NOISE', 1e-5),
      ('SEVERE_ADDITIVE_GAUSSIAN', 'SEVERE_ADDITIVE_GAUSSIAN', 3),
      ('MODERATE_ADDITIVE_GAUSSIAN', 'MODERATE_ADDITIVE_GAUSSIAN', 0.3),
      ('LIGHT_ADDITIVE_GAUSSIAN', 'LIGHT_ADDITIVE_GAUSSIAN', 0.03),
      ('MODERATE_GAUSSIAN', 'MODERATE_GAUSSIAN', 0.05),
      ('SEVERE_GAUSSIAN', 'SEVERE_GAUSSIAN', 0.5),
      ('MODERATE_UNIFORM', 'MODERATE_UNIFORM', 0.2),
      ('SEVERE_UNIFORM', 'SEVERE_UNIFORM', 3.5),
      ('MODERATE_SELDOM_CAUCHY', 'MODERATE_SELDOM_CAUCHY', 10.3),
      ('SEVERE_SELDOM_CAUCHY', 'SEVERE_SELDOM_CAUCHY', 100.3),
  )
  def testGaussianNoiseApply(self, noise: str, delta: float):
    dim = 2
    exptr = numpy_experimenter.NumpyExperimenter(
        bbob.Sphere, bbob.DefaultBBOBProblemStatement(dim))
    noisy_exptr = noisy_experimenter.NoisyExperimenter.from_type(
        exptr=exptr, noise_type=noise
    )

    parameters = exptr.problem_statement().search_space.parameters
    t = pyvizier.Trial(parameters={
        param.name: float(index) for index, param in enumerate(parameters)
    })

    exptr.evaluate([t])
    metric_name = exptr.problem_statement().metric_information.item().name
    unnoised_value = t.final_measurement_or_die.metrics[metric_name].value

    noisy_exptr.evaluate([t])
    noised_value1 = t.final_measurement_or_die.metrics[metric_name].value

    noisy_exptr.evaluate([t])
    noised_value2 = t.final_measurement_or_die.metrics[metric_name].value

    # Seldom noise is only injected sporadically.
    if 'SELDOM' not in noise and noise != 'NO_NOISE':
      self.assertNotEqual(noised_value1, noised_value2)
    self.assertAlmostEqual(noised_value1, unnoised_value, delta=delta)
    self.assertAlmostEqual(noised_value2, unnoised_value, delta=delta)

  def testSeedDeterminism(self):
    dim = 2
    seed = 7
    exptr = numpy_experimenter.NumpyExperimenter(
        bbob.Sphere, bbob.DefaultBBOBProblemStatement(dim)
    )
    noisy_exptr = noisy_experimenter.NoisyExperimenter.from_type(
        exptr=exptr, noise_type='SEVERE_UNIFORM', seed=seed
    )

    parameters = exptr.problem_statement().search_space.parameters
    t = pyvizier.Trial(
        parameters={
            param.name: float(index) for index, param in enumerate(parameters)
        }
    )
    metric_name = exptr.problem_statement().metric_information.item().name

    noise_value_sequence = []
    for _ in range(10):
      noisy_exptr.evaluate([t])
      noise_value_sequence.append(
          t.final_measurement_or_die.metrics[metric_name].value
      )

    # Global NP seed should not affect randomness.
    np.random.seed(0)

    noisy_exptr = noisy_experimenter.NoisyExperimenter.from_type(
        exptr=exptr, noise_type='SEVERE_UNIFORM', seed=seed
    )
    noise_value_sequence_after = []
    for _ in range(10):
      noisy_exptr.evaluate([t])
      noise_value_sequence_after.append(
          t.final_measurement_or_die.metrics[metric_name].value
      )
    self.assertSequenceAlmostEqual(
        noise_value_sequence, noise_value_sequence_after
    )


if __name__ == '__main__':
  absltest.main()
