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

from vizier import pyvizier
from vizier._src.benchmarks.experimenters import combo_experimenter
from vizier._src.benchmarks.experimenters import normalizing_experimenter
from vizier._src.benchmarks.experimenters import numpy_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob
from vizier._src.benchmarks.experimenters.synthetic import multiarm

from absl.testing import absltest
from absl.testing import parameterized


class NormalizingExperimenterTest(parameterized.TestCase):

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
  def testNormalizationApply(self, func):
    dim = 5
    exptr = numpy_experimenter.NumpyExperimenter(
        func, bbob.DefaultBBOBProblemStatement(dim)
    )
    normalizing_exptr = normalizing_experimenter.NormalizingExperimenter(
        exptr=exptr
    )

    parameters = exptr.problem_statement().search_space.parameters
    self.assertLen(parameters, dim)

    t = pyvizier.Trial(
        parameters={
            param.name: float(index) for index, param in enumerate(parameters)
        }
    )

    exptr.evaluate([t])
    metric_name = exptr.problem_statement().metric_information.item().name

    normalizing_exptr.evaluate([t])
    normalized_value = t.final_measurement_or_die.metrics[metric_name].value
    self.assertBetween(normalized_value, -10, 10)

  def test_NormalizingCategoricals(self):
    mab_exptr = multiarm.FixedMultiArmExperimenter(
        arms_to_rewards={'0': -1e6, '1': 0.0, '2': 1e6}
    )
    norm_exptr = normalizing_experimenter.NormalizingExperimenter(mab_exptr)
    metric_name = norm_exptr.problem_statement().metric_information.item().name

    for arm in range(3):
      t = pyvizier.Trial(parameters={'arm': str(arm)})
      norm_exptr.evaluate([t])
      normalized_value = t.final_measurement_or_die.metrics[metric_name].value
      self.assertBetween(normalized_value, -10, 10)


class HyperCubeExperimenterTest(parameterized.TestCase):

  def testE2E(self):
    num_boolean_params = 3

    original_exptr = combo_experimenter.ContaminationExperimenter(
        contamination_n_stages=num_boolean_params
    )
    hypercube_exptr = normalizing_experimenter.HyperCubeExperimenter(
        original_exptr
    )

    problem = hypercube_exptr.problem_statement()
    metric_name = problem.metric_information.item().name

    # 3 Booleans -> 6 total coordinates after one-hots.
    self.assertLen(problem.search_space.parameters, 2 * num_boolean_params)

    t = pyvizier.Trial(
        parameters={
            'h0': 0.0,
            'h1': 1.0,
            'h2': 1.0,
            'h3': 0.0,
            'h4': 0.3,
            'h5': 0.6,
        }
    )
    hypercube_exptr.evaluate([t])
    hypercube_value = t.final_measurement_or_die.metrics[metric_name].value

    # Argmax over hypercube one-hots.
    t = pyvizier.Trial(
        parameters={'x_0': 'True', 'x_1': 'False', 'x_2': 'True'}
    )
    original_exptr.evaluate([t])
    original_value = t.final_measurement_or_die.metrics[metric_name].value

    self.assertEqual(hypercube_value, original_value)


if __name__ == '__main__':
  absltest.main()
