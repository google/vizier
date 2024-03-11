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

import copy

from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import noisy_experimenter
from vizier._src.benchmarks.experimenters import numpy_experimenter
from vizier._src.benchmarks.experimenters import sign_flip_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest
from absl.testing import parameterized


# TODO: Add multimetric test once we have a multi-objective
# experimenter.
class SignFlipExperimenterTest(parameterized.TestCase):

  @parameterized.parameters((True,), (False))
  def test_flipped_bbob(self, flip_objectives_only: bool):
    exptr = numpy_experimenter.NumpyExperimenter(
        bbob.Sphere, bbob.DefaultBBOBProblemStatement(2)
    )
    exptr = noisy_experimenter.NoisyExperimenter(
        exptr, noise_fn=lambda v: v - 1
    )
    flipped_exptr = sign_flip_experimenter.SignFlipExperimenter(
        exptr, flip_objectives_only
    )

    metric_name = exptr.problem_statement().single_objective_metric_name
    self.assertEqual(
        flipped_exptr.problem_statement().metric_information.item().goal,
        vz.ObjectiveMetricGoal.MAXIMIZE,
    )

    suggestion_for_original = vz.Trial(parameters={'x0': 0.2, 'x1': -3.2})
    suggestion_for_flipped = copy.deepcopy(suggestion_for_original)

    exptr.evaluate([suggestion_for_original])
    flipped_exptr.evaluate([suggestion_for_flipped])

    self.assertEqual(
        suggestion_for_original.final_measurement_or_die.metrics[
            metric_name
        ].value,
        -1.0
        * suggestion_for_flipped.final_measurement_or_die.metrics[
            metric_name
        ].value,
    )

    aux_metric_name = metric_name + '_before_noise'
    if flip_objectives_only:
      self.assertEqual(
          suggestion_for_original.final_measurement_or_die.metrics[
              aux_metric_name
          ].value,
          suggestion_for_flipped.final_measurement_or_die.metrics[
              aux_metric_name
          ].value,
      )
    else:
      self.assertEqual(
          suggestion_for_original.final_measurement_or_die.metrics[
              aux_metric_name
          ].value,
          -1.0
          * suggestion_for_flipped.final_measurement_or_die.metrics[
              aux_metric_name
          ].value,
      )


if __name__ == '__main__':
  absltest.main()
