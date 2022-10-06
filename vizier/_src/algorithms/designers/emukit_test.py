# Copyright 2022 Google LLC.
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

"""Tests for emukit."""

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import emukit
from vizier._src.algorithms.testing import test_runners
from vizier.testing import test_studies

from absl.testing import absltest
from absl.testing import parameterized


class EmukitTest(parameterized.TestCase):

  @parameterized.parameters(((v,) for v in emukit.Version))
  def test_on_flat_space(self, version):
    config = vz.ProblemStatement(
        search_space=test_studies.flat_space_with_all_types(),
        metric_information=[
            vz.MetricInformation(
                name='x1', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
        ])
    designer = emukit.EmukitDesigner(
        config, num_random_samples=10, metadata_ns='emukit', version=version)
    trials = test_runners.run_with_random_metrics(
        designer,
        config,
        iters=15,
        batch_size=1,
        verbose=1,
        validate_parameters=True)
    self.assertLen(trials, 15)

    for t in trials[:10]:
      self.assertEqual(
          t.metadata.ns('emukit')['source'],
          'random',
          msg=f'Trial {t} should be generated at random.')

    for t in trials[10:]:
      self.assertEqual(
          t.metadata.ns('emukit')['source'],
          'bayesopt',
          msg=f'Trial {t} should be generated by bayesopt.')


if __name__ == '__main__':
  absltest.main()
