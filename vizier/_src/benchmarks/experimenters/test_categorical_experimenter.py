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

"""Tests for categorical_experimenter."""

from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import categorical_experimenter

from absl.testing import absltest


class CategoricalExperimenterTest(absltest.TestCase):
  """Tests for CategoricalExperimenter."""

  def test_evaluate_optimum(self):
    optimum = ['0', '1']
    exptr = categorical_experimenter.CategoricalExperimenter(
        categorical_dim=2, num_categories=3, optimum=optimum)
    suggestion = vz.Trial(parameters={'c0': '0', 'c1': '1'})
    exptr.evaluate([suggestion])
    self.assertEqual(suggestion.final_measurement.metrics['objective'].value, 0)

  def test_evaluate_non_optimum(self):
    optimum = ['0', '1']
    exptr = categorical_experimenter.CategoricalExperimenter(
        categorical_dim=2, num_categories=3, optimum=optimum)
    suggestion = vz.Trial(parameters={'c0': '1', 'c1': '0'})
    exptr.evaluate([suggestion])
    self.assertEqual(suggestion.final_measurement.metrics['objective'].value, 2)

  def test_generate_random_optimum(self):
    exptr = categorical_experimenter.CategoricalExperimenter(
        categorical_dim=20, num_categories=100)
    # pylint: disable=protected-access
    self.assertLen(exptr._optimum, 20)
    # pylint: disable=protected-access
    for value in exptr._optimum.values():
      self.assertIsInstance(value, str)
      self.assertTrue(0 <= int(value) < 100)

  def test_optimum_trial(self):
    optimum = ['0', '1']
    exptr = categorical_experimenter.CategoricalExperimenter(
        categorical_dim=2, num_categories=3, optimum=optimum)
    self.assertEqual(exptr.optimum_trial.parameters['c0'].value, '0')
    self.assertEqual(exptr.optimum_trial.parameters['c1'].value, '1')


if __name__ == '__main__':
  absltest.main()
