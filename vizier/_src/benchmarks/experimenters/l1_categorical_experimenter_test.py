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

"""Tests for categorical_experimenter."""

from absl.testing import parameterized
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import l1_categorical_experimenter

from absl.testing import absltest


class L1CategoricalExperimenterTest(parameterized.TestCase):
  """Tests for CategoricalExperimenter."""

  def test_evaluate_optimum(self):
    optimum = [0, 1]
    exptr = l1_categorical_experimenter.L1CategorialExperimenter(
        num_categories=[2, 2], optimum=optimum)
    suggestion = vz.Trial(parameters={'c0': '0', 'c1': '1'})
    exptr.evaluate([suggestion])
    self.assertEqual(
        suggestion.final_measurement_or_die.metrics['objective'].value, 0
    )

  def test_evaluate_non_optimum(self):
    optimum = [0, 1]
    exptr = l1_categorical_experimenter.L1CategorialExperimenter(
        num_categories=[2, 2], optimum=optimum)
    suggestion = vz.Trial(parameters={'c0': '1', 'c1': '0'})
    exptr.evaluate([suggestion])
    self.assertEqual(
        suggestion.final_measurement_or_die.metrics['objective'].value, 2
    )

  @parameterized.parameters({'num_categories': [10, 3, 2]},
                            {'num_categories': [10, 2, 10]},
                            {'num_categories': [10, 10, 12, 15, 2]},
                            {'num_categories': [10]})
  def test_generate_random_optimum(self, num_categories):
    exptr = l1_categorical_experimenter.L1CategorialExperimenter(
        num_categories=num_categories)
    self.assertLen(exptr._optimum, len(num_categories))
    for i, value in enumerate(exptr._optimum.values()):
      self.assertIsInstance(value, str)
      self.assertTrue(0 <= int(value) < num_categories[i])

  def test_optimal_trial(self):
    optimum = [9, 2, 1]
    exptr = l1_categorical_experimenter.L1CategorialExperimenter(
        num_categories=[10, 3, 2], optimum=optimum)
    self.assertEqual(exptr.optimal_trial.parameters['c0'].value, '9')
    self.assertEqual(exptr.optimal_trial.parameters['c1'].value, '2')
    self.assertEqual(exptr.optimal_trial.parameters['c2'].value, '1')

  def test_optimal_trial_validation(self):
    optimum = [9, 3, 1]
    with self.assertRaises(ValueError):
      l1_categorical_experimenter.L1CategorialExperimenter(
          num_categories=[10, 3, 2], optimum=optimum)


if __name__ == '__main__':
  absltest.main()
