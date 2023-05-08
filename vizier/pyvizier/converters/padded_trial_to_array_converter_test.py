# Copyright 2023 Google LLC.
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

"""Tests for PaddedTrialToArrayConverter."""

from vizier import pyvizier
from vizier.pyvizier.converters import padded_trial_to_array_converter as pttac
from vizier.pyvizier.converters import padding

from absl.testing import absltest
from absl.testing import parameterized


Trial = pyvizier.Trial


class PaddedTrialToArrayConverterTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(
          num_continuous=6,
          num_integer=1,
          categorical_size=5,
          num_trials=7,
          padding_schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.MULTIPLES_OF_10,
              num_features=padding.PaddingType.POWERS_OF_2,
          ),
          expected_num_trials=10,
          expected_dimension=16,
      ),
      dict(
          num_continuous=2,
          num_integer=4,
          categorical_size=3,
          num_trials=7,
          padding_schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.POWERS_OF_2,
              num_features=padding.PaddingType.MULTIPLES_OF_10,
          ),
          expected_num_trials=8,
          expected_dimension=10,
      ),
  ])
  def test_padding(
      self,
      num_continuous,
      num_integer,
      categorical_size,
      num_trials,
      padding_schedule,
      expected_num_trials,
      expected_dimension,
  ):
    space = pyvizier.SearchSpace()
    root = space.root
    for i in range(num_continuous):
      root.add_float_param(f'double_{i}', 0.0, 100.0)

    for i in range(num_integer):
      root.add_int_param(f'integer_{i}', 0, 100)
    root.add_categorical_param(
        'categorical', [str(k) for k in range(categorical_size)]
    )

    converter = pttac.PaddedTrialToArrayConverter.from_study_config(
        pyvizier.ProblemStatement(
            search_space=space,
            metric_information=[
                pyvizier.MetricInformation(
                    'x1', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
                )
            ],
        ),
        padding_schedule=padding_schedule,
    )
    trials = []
    for i in range(num_trials):
      parameters = {}
      for j in range(num_continuous):
        parameters[f'double_{j}'] = 1.0 * j

      for j in range(num_integer):
        parameters[f'integer_{j}'] = j

      parameters['categorical'] = str(i % categorical_size)
      final_measurement = pyvizier.Measurement(steps=1, metrics={'y': 3 * i})
      trials.append(
          pyvizier.Trial(
              parameters=parameters, final_measurement=final_measurement
          )
      )

    features = converter.to_features(trials).padded_array

    self.assertSequenceEqual(
        features.shape, [expected_num_trials, expected_dimension]
    )

    labels = converter.to_labels(trials).padded_array
    self.assertSequenceEqual(labels.shape, [expected_num_trials, 1])

    features, labels = converter.to_xy(trials)
    features = features.padded_array
    labels = labels.padded_array
    self.assertSequenceEqual(
        features.shape, [expected_num_trials, expected_dimension]
    )
    self.assertSequenceEqual(labels.shape, [expected_num_trials, 1])

    recovered_parameters = converter.to_parameters(features)
    self.assertSequenceEqual(
        recovered_parameters[:num_trials], [t.parameters for t in trials]
    )


if __name__ == '__main__':
  absltest.main()
