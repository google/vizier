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

"""Tests for jnp_converters."""

import numpy as np
from vizier import pyvizier
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.testing import test_runners
from vizier._src.jax import types
from vizier.pyvizier.converters import jnp_converters as jnpc
from vizier.pyvizier.converters import padding
from vizier.testing import test_studies

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

    converter = jnpc.PaddedTrialToArrayConverter.from_study_config(
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


class TrialToContinuousAndCategoricalConverterTest(parameterized.TestCase):
  """Test TrialToContinuousAndCategoricalConverter class."""

  def setUp(self):
    super().setUp()
    self._study_config = pyvizier.ProblemStatement(
        search_space=test_studies.flat_space_with_all_types(),
        metric_information=[
            pyvizier.MetricInformation(
                'x1', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
            )
        ],
    )

    self._designer = random.RandomDesigner(
        self._study_config.search_space, seed=0
    )
    self._trials = test_runners.run_with_random_metrics(
        self._designer, self._study_config, iters=1, batch_size=10
    )
    self.maxDiff = None

  def test_back_to_back_conversion(self):
    converter = jnpc.TrialToContinuousAndCategoricalConverter.from_study_config(
        self._study_config
    )
    self.assertSequenceEqual(
        [t.parameters for t in self._trials],
        converter.to_parameters(converter.to_features(self._trials)),
    )

  def test_dtype(self):
    space = pyvizier.SearchSpace()
    root = space.root
    root.add_float_param('double', -2.0, 2.0)
    root.add_int_param('integer', -2, 2)
    root.add_categorical_param('categorical', ['b', 'c'])
    converter = jnpc.TrialToContinuousAndCategoricalConverter.from_study_config(
        pyvizier.ProblemStatement(search_space=space)
    )
    self.assertEqual(converter.dtype.continuous, np.float64)
    self.assertEqual(converter.dtype.categorical, np.int32)

  def test_parameter_continuify(self):
    space = pyvizier.SearchSpace()
    root = space.root
    root.add_float_param('double', -2.0, 2.0)
    root.add_int_param('integer', -2, 2)
    root.add_categorical_param('categorical', ['b', 'c'])
    root.add_discrete_param('discrete', [-1.0, 2.0, 3.0])

    converter = jnpc.TrialToContinuousAndCategoricalConverter.from_study_config(
        pyvizier.ProblemStatement(search_space=space)
    )
    trial = pyvizier.Trial(
        parameters={
            'double': pyvizier.ParameterValue(3.0),
            'integer': pyvizier.ParameterValue(-1),
            'discrete': pyvizier.ParameterValue(2.0),
            'categorical': pyvizier.ParameterValue('d'),
        }
    )
    expected = types.ContinuousAndCategoricalArray(
        continuous=np.array([[1.25, 0.25, 0.75]]), categorical=np.array([[2]])
    )
    actual = converter.to_features([trial])
    np.testing.assert_equal(actual.continuous, expected.continuous)
    np.testing.assert_equal(actual.categorical, expected.categorical)

  def test_multi_metrics(self):
    search_space = pyvizier.SearchSpace()
    search_space.root.add_float_param('x', 0.0, 1.0)
    problem = pyvizier.ProblemStatement(
        search_space=search_space,
        metric_information=pyvizier.MetricsConfig(
            metrics=[
                pyvizier.MetricInformation(
                    'obj1', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
                ),
                pyvizier.MetricInformation(
                    'obj2', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
                ),
                pyvizier.MetricInformation(
                    'obj3', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
                ),
            ]
        ),
    )
    trial1 = pyvizier.Trial()
    trial2 = pyvizier.Trial()
    trial1.final_measurement = pyvizier.Measurement(
        metrics={'obj1': 1.0, 'obj2': 2.0, 'obj3': 3.0}
    )
    trial2.final_measurement = pyvizier.Measurement(
        metrics={'obj1': -1.0, 'obj2': 5.0, 'obj3': 0.0}
    )
    converter = jnpc.TrialToContinuousAndCategoricalConverter.from_study_config(
        problem
    )
    # Notice that the sign is flipped for MINIMIZE objective.
    expected_labels = np.array([[1.0, 2.0, -3.0], [-1.0, 5.0, 0.0]])
    np.testing.assert_equal(
        converter.to_labels([trial1, trial2]), expected_labels
    )
    self.assertEqual(converter.metric_specs[0].name, 'obj1')
    self.assertEqual(converter.metric_specs[1].name, 'obj2')
    self.assertEqual(converter.metric_specs[2].name, 'obj3')


if __name__ == '__main__':
  absltest.main()
