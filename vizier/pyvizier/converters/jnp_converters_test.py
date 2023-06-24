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
          input_dims=(11, 7, 3, 1),
          expected_dims=(20, 8, 4, 1),
      ),
      dict(
          input_dims=(21, 3, 0, 2),
          expected_dims=(30, 4, 0, 2),
      ),
  ])
  def test_padding(self, input_dims, expected_dims):
    """Tests all padding schedules.

    Uses multiples of 10, powers of 2, powers of 2, and none padding
    respectively.


    Args:
      input_dims: num_trials, continuous dimensions, categorical dimensions, and
        number of labels.
      expected_dims: num_trials, continuous dimensions, categorical dimensions,
        and number of labels after padding.
    """

    padding_schedule = padding.PaddingSchedule(
        num_trials=padding.PaddingType.MULTIPLES_OF_10,
        num_features=padding.PaddingType.POWERS_OF_2,
        num_metrics=padding.PaddingType.NONE,
    )

    space = pyvizier.SearchSpace()
    trials = [
        pyvizier.Trial().complete(pyvizier.Measurement())
        for _ in range(input_dims[0])
    ]

    root = space.root
    for i in range(input_dims[1]):
      root.add_float_param(f'double_{i}', 0.0, 100.0)
      for t in trials:
        t.parameters[f'double_{i}'] = i

    for i in range(input_dims[2]):
      root.add_categorical_param(f'categorical_{i}', [str(k) for k in range(7)])
      for t in trials:
        t.parameters[f'categorical_{i}'] = str(0)

    problem = pyvizier.ProblemStatement(search_space=space)
    for i in range(input_dims[3]):
      problem.metric_information.append(
          pyvizier.MetricInformation(
              f'y_{i}', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
          )
      )
      for t in trials:
        if t.final_measurement:
          t.final_measurement.metrics[f'y_{i}'] = 3 * i

    converter = jnpc.TrialToModelInputConverter(
        jnpc.TrialToContinuousAndCategoricalConverter.from_study_config(
            problem,
        ),
        padding_schedule=padding_schedule,
    )

    features = converter.to_features(trials)
    self.assertSequenceEqual(
        features.continuous.padded_array.shape,
        [expected_dims[0], expected_dims[1]],
    )
    self.assertSequenceEqual(
        features.categorical.padded_array.shape,
        [expected_dims[0], expected_dims[2]],
    )

    labels = converter.to_labels(trials)
    self.assertSequenceEqual(
        labels.padded_array.shape, [expected_dims[0], expected_dims[3]]
    )

    recovered_parameters = converter.to_parameters(features)
    self.assertSequenceEqual(
        recovered_parameters[: input_dims[0]], [t.parameters for t in trials]
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
