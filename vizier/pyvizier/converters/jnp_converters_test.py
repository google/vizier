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

"""Tests for jnp_converters."""

import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random as random_designer_lib
from vizier._src.algorithms.testing import test_runners
from vizier._src.jax import types
from vizier.pyvizier.converters import jnp_converters as jnpc
from vizier.pyvizier.converters import padding
from vizier.testing import test_studies

from absl.testing import absltest
from absl.testing import parameterized


Trial = vz.Trial


class PaddedTrialToArrayConverterTest(parameterized.TestCase):

  def test_padding(self):
    """Tests various padding schedules."""

    padding_schedule = padding.PaddingSchedule(
        num_trials=padding.PaddingType.POWERS_OF_2,
        num_features=padding.PaddingType.MULTIPLES_OF_10,
        num_metrics=padding.PaddingType.NONE,
    )
    problem = vz.ProblemStatement(test_studies.flat_space_with_all_types())
    problem.metric_information.append(
        vz.MetricInformation(
            name='metric',
            goal=vz.ObjectiveMetricGoal.MAXIMIZE,
            min_value=-1.0,
            max_value=1.0,
        )
    )
    trials = test_runners.RandomMetricsRunner(
        problem,
        iters=1,
        batch_size=13,
        verbose=False,
        validate_parameters=False,
    ).run_designer(random_designer_lib.RandomDesigner(problem.search_space))

    converter = jnpc.TrialToModelInputConverter.from_problem(
        problem,
        padding_schedule=padding_schedule,
    )

    data = converter.to_xy(trials)
    self.assertSequenceEqual(
        data.features.continuous.padded_array.shape,
        (16, 10),
    )
    self.assertSequenceEqual(
        data.features.categorical.padded_array.shape,
        (16, 10),
    )

    labels = converter.to_labels(trials)
    self.assertSequenceEqual(labels.padded_array.shape, (16, 1))

    recovererd_trials = converter.to_trials(data)

    for i in range(13):
      self.assertSequenceAlmostEqual(
          recovererd_trials[i].parameters.as_dict().values(),
          trials[i].parameters.as_dict().values(),
          places=3,
          msg=f'Failed at {i}th trial',
      )
      problem.search_space.assert_contains(recovererd_trials[i].parameters)


class TrialToContinuousAndCategoricalConverterTest(parameterized.TestCase):
  """Test TrialToContinuousAndCategoricalConverter class."""

  def setUp(self):
    super().setUp()
    self._study_config = vz.ProblemStatement(
        search_space=test_studies.flat_space_with_all_types(),
        metric_information=[
            vz.MetricInformation('x1', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
        ],
    )

    self._designer = random_designer_lib.RandomDesigner(
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
    space = vz.SearchSpace()
    root = space.root
    root.add_float_param('double', -2.0, 2.0)
    root.add_int_param('integer', -2, 2)
    root.add_categorical_param('categorical', ['b', 'c'])
    converter = jnpc.TrialToContinuousAndCategoricalConverter.from_study_config(
        vz.ProblemStatement(search_space=space)
    )
    self.assertEqual(converter.dtype.continuous, np.float64)
    self.assertEqual(converter.dtype.categorical, np.int32)

  def test_parameter_continuify(self):
    space = vz.SearchSpace()
    root = space.root
    root.add_float_param('double', -2.0, 2.0)
    root.add_int_param('integer', -2, 2)
    root.add_categorical_param('categorical', ['b', 'c'])
    root.add_discrete_param('discrete', [-1.0, 2.0, 3.0])

    converter = jnpc.TrialToContinuousAndCategoricalConverter.from_study_config(
        vz.ProblemStatement(search_space=space)
    )
    trial = vz.Trial(
        parameters={
            'double': vz.ParameterValue(3.0),
            'integer': vz.ParameterValue(-1),
            'discrete': vz.ParameterValue(2.0),
            'categorical': vz.ParameterValue('d'),
        }
    )
    expected = types.ContinuousAndCategoricalArray(
        continuous=np.array([[1.25, 0.25, 0.75]]), categorical=np.array([[2]])
    )
    actual = converter.to_features([trial])
    np.testing.assert_equal(actual.continuous, expected.continuous)
    np.testing.assert_equal(actual.categorical, expected.categorical)

  def test_multi_metrics(self):
    search_space = vz.SearchSpace()
    search_space.root.add_float_param('x', 0.0, 1.0)
    problem = vz.ProblemStatement(
        search_space=search_space,
        metric_information=vz.MetricsConfig(
            metrics=[
                vz.MetricInformation(
                    'obj1', goal=vz.ObjectiveMetricGoal.MAXIMIZE
                ),
                vz.MetricInformation(
                    'obj2', goal=vz.ObjectiveMetricGoal.MAXIMIZE
                ),
                vz.MetricInformation(
                    'obj3', goal=vz.ObjectiveMetricGoal.MINIMIZE
                ),
            ]
        ),
    )
    trial1 = vz.Trial()
    trial2 = vz.Trial()
    trial1.final_measurement = vz.Measurement(
        metrics={'obj1': 1.0, 'obj2': 2.0, 'obj3': 3.0}
    )
    trial2.final_measurement = vz.Measurement(
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
