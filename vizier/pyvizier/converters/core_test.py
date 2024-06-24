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

"""Tests for core."""

from absl import logging
import numpy as np
from vizier import pyvizier
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.testing import test_runners
from vizier.pyvizier.converters import core
from vizier.testing import test_studies

from absl.testing import absltest
from absl.testing import parameterized


Trial = pyvizier.Trial


class TrialToArrayConverterConditionalSpaceTest(parameterized.TestCase):

  def test_automl_study(self):
    space = test_studies.conditional_automl_space()
    converter = core.TrialToArrayConverter.from_study_config(
        pyvizier.ProblemStatement(search_space=space)
    )
    features = converter.to_features([pyvizier.Trial()])
    np.testing.assert_equal(
        features,
        np.array(
            [[0.0, 0.0, 1.0, np.nan, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, np.nan]]
        ),
    )


class TrialToArrayConverterTest(parameterized.TestCase):
  """Test TrialToArrayConverter class."""

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
    converter = core.TrialToArrayConverter.from_study_config(self._study_config)
    self.assertSequenceEqual(
        [t.parameters for t in self._trials],
        converter.to_parameters(converter.to_features(self._trials)),
    )

  def test_dtype(self):
    space = pyvizier.SearchSpace()
    root = space.root
    root.add_float_param('double', -2.0, 2.0)
    root.add_int_param('integer', -2, 2)
    converter = core.TrialToArrayConverter.from_study_config(
        pyvizier.ProblemStatement(search_space=space)
    )
    self.assertEqual(converter.dtype, np.float64)

  def test_parameter_continuify(self):
    space = pyvizier.SearchSpace()
    root = space.root
    root.add_float_param('double', -2.0, 2.0)
    root.add_int_param('integer', -2, 2)
    root.add_categorical_param('categorical', ['b', 'c'])
    root.add_discrete_param('discrete', [-1.0, 2.0, 3.0])

    converter = core.TrialToArrayConverter.from_study_config(
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
    # Scaled outputs + 1 hot embedding for categoricals.
    expected = np.array([[1.25, 0.25, 0, 0, 1, 0.75]])
    np.testing.assert_equal(converter.to_features([trial]), expected)

  def test_onehot_embedding(self):
    space = pyvizier.SearchSpace()
    root = space.root
    root.add_categorical_param('c1', ['0', '1', '2'])
    root.add_categorical_param('c2', ['0', '1', '2', '3'])

    converter = core.TrialToArrayConverter.from_study_config(
        pyvizier.ProblemStatement(search_space=space)
    )
    n_trials = 10
    n_repeat_check = 50
    for _ in range(n_repeat_check):
      trials = []
      for _ in range(n_trials):
        trials.append(
            pyvizier.Trial(
                parameters={
                    'c1': pyvizier.ParameterValue(str(np.random.randint(0, 3))),
                    'c2': pyvizier.ParameterValue(str(np.random.randint(0, 4))),
                }
            )
        )
      features = converter.to_features(trials)
      np.testing.assert_equal(
          np.sum(features, axis=1), np.array([2] * n_trials)
      )

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
    converter = core.TrialToArrayConverter.from_study_config(problem)
    # Notice that the sign is flipped for MINIMIZE objective.
    expected_labels = np.array([[1.0, 2.0, -3.0], [-1.0, 5.0, 0.0]])
    np.testing.assert_equal(
        converter.to_labels([trial1, trial2]), expected_labels
    )
    self.assertEqual(converter.metric_specs[0].name, 'obj1')
    self.assertEqual(converter.metric_specs[1].name, 'obj2')
    self.assertEqual(converter.metric_specs[2].name, 'obj3')


class DictToArrayTest(absltest.TestCase):

  def test_2d(self):
    self.assertSequenceEqual(
        core.dict_to_array(
            {1: np.random.random((10, 2)), 2: np.random.random((10, 1))}
        ).shape,
        (10, 3),
    )

  def test_4d(self):
    self.assertSequenceEqual(
        core.dict_to_array({
            1: np.random.random((2, 1, 10, 2)),
            2: np.random.random((2, 1, 10, 1)),
        }).shape,
        (2, 1, 10, 3),
    )

  def test_dict_like(self):
    d = core.DictOf2DArrays({
        'p1': np.array([[1], [2], [3]]),
        'p2': np.array([[0.0, 0.1], [0.5, 0.5], [0.1, 0.0]]),
    })
    self.assertCountEqual(
        d.dict_like(np.array([[1, 1, 0.1], [3, 0, 0.1]])),
        {'p1': np.array([[1], [3]]), 'p2': np.array([[1, 0.1], [0, 0.1]])},
    )

  def test_dict_like_and_as_array_are_inverse(self):
    d = core.DictOf2DArrays({
        'p1': np.array([[1], [2], [3]]),
        'p2': np.array([[0.0, 0.1], [0.5, 0.5], [0.1, 0.0]]),
    })
    self.assertCountEqual(d, d.dict_like(d.asarray()))


rnd = np.random.random


class DictOf2DArraysTest(absltest.TestCase):

  def test_bad_shapes(self):
    with self.assertRaises(ValueError):
      core.DictOf2DArrays({'a': rnd([10, 2]), 'b': rnd([5, 2])})

  def test_bad_dim(self):
    with self.assertRaises(ValueError):
      core.DictOf2DArrays({'a': rnd([10, 2, 1])})

  def test_size(self):
    d = core.DictOf2DArrays({'a': rnd([10, 2]), 'b': rnd([10, 1])})
    self.assertEqual(d.size, 10)

  def test_addition(self):
    d1 = core.DictOf2DArrays({'a': rnd([10, 2]), 'b': rnd([10, 1])})
    d2 = core.DictOf2DArrays({'a': rnd([5, 2]), 'b': rnd([5, 1])})
    d = d1 + d2
    self.assertEqual(d.size, 15)
    self.assertLen(d, 2)
    self.assertSequenceEqual(d['a'].shape, (15, 2))
    self.assertSequenceEqual(d['b'].shape, (15, 1))

  def test_bad_addition(self):
    # Shape of b doesn't match.
    d1 = core.DictOf2DArrays({'a': rnd([10, 2]), 'b': rnd([10, 1])})
    d2 = core.DictOf2DArrays({'a': rnd([5, 2]), 'b': rnd([5, 3])})
    with self.assertRaises(ValueError):
      _ = d1 + d2


class DefaultTrialConverterFromStudyConfigsTest(absltest.TestCase):

  @property
  def _study_configs(self):
    study_configs = []
    space1 = pyvizier.SearchSpace()
    root = space1.root
    root.add_float_param('double', -1.0, 1.0)
    root.add_int_param('integer', -1, 1)
    root.add_categorical_param('categorical', ['a', 'b'])
    root.add_discrete_param('discrete', [-1.0, 1.0])
    study_configs.append(
        pyvizier.ProblemStatement(
            metadata=pyvizier.Metadata({core.STUDY_ID_FIELD: 'study1'}),
            search_space=space1,
        )
    )

    space2 = pyvizier.SearchSpace()
    root = space2.root
    root.add_float_param('double', -1.0, 2.0)
    root.add_int_param('integer', -2, 1)
    root.add_categorical_param('categorical', ['b', 'c'])
    root.add_discrete_param('discrete', [-1.0, 1.0, 2.0])
    study_configs.append(
        pyvizier.ProblemStatement(
            metadata=pyvizier.Metadata({core.STUDY_ID_FIELD: 'study2'}),
            search_space=space2,
        )
    )
    return study_configs

  def test_parameters(self):
    converter = core.DefaultTrialConverter.from_study_configs(
        self._study_configs, [], use_study_id_feature=True
    )
    self.assertCountEqual(
        ['study1', 'study2'],
        converter.parameter_configs[core.STUDY_ID_FIELD].feasible_values,
    )
    self.assertEqual((-1.0, 2.0), converter.parameter_configs['double'].bounds)
    self.assertEqual((-2, 1), converter.parameter_configs['integer'].bounds)
    self.assertCountEqual(
        ('a', 'b', 'c'),
        converter.parameter_configs['categorical'].feasible_values,
    )
    self.assertCountEqual(
        (-1.0, 1.0, 2.0),
        converter.parameter_configs['discrete'].feasible_values,
    )

    expected = {
        'metalearn_study_id': (None, 1),
        'double': (None, 1),
        'integer': (None, 1),
        'categorical': (None, 1),
        'discrete': (None, 1),
    }
    self.assertDictEqual(converter.features_shape, expected)

    trial = pyvizier.Trial(
        parameters={
            'double': pyvizier.ParameterValue(3.0),
            'integer': pyvizier.ParameterValue(-1),
            'categorical': pyvizier.ParameterValue('d'),
        }
    )
    expected = {
        'categorical': np.array([[3]]),
        'discrete': np.array([[3]]),
        'double': np.array([[3.0]]),
        'integer': np.array([[1]]),
        'metalearn_study_id': np.array([[2]]),
    }
    self.assertDictEqual(converter.to_features([trial]), expected)

    # The value for `categorical` was invalid, so we can't recover it.
    expected_parameters = pyvizier.ParameterDict({
        'double': pyvizier.ParameterValue(2.0),
        'integer': pyvizier.ParameterValue(-1),
    })
    self.assertEqual(
        converter.to_trials(expected)[0].parameters, expected_parameters
    )

  def test_parameters_and_labels(self):
    study_config = pyvizier.ProblemStatement()
    root = study_config.search_space.root
    root.add_float_param('x1', 0.0, 10.0)
    root.add_float_param('x2', 0.0, 10.0)

    study_config.metric_information.extend([
        pyvizier.MetricInformation(
            name='y1', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
        ),
        pyvizier.MetricInformation(
            name='y2', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
        pyvizier.MetricInformation(
            name='y3', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
    ])
    converter = core.DefaultTrialConverter.from_study_config(study_config)
    actual_features = {
        'x1': np.array([[1.0, 3.0]]).T,
        'x2': np.array([[2.0, 4.0]]).T,
    }
    actual_labels = {
        'y1': np.array([[10.0, 40.0]]).T,
        'y2': np.array([[20.0, 50.0]]).T,
        'y3': np.array([[np.nan, 60.0]]).T,
    }
    actual_trials = [
        pyvizier.Trial(
            parameters={
                'x1': pyvizier.ParameterValue(1.0),
                'x2': pyvizier.ParameterValue(2.0),
            },
            final_measurement=pyvizier.Measurement(
                steps=1, metrics={'y1': 10.0, 'y2': 20.0}
            ),
            creation_time=None,
        ),
        pyvizier.Trial(
            parameters={
                'x1': pyvizier.ParameterValue(3.0),
                'x2': pyvizier.ParameterValue(4.0),
            },
            final_measurement=pyvizier.Measurement(
                steps=1, metrics={'y1': 40.0, 'y2': 50.0, 'y3': 60.0}
            ),
            creation_time=None,
        ),
    ]
    trials = converter.to_trials(actual_features, actual_labels)
    for t in trials:
      t.creation_time = None
    self.assertEqual(
        actual_trials,
        trials,
        msg='conversion from features and labels to trials failed.',
    )
    features, labels = converter.to_xy(actual_trials)
    np.testing.assert_equal(
        actual_features,
        features,
        err_msg='conversion from trials to features failed.',
    )
    np.testing.assert_equal(
        actual_labels,
        labels,
        err_msg='conversion from trials to lables failed.',
    )
    features_, labels_ = converter.to_xy(trials)
    np.testing.assert_equal(
        actual_features,
        features_,
        err_msg='roundtrip conversion for features failed.',
    )
    np.testing.assert_equal(
        actual_labels,
        labels_,
        err_msg='roundtrip conversion for labels failed.',
    )

  def test_infeasible_trials(self):
    labels = np.array([np.nan, np.nan, -100, -200, 1, 2, 3, 4])[:, np.newaxis]
    features = np.arange(labels.shape[0])[:, np.newaxis]
    study_config = pyvizier.ProblemStatement(
        metric_information=[
            pyvizier.MetricInformation(
                'y', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    study_config.search_space.root.add_float_param('x', 0, 20)
    converter = core.DefaultTrialConverter.from_study_config(study_config)
    trials = converter.to_trials({'x': features}, {'y': labels})
    for t in trials:
      assert t.final_measurement is not None
      if not t.final_measurement.metrics:
        self.assertTrue(t.infeasible)
    for label, t in zip(labels.flatten(), trials):
      if np.isnan(label):
        self.assertEmpty(t.final_measurement.metrics)

  def test_parameters_and_labels_shape(self):
    study_config = pyvizier.ProblemStatement()
    root = study_config.search_space.root
    root.add_float_param('x1', 0.0, 10.0)

    study_config.metric_information.extend([
        pyvizier.MetricInformation(
            name='y1', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
        ),
        pyvizier.MetricInformation(
            name='y2', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
    ])
    converter = core.DefaultTrialConverter.from_study_config(study_config)
    features_1d = {
        'x1': np.array([1.0, 3.0]),
        'x2': np.array([2.0, 4.0]),
    }

    with self.assertRaisesRegex(
        ValueError,
        'Features need to contain 2d arrays with shape (num_obs, 1)*',
    ):
      converter.to_trials(features_1d)

    features_right_shape = {
        'x1': np.array([[1.0, 3.0]]).T,
        'x2': np.array([[2.0, 4.0]]).T,
    }
    labels_1d = {
        'y1': np.array([10.0, 40.0]),
        'y2': np.array([20.0, 50.0]),
    }
    with self.assertRaisesRegex(
        ValueError, 'Labels need to contain 2d arrays with shape (num_obs, 1)*'
    ):
      converter.to_trials(features_right_shape, labels_1d)

    features_2d_reverse_shape = {
        'x1': np.array([[1.0, 3.0]]),
        'x2': np.array([[2.0, 4.0]]),
    }
    with self.assertRaisesRegex(
        ValueError,
        'Features need to contain 2d arrays with shape (num_obs, 1)*',
    ):
      converter.to_trials(features_2d_reverse_shape)

    labels_2d_reverse_shape = {
        'y1': np.array([[10.0, 40.0]]),
        'y2': np.array([[20.0, 50.0]]),
    }
    with self.assertRaisesRegex(
        ValueError, 'Labels need to contain 2d arrays with shape (num_obs, 1)*'
    ):
      converter.to_trials(features_right_shape, labels_2d_reverse_shape)

  def test_metrics(self):
    converter = core.DefaultTrialConverter.from_study_configs(
        [],
        [
            pyvizier.MetricInformation(
                name='metric1', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
            )
        ],
        use_study_id_feature=False,
    )

    expected = {'metric1': (None, 1)}
    self.assertDictEqual(converter.labels_shape, expected)


class DefaultModelOutputConverterTest(parameterized.TestCase):

  @property
  def _measurements(self):
    return [
        pyvizier.Measurement(
            metrics={
                'metric1': pyvizier.Metric(1.0),
                'metric2': pyvizier.Metric(1.1),
            }
        ),
        pyvizier.Measurement(
            metrics={
                'metric1': pyvizier.Metric(2.0),
                'metric2': pyvizier.Metric(2.1),
            }
        ),
        pyvizier.Measurement(
            metrics={
                'metric1': pyvizier.Metric(4.0),
                'metric2': pyvizier.Metric(4.1),
            }
        ),
    ]

  def test_empty(self):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
        flip_sign_for_minimization_metrics=True,
    )
    np.testing.assert_array_equal(
        converter.convert([]), np.zeros([0, 1], dtype=converter.dtype)
    )

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_sign_flips(self, dtype):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
        flip_sign_for_minimization_metrics=True,
        dtype=dtype,
    )
    actual = converter.convert(self._measurements)

    expected = -np.asarray([[1.0], [2.0], [4.0]], dtype)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)
    self.assertEqual(
        converter.metric_information,
        pyvizier.MetricInformation(
            name='metric1', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
        ),
    )

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_no_sign_flips(self, dtype):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric2', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
        flip_sign_for_minimization_metrics=False,
        dtype=dtype,
    )
    actual = converter.convert(self._measurements)

    expected = np.asarray([[1.1], [2.1], [4.1]], dtype)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)
    self.assertEqual(
        converter.metric_information,
        pyvizier.MetricInformation(
            name='metric2', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
    )

  def test_shift_threshold(self):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric2',
            goal=pyvizier.ObjectiveMetricGoal.MINIMIZE,
            safety_threshold=5.0,
        ),
        flip_sign_for_minimization_metrics=False,
        dtype=float,
    )
    converter.shift_safe_metrics = False
    self.assertEqual(5.0, converter.metric_information.safety_threshold)
    converter.shift_safe_metrics = True
    self.assertEqual(0.0, converter.metric_information.safety_threshold)

  def test_raise_errors_for_missing_metrics(self):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric3', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
        flip_sign_for_minimization_metrics=False,
        raise_errors_for_missing_metrics=True,
    )
    with self.assertRaises(KeyError):
      converter.convert(self._measurements)

  def test_do_not_raise_errors_for_missing_metrics(self):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric3', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
        flip_sign_for_minimization_metrics=False,
        raise_errors_for_missing_metrics=False,
    )
    np.testing.assert_equal(
        converter.convert(self._measurements), np.asarray([[np.nan]] * 3)
    )

  @parameterized.parameters([dict(flip_sign=True), dict(flip_sign=False)])
  def test_safe_maximize_metric(self, flip_sign):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1',
            goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE,
            safety_threshold=3.0,
        ),
        flip_sign_for_minimization_metrics=flip_sign,
    )
    actual = converter.convert(self._measurements)
    np.testing.assert_equal(actual, [[-2.0], [-1.0], [1.0]])

  @parameterized.parameters([dict(flip_sign=True), dict(flip_sign=False)])
  def test_safe_minimize_metric(self, flip_sign):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1',
            goal=pyvizier.ObjectiveMetricGoal.MINIMIZE,
            safety_threshold=3.0,
        ),
        flip_sign_for_minimization_metrics=flip_sign,
    )
    actual = converter.convert(self._measurements)
    expected = np.array([[-2.0], [-1.0], [1.0]], dtype=np.float32) * (
        -1 if flip_sign else 1
    )
    np.testing.assert_equal(actual, expected)

  @parameterized.parameters([
      dict(flip_sign=True, raise_error=True),
      dict(flip_sign=False, raise_error=True),
      dict(flip_sign=True, raise_error=False),
      dict(flip_sign=False, raise_error=False),
  ])
  def test_to_metrics(self, flip_sign: bool, raise_error: bool):
    expected = [
        pyvizier.Metric(1.0),
        pyvizier.Metric(2.0),
        None,
        pyvizier.Metric(4.0),
        None,
    ]
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
        flip_sign_for_minimization_metrics=flip_sign,
        raise_errors_for_missing_metrics=raise_error,
        dtype=float,
    )
    arr = np.array([[1.0], [2.0], [np.nan], [4.0], [np.nan]]) * (
        -1 if flip_sign else 1
    )
    actual = converter.to_metrics(arr)
    self.assertEqual(actual, expected)

  @parameterized.parameters([
      dict(flip_sign=True, raise_error=True, safety_threshold=5.0),
      dict(flip_sign=False, raise_error=True, safety_threshold=5.0),
      dict(flip_sign=True, raise_error=False, safety_threshold=5.0),
      dict(flip_sign=False, raise_error=False, safety_threshold=5.0),
      dict(flip_sign=True, raise_error=True, safety_threshold=-5.0),
      dict(flip_sign=False, raise_error=True, safety_threshold=-5.0),
      dict(flip_sign=True, raise_error=False, safety_threshold=-5.0),
      dict(flip_sign=False, raise_error=False, safety_threshold=-5.0),
  ])
  def test_to_safe_minimize_metrics_parametrized(
      self, flip_sign: bool, raise_error: bool, safety_threshold: float
  ):
    expected = [
        pyvizier.Metric((1.0 - safety_threshold) * (-1 if flip_sign else 1)),
        pyvizier.Metric((2.0 - safety_threshold) * (-1 if flip_sign else 1)),
        None,
        pyvizier.Metric((4.0 - safety_threshold) * (-1 if flip_sign else 1)),
        None,
    ]
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1',
            goal=pyvizier.ObjectiveMetricGoal.MINIMIZE,
            safety_threshold=safety_threshold,
        ),
        flip_sign_for_minimization_metrics=flip_sign,
        raise_errors_for_missing_metrics=raise_error,
        dtype=float,
    )
    arr = np.array([[1.0], [2.0], [np.nan], [4.0], [np.nan]])
    actual = converter.to_metrics(arr)
    self.assertEqual(actual, expected)

  def test_to_safe_maximize_metrics(self):
    expected = [
        pyvizier.Metric(-4.0),
        pyvizier.Metric(-3.0),
        None,
        pyvizier.Metric(-1.0),
        None,
    ]
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1',
            goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE,
            safety_threshold=5.0,
        ),
        dtype=float,
    )
    arr = np.array([[1.0], [2.0], [np.nan], [4.0], [np.nan]])
    actual = converter.to_metrics(arr)
    self.assertEqual(actual, expected)

  def test_to_safe_minimize_metrics(self):
    expected = [
        pyvizier.Metric(4.0),
        pyvizier.Metric(3.0),
        None,
        pyvizier.Metric(1.0),
        None,
    ]
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1',
            goal=pyvizier.ObjectiveMetricGoal.MINIMIZE,
            safety_threshold=5.0,
        ),
        flip_sign_for_minimization_metrics=True,
        dtype=float,
    )
    arr = np.array([[1.0], [2.0], [np.nan], [4.0], [np.nan]])
    actual = converter.to_metrics(arr)
    self.assertEqual(actual, expected)

  @parameterized.parameters([
      dict(flip_sign=True, safety_threshold=5.0),
      dict(flip_sign=True, safety_threshold=-5.0),
  ])
  def test_first_shift_then_flip(
      self, flip_sign: bool, safety_threshold: float
  ):
    not_expected = [
        pyvizier.Metric((1.0 * (-1 if flip_sign else 1)) - safety_threshold),
        pyvizier.Metric((2.0 * (-1 if flip_sign else 1)) - safety_threshold),
        None,
        pyvizier.Metric((4.0 * (-1 if flip_sign else 1)) - safety_threshold),
        None,
    ]
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1',
            goal=pyvizier.ObjectiveMetricGoal.MINIMIZE,
            safety_threshold=safety_threshold,
        ),
        flip_sign_for_minimization_metrics=flip_sign,
        dtype=float,
    )
    arr = np.array([[1.0], [2.0], [np.nan], [4.0], [np.nan]])
    actual = converter.to_metrics(arr)
    self.assertNotEqual(actual, not_expected)

  @parameterized.parameters([
      dict(flip_sign=True, raise_error=True),
      dict(flip_sign=False, raise_error=True),
      dict(flip_sign=True, raise_error=False),
      dict(flip_sign=False, raise_error=False),
  ])
  def test_to_metrics_from_measurements(
      self, flip_sign: bool, raise_error: bool
  ):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric2', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
        flip_sign_for_minimization_metrics=flip_sign,
        raise_errors_for_missing_metrics=raise_error,
        dtype=float,
    )
    measurements = self._measurements
    ys = converter.convert(measurements)
    ms = converter.to_metrics(ys)
    from_ms = np.array(
        [measurements[i].metrics['metric2'].value for i in range(len(ms))]
    )[:, None] * (-1 if flip_sign else 1)
    self.assertTrue((from_ms == ys).all())

  @parameterized.parameters(
      [dict(labels=np.array([[1, 2]])), dict(labels=np.array([[[1, 2]]]))]
  )
  def test_bad_labels(self, labels: np.ndarray):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric2', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
        flip_sign_for_minimization_metrics=True,
        raise_errors_for_missing_metrics=True,
        dtype=float,
    )
    with self.assertRaises(ValueError):
      converter.to_metrics(labels)


class DefaultModelInputConverterTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_double_into_double(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', bounds=(-3.0, 3.0)),
        scale=False,
        onehot_embed=True,
        float_dtype=dtype,
    )

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial(),
    ])
    expected = np.asarray([[1.0], [2.0], [-3.0], [np.nan], [np.nan]], dtype)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_double_into_double_log(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1', bounds=(1e-4, 1e2), scale_type=pyvizier.ScaleType.LOG
        ),
        scale=True,
        float_dtype=dtype,
    )

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1e-4)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(1e2)}),
    ])
    expected = np.asarray([[0.0], [1.0]], dtype)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_double_into_double_log_inverse(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1', bounds=(1e-4, 1e2), scale_type=pyvizier.ScaleType.LOG
        ),
        scale=True,
        float_dtype=dtype,
    )

    scaled = np.asarray([[0.0], [0.5], [1.1]], dtype)
    # Pytype still thinks `actual` entries might be None, hence we specify type.
    actual: list[pyvizier.ParameterValue] = converter.to_parameter_values(  # pytype:disable=annotation-type-mismatch
        scaled
    )
    self.assertAlmostEqual(actual[0].value, 1e-4, delta=1e-6)
    self.assertGreaterEqual(actual[0].value, 1e-4)
    self.assertAlmostEqual(actual[1].value, 0.1, delta=1e-5)
    self.assertAlmostEqual(actual[2].value, 100.0, delta=1e-2)
    self.assertLessEqual(actual[2].value, 100.0)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_double_into_double_log_inverse_noclipping(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1', bounds=(1e-4, 1e2), scale_type=pyvizier.ScaleType.LOG
        ),
        scale=True,
        float_dtype=dtype,
        should_clip=False,
    )

    scaled = np.asarray([[0.0], [0.5], [1.1]], dtype)
    # Pytype still thinks `actual` entries might be None, hence we specify type.
    actual: list[pyvizier.ParameterValue] = converter.to_parameter_values(  # pytype:disable=annotation-type-mismatch
        scaled
    )
    self.assertAlmostEqual(actual[0].value, 1e-4, delta=1e-6)
    self.assertAlmostEqual(actual[1].value, 0.1, delta=1e-5)
    self.assertAlmostEqual(actual[2].value, 398, delta=1)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_double_into_double_reverse_log(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1', bounds=(1e-4, 1e2), scale_type=pyvizier.ScaleType.REVERSE_LOG
        ),
        scale=True,
        float_dtype=dtype,
    )

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1e-4)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(1.0)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(1e2)}),
    ])
    expected = np.asarray([[0.0], [7.273945e-4], [1.0]], dtype)
    np.testing.assert_allclose(expected, actual, rtol=1e-3)
    self.assertEqual(expected.dtype, actual.dtype)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_double_into_double_reverse_log_inverse(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1', bounds=(1e-4, 1e2), scale_type=pyvizier.ScaleType.REVERSE_LOG
        ),
        scale=True,
        float_dtype=dtype,
    )

    scaled = np.asarray([[0.0], [0.5], [1.0]], dtype)
    # Pytype still thinks `actual` entries might be None, hence we specify type.
    actual: list[pyvizier.ParameterValue] = converter.to_parameter_values(  # pytype:disable=annotation-type-mismatch
        scaled
    )
    self.assertGreaterEqual(actual[0].value, 1e-4)
    self.assertGreaterEqual(actual[1].value, 0.5)
    self.assertLessEqual(actual[2].value, 1e2)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_zero_range_linear_double(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1', bounds=(0.9, 0.9), scale_type=pyvizier.ScaleType.LINEAR
        ),
        scale=True,
        onehot_embed=True,
        float_dtype=dtype,
    )
    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(0.9)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(1.0)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial(),
    ])
    expected = np.asarray([[0.5], [0.6], [np.nan], [np.nan]], dtype)
    np.testing.assert_equal(expected, actual)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_zero_range_log_double(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1',
            bounds=(np.exp(0.9), np.exp(0.9)),
            scale_type=pyvizier.ScaleType.LOG,
        ),
        scale=True,
        onehot_embed=True,
        float_dtype=dtype,
    )
    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(np.exp(0.9))}),
        Trial(parameters={'x1': pyvizier.ParameterValue(np.exp(1.2))}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial(),
    ])
    expected = np.asarray(
        [[0.5], [0.5 + np.exp(1.2) - np.exp(0.9)], [np.nan], [np.nan]],
        dtype=dtype,
    )
    np.testing.assert_almost_equal(expected, actual)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_zero_range_reverse_log_double(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1',
            bounds=(np.exp(0.9), np.exp(0.9)),
            scale_type=pyvizier.ScaleType.REVERSE_LOG,
        ),
        scale=True,
        onehot_embed=True,
        float_dtype=dtype,
    )
    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(np.exp(0.9))}),
        Trial(parameters={'x1': pyvizier.ParameterValue(np.exp(1.2))}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial(),
    ])
    expected = np.asarray(
        [[0.5], [0.5 + np.exp(1.2) - np.exp(0.9)], [np.nan], [np.nan]],
        dtype=dtype,
    )
    np.testing.assert_almost_equal(expected, actual)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_double_into_scaled_double(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1', bounds=(-3.0, 3.0), scale_type=pyvizier.ScaleType.LINEAR
        ),
        scale=True,
        onehot_embed=True,
        float_dtype=dtype,
    )

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial(),
    ])
    expected = np.asarray(
        [[4 / 6], [5 / 6], [0 / 6], [np.nan], [np.nan]], dtype=dtype
    )
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)

  def test_integer_discretes_into_discretes(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1, 2, 3)),
        max_discrete_indices=10,
    )

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial(),
    ])
    expected = np.asarray([[0], [1], [3], [3], [3]], np.int32)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_integer_discretes_into_doubles(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1, 2, 3)),
        max_discrete_indices=1,
        float_dtype=dtype,
    )

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial(),
    ])
    expected = np.asarray(
        [[1.0], [2.0], [-3.0], [np.nan], [np.nan]], dtype=dtype
    )
    np.testing.assert_allclose(expected, actual)

    pvs = converter.to_parameter_values(actual)
    self.assertSequenceEqual(
        pvs,
        [
            pyvizier.ParameterValue(1.0),
            pyvizier.ParameterValue(2.0),
            # TODO: This rounding behavior can be harmful.
            pyvizier.ParameterValue(1.0),
            None,
            None,
        ],
    )

    self.assertEqual(expected.dtype, actual.dtype)

  def test_integer_discretes_into_onehot(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1, 2, 3)),
        scale=True,
        onehot_embed=True,
        max_discrete_indices=10,
    )

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial(),
    ])
    expected = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)
    self.assertEqual(
        converter.output_spec,
        core.NumpyArraySpec(
            core.NumpyArraySpecType.ONEHOT_EMBEDDING,
            np.dtype(np.float32),
            (0, 1),
            4,
            'x1',
            1,
        ),
        msg=repr(converter.output_spec),
    )

  def test_integer_discretes_into_onehot_empty(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1, 2, 3)),
        scale=True,
        onehot_embed=True,
        max_discrete_indices=10,
    )

    actual = converter.convert([])
    expected = np.zeros([0, 4], dtype=np.float32)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)

  @parameterized.parameters([
      dict(max_discrete_indices=10),
      dict(max_discrete_indices=np.inf),
  ])
  def test_discretes_into_discretes(self, max_discrete_indices):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1.0, 2.0, 3.0)),
        max_discrete_indices=max_discrete_indices,
    )

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1.0)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2.0)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3.0)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial(),
    ])
    expected = np.asarray([[0], [1], [3], [3], [3]], dtype=np.float32)
    np.testing.assert_equal(expected, actual)

  def test_discretes_into_doubles(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1.0, 2.0, 3.0)),
        max_discrete_indices=1,
    )

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1.0)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2.0)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3.0)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial(),
    ])
    expected = np.asarray(
        [[1.0], [2.0], [-3.0], [np.nan], [np.nan]], dtype=np.float32
    )
    np.testing.assert_equal(expected, actual)

  def test_discretes_into_scaled_doubles(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1.0, 2.0, 3.0)),
        max_discrete_indices=1,
        scale=True,
    )

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1.0)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2.0)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3.0)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial(),
    ])
    expected = np.asarray(
        [[0.0], [0.5], [-2.0], [np.nan], [np.nan]], dtype=np.float32
    )
    np.testing.assert_equal(expected, actual)

  def test_categorical(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=('1', '2', '3')),
        max_discrete_indices=1,
    )

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue('1')}),
        Trial(parameters={'x1': pyvizier.ParameterValue('2')}),
        Trial(parameters={'x1': pyvizier.ParameterValue('-3')}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial(),
    ])
    expected = np.asarray([[0], [1], [3], [3], [3]], dtype=np.float32)
    np.testing.assert_equal(expected, actual)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64'),
  ])
  def test_bounds(self, dtype):
    input_converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', bounds=(-3.0, 3.0)),
        scale=True,
        onehot_embed=False,
        float_dtype=dtype,
    )
    self.assertEqual(input_converter.output_spec.bounds[0], 0.0)
    self.assertEqual(input_converter.output_spec.bounds[1], 1.0)


class ModelInputArrayBijectorTest(absltest.TestCase):

  def testIdentityBijector(self):
    parameter_config = pyvizier.ParameterConfig.factory('p', bounds=(0.0, 1.0))
    spec = core.NumpyArraySpec(
        name='p',
        type=core.NumpyArraySpecType.default_factory(parameter_config),
        dtype=np.dtype(np.float32),
        bounds=(0.0, 1.0),
        num_dimensions=1,
        num_oovs=0,
    )
    bijector = core.ModelInputArrayBijector.identity(spec)
    for x in [0.0, 0.25, 0.5, 0.625, 1.0]:
      x = np.array(x)
      self.assertLessEqual(
          bijector.forward_fn(x), bijector.output_spec.bounds[1]
      )
      self.assertGreaterEqual(
          bijector.forward_fn(x), bijector.output_spec.bounds[0]
      )
      self.assertEqual(bijector.backward_fn(bijector.forward_fn(x)), x)

  def testScalarFromSpec(self):
    parameter_config = pyvizier.ParameterConfig.factory('p', bounds=(1.0, 3.0))
    spec = core.NumpyArraySpec(
        name='p',
        type=core.NumpyArraySpecType.default_factory(parameter_config),
        dtype=np.dtype(np.float32),
        bounds=(1.0, 3.0),
        num_dimensions=1,
        num_oovs=0,
    )
    bijector = core.ModelInputArrayBijector.scaler_from_spec(spec)
    for x in [1.0, 1.25, 1.5, 2.625, 3.0]:
      x = np.array(x)
      self.assertLessEqual(
          bijector.forward_fn(x), bijector.output_spec.bounds[1]
      )
      self.assertGreaterEqual(
          bijector.forward_fn(x), bijector.output_spec.bounds[0]
      )
      self.assertEqual(bijector.backward_fn(bijector.forward_fn(x)), x)

  def testScalarFromSinglePointSpec(self):
    parameter_config = pyvizier.ParameterConfig.factory('p', bounds=(1.0, 1.0))
    spec = core.NumpyArraySpec(
        name='p',
        type=core.NumpyArraySpecType.default_factory(parameter_config),
        dtype=np.dtype(np.float32),
        bounds=(1.0, 1.0),
        num_dimensions=1,
        num_oovs=0,
    )
    bijector = core.ModelInputArrayBijector.scaler_from_spec(spec)
    x = np.array(1.0)
    logging.info(
        'fwd(x)=%s output_spec.bounds=%s',
        bijector.forward_fn(x),
        bijector.output_spec.bounds,
    )
    self.assertLessEqual(bijector.forward_fn(x), bijector.output_spec.bounds[1])
    self.assertGreaterEqual(
        bijector.forward_fn(x), bijector.output_spec.bounds[0]
    )
    self.assertLessEqual(bijector.forward_fn(x), bijector.output_spec.bounds[1])
    self.assertEqual(bijector.backward_fn(bijector.forward_fn(x)), x)


if __name__ == '__main__':
  absltest.main()
