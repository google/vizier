"""Tests for core."""

import numpy as np
from vizier import pyvizier
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.testing import test_runners
from vizier.pyvizier.converters import core
from vizier.testing import test_studies

from absl.testing import absltest
from absl.testing import parameterized

Trial = pyvizier.Trial


class TrialToArrayConverterTest(absltest.TestCase):
  """Test TrialToArrayConverter class."""

  def setUp(self):
    super().setUp()
    self._study_config = pyvizier.StudyConfig(
        search_space=test_studies.flat_space_with_all_types(),
        metric_information=[
            pyvizier.MetricInformation(
                'x1', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
        ])

    self._designer = random.RandomDesigner(
        self._study_config.search_space, seed=0)
    self._trials = test_runners.run_with_random_metrics(
        self._designer, self._study_config, iters=1, batch_size=10)
    self.maxDiff = None  # pylint: disable=invalid-name

  def test_back_to_back_conversion(self):
    converter = core.TrialToArrayConverter.from_study_config(self._study_config)
    self.assertSequenceEqual([t.parameters for t in self._trials],
                             converter.to_parameters(
                                 converter.to_features(self._trials)))


class DictToArrayTest(absltest.TestCase):

  def test_2d(self):
    self.assertSequenceEqual(
        core.dict_to_array({
            1: np.random.random((10, 2)),
            2: np.random.random((10, 1))
        }).shape, (10, 3))

  def test_4d(self):
    self.assertSequenceEqual(
        core.dict_to_array({
            1: np.random.random((2, 1, 10, 2)),
            2: np.random.random((2, 1, 10, 1))
        }).shape, (2, 1, 10, 3))

  def test_dict_like(self):
    d = core.DictOf2DArrays({
        'p1': np.array([[1], [2], [3]]),
        'p2': np.array([[.0, .1], [.5, .5], [.1, .0]])
    })
    self.assertCountEqual(
        d.dict_like(np.array([[1, 1, .1], [3, 0, .1]])), {
            'p1': np.array([[1], [3]]),
            'p2': np.array([[1, .1], [0, .1]])
        })

  def test_dict_like_and_as_array_are_inverse(self):
    d = core.DictOf2DArrays({
        'p1': np.array([[1], [2], [3]]),
        'p2': np.array([[.0, .1], [.5, .5], [.1, .0]])
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
    root = space1.select_root()
    root.add_float_param('double', -1., 1.)
    root.add_int_param('integer', -1, 1)
    root.add_categorical_param('categorical', ['a', 'b'])
    root.add_discrete_param('discrete', [-1., 1.])
    study_configs.append(
        pyvizier.StudyConfig(
            metadata=pyvizier.Metadata({core.STUDY_ID_FIELD: 'study1'}),
            search_space=space1))

    space2 = pyvizier.SearchSpace()
    root = space2.select_root()
    root.add_float_param('double', -1., 2.)
    root.add_int_param('integer', -2, 1)
    root.add_categorical_param('categorical', ['b', 'c'])
    root.add_discrete_param('discrete', [-1., 1., 2.])
    study_configs.append(
        pyvizier.StudyConfig(
            metadata=pyvizier.Metadata({core.STUDY_ID_FIELD: 'study2'}),
            search_space=space2))
    return study_configs

  def test_parameters(self):
    converter = core.DefaultTrialConverter.from_study_configs(
        self._study_configs, [], use_study_id_feature=True)
    self.assertCountEqual(
        ['study1', 'study2'],
        converter.parameter_configs[core.STUDY_ID_FIELD].feasible_values)
    self.assertEqual((-1., 2.), converter.parameter_configs['double'].bounds)
    self.assertEqual((-2, 1), converter.parameter_configs['integer'].bounds)
    self.assertCountEqual(
        ('a', 'b', 'c'),
        converter.parameter_configs['categorical'].feasible_values)
    self.assertCountEqual(
        (-1., 1., 2.), converter.parameter_configs['discrete'].feasible_values)

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
            'double': pyvizier.ParameterValue(3.),
            'integer': pyvizier.ParameterValue(-1),
            'categorical': pyvizier.ParameterValue('d')
        })
    expected = {
        'categorical': np.array([[3]]),
        'discrete': np.array([[3]]),
        'double': np.array([[3.]]),
        'integer': np.array([[1]]),
        'metalearn_study_id': np.array([[2]])
    }
    self.assertDictEqual(converter.to_features([trial]), expected)

    # The value for `categorical` was invalid, so we can't recover it.
    expected_parameters = pyvizier.ParameterDict({
        'double': pyvizier.ParameterValue(2.),
        'integer': pyvizier.ParameterValue(-1),
    })
    self.assertEqual(
        converter.to_trials(expected)[0].parameters, expected_parameters)

  def test_metrics(self):
    converter = core.DefaultTrialConverter.from_study_configs(
        [], [
            pyvizier.MetricInformation(
                name='metric1', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE)
        ],
        use_study_id_feature=False)

    expected = {'metric1': (None, 1)}
    self.assertDictEqual(converter.labels_shape, expected)


class DefaultModelOutputConverterTest(parameterized.TestCase):

  @property
  def _measurements(self):
    return [
        pyvizier.Measurement(metrics={
            'metric1': pyvizier.Metric(1.0),
            'metric2': pyvizier.Metric(1.1)
        }),
        pyvizier.Measurement(metrics={
            'metric1': pyvizier.Metric(2.0),
            'metric2': pyvizier.Metric(2.1)
        }),
        pyvizier.Measurement(metrics={
            'metric1': pyvizier.Metric(4.0),
            'metric2': pyvizier.Metric(4.1)
        }),
    ]

  def test_empty(self):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE),
        flip_sign_for_minimization_metrics=True)
    np.testing.assert_array_equal(
        converter.convert([]), np.zeros([0, 1], dtype=converter.dtype))

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64')
  ])
  def test_sign_flips(self, dtype):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE),
        flip_sign_for_minimization_metrics=True,
        dtype=dtype)
    actual = converter.convert(self._measurements)

    expected = -np.asarray([[1.], [2.], [4.]], dtype)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)
    self.assertEqual(
        converter.metric_information,
        pyvizier.MetricInformation(
            name='metric1', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE))

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64')
  ])
  def test_no_sign_flips(self, dtype):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric2', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE),
        flip_sign_for_minimization_metrics=False,
        dtype=dtype)
    actual = converter.convert(self._measurements)

    expected = np.asarray([[1.1], [2.1], [4.1]], dtype)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)
    self.assertEqual(
        converter.metric_information,
        pyvizier.MetricInformation(
            name='metric2', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE))

  def test_raise_errors_for_missing_metrics(self):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric3', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE),
        flip_sign_for_minimization_metrics=False,
        raise_errors_for_missing_metrics=True)
    with self.assertRaises(KeyError):
      converter.convert(self._measurements)

  def test_do_not_raise_errors_for_missing_metrics(self):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric3', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE),
        flip_sign_for_minimization_metrics=False,
        raise_errors_for_missing_metrics=False)
    np.testing.assert_equal(
        converter.convert(self._measurements), np.asarray([[np.nan]] * 3))

  @parameterized.parameters([dict(flip_sign=True), dict(flip_sign=False)])
  def test_safe_maximize_metric(self, flip_sign):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1',
            goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE,
            safety_threshold=3.),
        flip_sign_for_minimization_metrics=flip_sign)
    actual = converter.convert(self._measurements)
    np.testing.assert_equal(actual, [[-2.], [-1.], [1.]])

  @parameterized.parameters([dict(flip_sign=True), dict(flip_sign=False)])
  def test_safe_minimize_metric(self, flip_sign):
    converter = core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='metric1',
            goal=pyvizier.ObjectiveMetricGoal.MINIMIZE,
            safety_threshold=3.),
        flip_sign_for_minimization_metrics=flip_sign)
    actual = converter.convert(self._measurements)
    expected = np.array([[-2.], [-1.], [1.]],
                        dtype=np.float32) * (-1 if flip_sign else 1)
    np.testing.assert_equal(actual, expected)


class DefaultModelInputConverterTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64')
  ])
  def test_double_into_double(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', bounds=(-3., 3.)),
        scale=False,
        onehot_embed=True,
        float_dtype=dtype)

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial()
    ])
    expected = np.asarray([[1.0], [2.0], [-3.0], [np.NaN], [np.NaN]], dtype)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64')
  ])
  def test_double_into_double_log(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1', bounds=(1e-4, 1e2), scale_type=pyvizier.ScaleType.LOG),
        scale=True,
        float_dtype=dtype)

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
      dict(dtype='float64')
  ])
  def test_double_into_double_log_inverse(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1', bounds=(1e-4, 1e2), scale_type=pyvizier.ScaleType.LOG),
        scale=True,
        float_dtype=dtype)

    scaled = np.asarray([[0.0], [1.0]], dtype)
    actual = converter.to_parameter_values(scaled)
    self.assertGreaterEqual(actual[0].value, 1e-4)
    self.assertLessEqual(actual[1].value, 1e2)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64')
  ])
  def test_zero_range_linear_double(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1', bounds=(.9, .9), scale_type=pyvizier.ScaleType.LINEAR),
        scale=True,
        onehot_embed=True,
        float_dtype=dtype)
    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(.9)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(1.)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial()
    ])
    expected = np.asarray([[.0], [.0], [np.NaN], [np.NaN]], dtype)
    np.testing.assert_equal(expected, actual)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64')
  ])
  def test_zero_range_log_double(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1',
            bounds=(np.exp(.9), np.exp(.9)),
            scale_type=pyvizier.ScaleType.LOG),
        scale=True,
        onehot_embed=True,
        float_dtype=dtype)
    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(np.exp(.9))}),
        Trial(parameters={'x1': pyvizier.ParameterValue(np.exp(1.2))}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial()
    ])
    expected = np.asarray([[.0], [.0], [np.NaN], [np.NaN]], dtype=dtype)
    np.testing.assert_equal(expected, actual)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64')
  ])
  def test_double_into_scaled_double(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(
            'x1', bounds=(-3., 3.), scale_type=pyvizier.ScaleType.LINEAR),
        scale=True,
        onehot_embed=True,
        float_dtype=dtype)

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial()
    ])
    expected = np.asarray([[4 / 6], [5 / 6], [0 / 6], [np.NaN], [np.NaN]],
                          dtype=dtype)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)

  def test_integer_discretes_into_discretes(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1, 2, 3)),
        max_discrete_indices=10)

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial()
    ])
    expected = np.asarray([[0], [1], [3], [3], [3]], np.int32)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)

  @parameterized.parameters([
      dict(dtype=np.float32),
      dict(dtype=np.float64),
      dict(dtype='float32'),
      dict(dtype='float64')
  ])
  def test_integer_discretes_into_doubles(self, dtype):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1, 2, 3)),
        max_discrete_indices=1,
        float_dtype=dtype)

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial()
    ])
    expected = np.asarray([[1.0], [2.0], [-3.0], [np.NaN], [np.NaN]],
                          dtype=dtype)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)

  def test_integer_discretes_into_onehot(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1, 2, 3)),
        scale=True,
        onehot_embed=True,
        max_discrete_indices=10)

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial()
    ])
    expected = np.asarray([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [0., 0., 0., 1.],
        [0., 0., 0., 1.],
    ],
                          dtype=np.float32)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)
    self.assertEqual(
        converter.output_spec,
        core.NumpyArraySpec(core.NumpyArraySpecType.ONEHOT_EMBEDDING,
                            np.float32, (0, 1), 4, 'x1', 1),
        msg=repr(converter.output_spec))

  def test_integer_discretes_into_onehot_empty(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1, 2, 3)),
        scale=True,
        onehot_embed=True,
        max_discrete_indices=10)

    actual = converter.convert([])
    expected = np.zeros([0, 4], dtype=np.float32)
    np.testing.assert_allclose(expected, actual)
    self.assertEqual(expected.dtype, actual.dtype)

  def test_discretes_into_discretes(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1., 2., 3.)),
        max_discrete_indices=10)

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1.)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2.)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3.)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial()
    ])
    expected = np.asarray([[0], [1], [3], [3], [3]], dtype=np.float32)
    np.testing.assert_equal(expected, actual)

  def test_discretes_into_doubles(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1., 2., 3.)),
        max_discrete_indices=1)

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1.)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2.)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3.)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial()
    ])
    expected = np.asarray([[1.0], [2.0], [-3.0], [np.NaN], [np.NaN]],
                          dtype=np.float32)
    np.testing.assert_equal(expected, actual)

  def test_discretes_into_scaled_doubles(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=(1., 2., 3.)),
        max_discrete_indices=1,
        scale=True)

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue(1.)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(2.)}),
        Trial(parameters={'x1': pyvizier.ParameterValue(-3.)}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial()
    ])
    expected = np.asarray([[0.], [.5], [-2.], [np.NaN], [np.NaN]],
                          dtype=np.float32)
    np.testing.assert_equal(expected, actual)

  def test_categorical(self):
    converter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory('x1', feasible_values=('1', '2', '3')),
        max_discrete_indices=1)

    actual = converter.convert([
        Trial(parameters={'x1': pyvizier.ParameterValue('1')}),
        Trial(parameters={'x1': pyvizier.ParameterValue('2')}),
        Trial(parameters={'x1': pyvizier.ParameterValue('-3')}),
        Trial(parameters={'x1': pyvizier.ParameterValue('a')}),
        Trial()
    ])
    expected = np.asarray([[0], [1], [3], [3], [3]], dtype=np.float32)
    np.testing.assert_equal(expected, actual)


if __name__ == '__main__':
  absltest.main()
