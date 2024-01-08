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

"""Tests for spatio_temporal."""

import numpy as np
from vizier import pyvizier
from vizier.pyvizier.converters import core
from vizier.pyvizier.converters import spatio_temporal as st

from absl.testing import absltest

_metric_converters = [
    core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='y1', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)),
    core.DefaultModelOutputConverter(
        pyvizier.MetricInformation(
            name='y2', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE))
]

_trials = [
    pyvizier.Trial(
        id=1,
        parameters={'x1': pyvizier.ParameterValue(1)},
        measurements=[
            pyvizier.Measurement(
                steps=1, elapsed_secs=10, metrics={
                    'y1': 1,
                    'y2': -1
                }),
            pyvizier.Measurement(
                steps=2, elapsed_secs=20, metrics={
                    'y1': 3,
                    'y2': -3
                }),
            pyvizier.Measurement(
                steps=3, elapsed_secs=30, metrics={
                    'y1': 2,
                    'y2': -2
                })
        ]),
    pyvizier.Trial(
        id=2,
        parameters={'x1': pyvizier.ParameterValue(2)},
        measurements=[
            pyvizier.Measurement(
                steps=1, elapsed_secs=10, metrics={
                    'y1': -4,
                    'y2': 4
                }),
            pyvizier.Measurement(
                steps=5, elapsed_secs=50, metrics={
                    'y1': -6,
                    'y2': 6
                }),
            pyvizier.Measurement(
                steps=6, elapsed_secs=60, metrics={
                    'y1': -5,
                    'y2': 5
                })
        ])
]


class TimedLabelsExtractorTest(absltest.TestCase):

  def test_steps(self):
    converter = st.TimedLabelsExtractor(
        _metric_converters, 'steps', value_extraction='raw')
    timed_labels = converter.convert(_trials)
    np.testing.assert_almost_equal(
        timed_labels[0].times,
        np.asarray([1, 2, 3], dtype=np.float32)[:, np.newaxis])
    np.testing.assert_almost_equal(
        timed_labels[1].times,
        np.asarray([1, 5, 6], dtype=np.float32)[:, np.newaxis])

  def test_elapsed_secs(self):
    converter = st.TimedLabelsExtractor(
        _metric_converters, 'elapsed_secs', value_extraction='raw')
    timed_labels = converter.convert(_trials)
    np.testing.assert_almost_equal(
        timed_labels[0].times,
        np.asarray([10, 20, 30], dtype=np.float32)[:, np.newaxis])
    np.testing.assert_almost_equal(
        timed_labels[1].times,
        np.asarray([10, 50, 60], dtype=np.float32)[:, np.newaxis])

  def test_index(self):
    converter = st.TimedLabelsExtractor(
        _metric_converters, 'index', value_extraction='raw')
    timed_labels = converter.convert(_trials)
    np.testing.assert_almost_equal(
        timed_labels[0].times,
        np.asarray([0, 1, 2], dtype=np.float32)[:, np.newaxis])
    np.testing.assert_almost_equal(
        timed_labels[1].times,
        np.asarray([0, 1, 2], dtype=np.float32)[:, np.newaxis])

  def test_labels_raw(self):
    converter = st.TimedLabelsExtractor(
        _metric_converters, 'elapsed_secs', value_extraction='raw')
    timed_labels = converter.convert(_trials)
    np.testing.assert_almost_equal(
        timed_labels[0].labels['y1'],
        np.asarray([1, 3, 2], dtype=np.float32)[:, np.newaxis])
    np.testing.assert_almost_equal(
        timed_labels[0].labels['y2'],
        np.asarray([-1, -3, -2], dtype=np.float32)[:, np.newaxis])

  def test_labels_cummax(self):
    converter = st.TimedLabelsExtractor(
        _metric_converters, 'elapsed_secs', value_extraction='cummax')
    timed_labels = converter.convert(_trials)
    np.testing.assert_almost_equal(
        timed_labels[0].labels['y1'],
        np.asarray([1, 3, 3], dtype=np.float32)[:, np.newaxis])
    np.testing.assert_almost_equal(
        timed_labels[0].labels['y2'],
        np.asarray([-1, -3, -3], dtype=np.float32)[:, np.newaxis])

  def test_labels_strict_cummax_firstonly(self):
    converter = st.TimedLabelsExtractor([_metric_converters[0]],
                                        'elapsed_secs',
                                        value_extraction='cummax_firstonly')
    timed_labels = converter.convert(_trials)
    np.testing.assert_almost_equal(
        timed_labels[0].times,
        np.asarray([10, 20, 30], dtype=np.float32)[:, np.newaxis])
    np.testing.assert_almost_equal(
        timed_labels[0].labels['y1'],
        np.asarray([1, 3, 3], dtype=np.float32)[:, np.newaxis])
    np.testing.assert_almost_equal(
        timed_labels[1].times,
        np.asarray([10, 60], dtype=np.float32)[:, np.newaxis])
    np.testing.assert_almost_equal(
        timed_labels[1].labels['y1'],
        np.asarray([-4, -4], dtype=np.float32)[:, np.newaxis])

  def test_labels_cummax_lastonly(self):
    converter = st.TimedLabelsExtractor([_metric_converters[0]],
                                        'elapsed_secs',
                                        value_extraction='cummax_lastonly')
    timed_labels = converter.convert(_trials)
    np.testing.assert_almost_equal(
        timed_labels[0].times,
        np.asarray([10, 30], dtype=np.float32)[:, np.newaxis])
    np.testing.assert_almost_equal(
        timed_labels[0].labels['y1'],
        np.asarray([1, 3], dtype=np.float32)[:, np.newaxis])
    np.testing.assert_almost_equal(
        timed_labels[1].times,
        np.asarray([60], dtype=np.float32)[:, np.newaxis])
    np.testing.assert_almost_equal(
        timed_labels[1].labels['y1'],
        np.asarray([-4], dtype=np.float32)[:, np.newaxis])

  def test_extract_all_timestamps(self):
    converter = st.TimedLabelsExtractor(
        _metric_converters, 'steps', value_extraction='cummax')

    all_ts = converter.extract_all_timestamps(_trials)
    np.testing.assert_almost_equal(
        all_ts, np.asarray([1, 2, 3, 5, 6], dtype=np.float32))


class SparseSpatioTemporalConverterTest(absltest.TestCase):

  def test_all(self):
    extractor = st.TimedLabelsExtractor(
        _metric_converters, 'steps', value_extraction='raw')
    converter = st.SparseSpatioTemporalConverter([], extractor)
    features, labels = converter.to_xy(_trials)

    np.testing.assert_equal(features, {
        'steps':
            np.array([[1.], [2.], [3.], [1.], [5.], [6.]], dtype=np.float32)
    })
    np.testing.assert_equal(
        labels, {
            'y1': [[1.], [3.], [2.], [-4.], [-6.], [-5.]],
            'y2': [[-1.], [-3.], [-2.], [4.], [6.], [5.]],
        })

    self.assertEqual(converter.features_shape, {'steps': (None, 1)})
    self.assertEqual(converter.labels_shape, {'y1': (None, 1), 'y2': (None, 1)})
    self.assertEqual(
        converter.output_specs, {
            'steps':
                core.NumpyArraySpec.from_parameter_config(
                    pyvizier.ParameterConfig.factory(
                        name='steps', bounds=(0.0, np.finfo(float).max)),
                    core.NumpyArraySpecType.default_factory)
        })


class DenseSpatioTemporalConverterTest(absltest.TestCase):

  def test_all(self):
    extractor = st.TimedLabelsExtractor(
        _metric_converters, 'steps', value_extraction='raw')
    converter = st.DenseSpatioTemporalConverter([],
                                                extractor,
                                                temporal_index_points=np.array(
                                                    [1., 2., 3., 5., 6.]))
    features, labels = converter.to_xy(_trials)

    np.testing.assert_equal(features, {})
    self.assertEqual(converter.features_shape, {})

    np.testing.assert_equal(
        labels, {
            'y1':
                np.array([[1., 3., 2., np.nan, np.nan],
                          [-4., np.nan, np.nan, -6., -5.]]),
            'y2':
                np.array([[-1., -3., -2., np.nan, np.nan],
                          [4., np.nan, np.nan, 6., 5.]])
        },
        err_msg=f'{labels}')

    self.assertEqual(converter.labels_shape, {'y1': (None, 5), 'y2': (None, 5)})

  def test_xty(self):
    extractor = st.TimedLabelsExtractor(
        _metric_converters, 'steps', value_extraction='raw')
    parameter = core.DefaultModelInputConverter(
        pyvizier.ParameterConfig.factory(name='x1', bounds=(0, 5)))
    converter = st.DenseSpatioTemporalConverter([parameter],
                                                extractor,
                                                temporal_index_points=np.array(
                                                    [1., 5., 6.]))
    features, temporal_index_points, labels = converter.to_xty(_trials, 'infer')

    np.testing.assert_equal(features, {'x1': [[1], [2]]})

    np.testing.assert_equal(temporal_index_points, [
        1.,
        2.,
        3.,
        5.,
        6.,
    ])

    np.testing.assert_equal(
        labels, {
            'y1':
                np.array([[1., 3., 2., np.nan, np.nan],
                          [-4., np.nan, np.nan, -6., -5.]]),
            'y2':
                np.array([[-1., -3., -2., np.nan, np.nan],
                          [4., np.nan, np.nan, 6., 5.]])
        },
        err_msg=f'{labels}')


if __name__ == '__main__':
  absltest.main()
