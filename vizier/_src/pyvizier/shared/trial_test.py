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

"""Tests for vizier.pyvizier.shared.trial."""
import copy
import datetime

from typing import Sequence

import numpy as np

from vizier._src.pyvizier.shared import trial
from absl.testing import absltest
from absl.testing import parameterized

Metric = trial.Metric
Measurement = trial.Measurement


class MetricTest(absltest.TestCase):

  def testMetricCreation(self):
    _ = Metric(value=0, std=0.5)

  def testMetricCannotHaveNaN(self):
    with self.assertRaises(ValueError):
      _ = Metric(value=np.nan, std=-np.nan)

  def testMetricCannotHaveNegativeStd(self):
    with self.assertRaises(ValueError):
      _ = Metric(value=0, std=-0.5)


class MeasurementTest(absltest.TestCase):

  def testMetricsInitializedFromFloats(self):
    m = Measurement()
    m.metrics = dict(a=0.3)
    self.assertEqual(m.metrics['a'], Metric(0.3))
    m.metrics['b'] = 0.5
    self.assertEqual(m.metrics, {'a': Metric(0.3), 'b': Metric(0.5)})

  def testMetrics(self):
    m = Measurement()
    m.metrics = dict(a=Metric(0.3))
    self.assertEqual(m.metrics['a'], Metric(0.3))

  def testTimeStampsAreNotFrozen(self):
    m = Measurement()
    m.elapsed_secs = 1.0
    m.steps = 5


ParameterValue = trial.ParameterValue


class ParameterValueTest(parameterized.TestCase):

  @parameterized.parameters((True,), (False,))
  def testBool(self, bool_value):
    value = ParameterValue(bool_value)
    self.assertEqual(value.as_float, float(bool_value))
    self.assertEqual(value.as_int, int(bool_value))
    self.assertEqual(value.as_str, str(bool_value))

  def testIntegralFloat0(self):
    value = ParameterValue(0.0)
    self.assertEqual(value.as_float, 0.0)
    self.assertEqual(value.as_int, 0)
    self.assertEqual(value.as_bool, False)
    self.assertEqual(value.as_str, '0.0')

  def testIntegralFloat1(self):
    value = ParameterValue(1.0)
    self.assertEqual(value.as_float, 1.0)
    self.assertEqual(value.as_int, 1)
    self.assertEqual(value.as_bool, True)
    self.assertEqual(value.as_str, '1.0')

  def testIntegralFloat2(self):
    value = ParameterValue(2.0)
    self.assertEqual(value.as_float, 2.0)
    self.assertEqual(value.as_int, 2)
    self.assertIsNone(value.as_bool)
    self.assertEqual(value.as_str, '2.0')

  def testInteger0(self):
    value = ParameterValue(0)
    self.assertEqual(value.as_float, 0)
    self.assertEqual(value.as_int, 0)
    self.assertEqual(value.as_bool, False)
    self.assertEqual(value.as_str, '0')

  def testInteger1(self):
    value = ParameterValue(1)
    self.assertEqual(value.as_float, 1)
    self.assertEqual(value.as_int, 1)
    self.assertEqual(value.as_bool, True)
    self.assertEqual(value.as_str, '1')

  def testInteger2(self):
    value = ParameterValue(2)
    self.assertEqual(value.as_float, 2)
    self.assertEqual(value.as_int, 2)
    self.assertIsNone(value.as_bool)
    self.assertEqual(value.as_str, '2')

  def testStringTrue(self):
    value = ParameterValue(trial.TRUE_VALUE)
    self.assertEqual(value.as_bool, True)
    self.assertEqual(value.as_str, trial.TRUE_VALUE)

  def testStringFalse(self):
    value = ParameterValue(trial.FALSE_VALUE)
    self.assertEqual(value.as_bool, False)
    self.assertEqual(value.as_str, trial.FALSE_VALUE)

  def testStringFloat1(self):
    value = ParameterValue('1.0')
    self.assertEqual(value.as_float, 1.0)
    self.assertIsNone(value.as_int)
    self.assertIsNone(value.as_bool)
    self.assertEqual(value.as_str, '1.0')

  def testStringInt1(self):
    value = ParameterValue('1')
    self.assertEqual(value.as_float, 1.0)
    self.assertEqual(value.as_int, 1)
    self.assertIsNone(value.as_bool)
    self.assertEqual(value.as_str, '1')

  def testParameterCanHaveNonFiniteValues(self):
    ParameterValue(float('nan'))
    ParameterValue(value=float('inf'))
    ParameterValue(value=float('inf'))


class TrialTest(absltest.TestCase):

  def testCompleteInplace(self):
    test = trial.Trial()
    measurement = Measurement(metrics={
        'pr-auc': Metric(value=0.8),
        'latency': Metric(value=32)
    })
    completed = test.complete(measurement, inplace=True)

    # The trial was completed in place.
    self.assertEqual(test.final_measurement, measurement)
    self.assertLessEqual(test.completion_time,
                         datetime.datetime.now().astimezone())
    self.assertGreaterEqual(test.completion_time, test.creation_time)
    self.assertGreaterEqual(test.duration.total_seconds(), 0)

    self.assertEqual(completed.final_measurement, measurement)
    self.assertLessEqual(completed.completion_time,
                         datetime.datetime.now().astimezone())
    self.assertGreaterEqual(completed.completion_time, completed.creation_time)
    self.assertGreaterEqual(completed.duration.total_seconds(), 0)

    # completed is the same reference as test.
    self.assertEqual(test, completed)

  def testCompleteNotInplace(self):
    """Complete with inplace=False."""
    test = trial.Trial()
    measurement = Measurement(metrics={
        'pr-auc': Metric(value=0.8),
        'latency': Metric(value=32)
    })

    test_copy = copy.deepcopy(test)

    completed = test.complete(measurement, inplace=False)

    # The returned Trial is completed.
    self.assertEqual(completed.final_measurement, measurement)
    self.assertGreaterEqual(completed.completion_time, completed.creation_time)
    self.assertLessEqual(completed.completion_time,
                         datetime.datetime.now().astimezone())
    self.assertGreaterEqual(completed.duration.total_seconds(), 0)
    self.assertEqual(completed.status, trial.TrialStatus.COMPLETED)
    self.assertTrue(completed.is_completed)

    # The original Trial is unchanged.
    self.assertEqual(test_copy, test)
    self.assertIsNone(test.final_measurement)
    self.assertIsNone(test.completion_time)
    self.assertIsNone(test.duration)
    self.assertEqual(test.status, trial.TrialStatus.ACTIVE)
    self.assertFalse(test.is_completed)

  def testCompleteInfeasible(self):
    test = trial.Trial()
    measurement = Measurement(metrics={
        'pr-auc': Metric(value=0.8),
        'latency': Metric(value=32)
    })
    completed = test.complete(
        measurement, inplace=False, infeasibility_reason='reason')
    # Test infeasibility.
    self.assertTrue(completed.infeasible)
    self.assertEqual(completed.infeasibility_reason, 'reason')

  def testCompleteInfeasible2(self):
    test = trial.Trial(infeasibility_reason='reason')
    measurement = Measurement(metrics={
        'pr-auc': Metric(value=0.8),
        'latency': Metric(value=32)
    })
    completed = test.complete(measurement, inplace=False)
    # When infeasibility not provided, the trial shoud remain infeasible.
    self.assertTrue(completed.infeasible)
    self.assertEqual(completed.infeasibility_reason, 'reason')

  def testCompleteInfeasible3(self):
    test = trial.Trial(infeasibility_reason='reason')
    measurement = Measurement(metrics={
        'pr-auc': Metric(value=0.8),
        'latency': Metric(value=32)
    })
    completed = test.complete(
        measurement, inplace=False, infeasibility_reason='other')
    # Infeasibility reason should be updated.
    self.assertTrue(completed.infeasible)
    self.assertEqual(completed.infeasibility_reason, 'other')
    # The original trial should retain the original reason.
    self.assertTrue(test.infeasible)
    self.assertEqual(test.infeasibility_reason, 'reason')

  def testCompleteEmptyInfeasible(self):
    test = trial.Trial()
    measurement = Measurement(metrics={
        'pr-auc': Metric(value=0.8),
        'latency': Metric(value=32)
    })
    completed = test.complete(
        measurement, inplace=False, infeasibility_reason='')
    # Test infeasibility.
    self.assertTrue(completed.infeasible)
    self.assertEqual(completed.infeasibility_reason, '')

  def testCompleteInfeasibleInplace(self):
    test = trial.Trial()
    measurement = Measurement(metrics={
        'pr-auc': Metric(value=0.8),
        'latency': Metric(value=32)
    })
    test.complete(measurement, inplace=True, infeasibility_reason='reason')
    # Test infeasibility.
    self.assertTrue(test.infeasible)
    self.assertEqual(test.infeasibility_reason, 'reason')

  def testCompleteInfeasibleInplace2(self):
    test = trial.Trial(infeasibility_reason='reason')
    measurement = Measurement(metrics={
        'pr-auc': Metric(value=0.8),
        'latency': Metric(value=32)
    })
    test.complete(measurement, inplace=True)
    # When infeasibility not provided, the trial shoud remain infeasible.
    self.assertTrue(test.infeasible)
    self.assertEqual(test.infeasibility_reason, 'reason')

  def testCompleteInfeasibleInplace3(self):
    test = trial.Trial(infeasibility_reason='reason')
    measurement = Measurement(metrics={
        'pr-auc': Metric(value=0.8),
        'latency': Metric(value=32)
    })
    test.complete(measurement, inplace=True, infeasibility_reason='other')
    # Infeasibility reason should be updated.
    self.assertTrue(test.infeasible)
    self.assertEqual(test.infeasibility_reason, 'other')

  def testCompleteEmptyInfeasibleInplace(self):
    test = trial.Trial()
    measurement = Measurement(metrics={
        'pr-auc': Metric(value=0.8),
        'latency': Metric(value=32)
    })
    completed = test.complete(
        measurement, inplace=True, infeasibility_reason='')
    # Test infeasibility.
    self.assertTrue(completed.infeasible)
    self.assertEqual(completed.infeasibility_reason, '')

  def testDefaultsNotShared(self):
    """Make sure default parameters are not shared between instances."""
    trial1 = trial.Trial()
    trial2 = trial.Trial()
    trial1.parameters['x1'] = trial.ParameterValue(5)
    self.assertEmpty(trial2.parameters)

  def testCreationTime(self):
    trial1 = trial.Trial()
    trial2 = trial.Trial()
    self.assertGreater(trial2.creation_time, trial1.creation_time)


class ParameterDictTest(parameterized.TestCase):

  @parameterized.parameters((True,), (3,), (1.,), ('aa',))
  def testAssignRawValue(self, v):
    d = trial.ParameterDict()
    d['p1'] = v
    self.assertEqual(d.get('p1'), trial.ParameterValue(v))
    self.assertEqual(d.get_value('p1'), v)
    self.assertEqual(d.get_value('p2', 'default'), 'default')
    self.assertLen(d, 1)
    self.assertLen(d.items(), 1)

  @parameterized.parameters((True,), (3,), (1.,), ('aa',))
  def testAssignWrappedValue(self, v):
    d = trial.ParameterDict()
    v = trial.ParameterValue(v)
    d['p1'] = v
    self.assertEqual(d.get('p1'), v)
    self.assertEqual(d.get_value('p1'), v.value)
    self.assertEqual(d.get_value('p2', 'default'), 'default')
    self.assertLen(d, 1)
    self.assertLen(d.items(), 1)


class SuggestionTestI(absltest.TestCase):

  def testToTrial(self):
    suggestion = trial.TrialSuggestion({'a': 3, 'b': True})
    suggestion.metadata['key'] = 'value'

    t = suggestion.to_trial(1)
    self.assertEqual(t.id, 1)
    self.assertEqual(t.parameters, suggestion.parameters)
    self.assertEqual(t.metadata, suggestion.metadata)


class TrialFilterTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(filtr=trial.TrialFilter(), answers=[True, True, True, True]),
      dict(
          filtr=trial.TrialFilter(ids=(2, 3)),
          answers=[False, True, True, False]),
      dict(
          filtr=trial.TrialFilter(min_id=3, ids=(2, 3)),
          answers=[False, False, True, False]),
      dict(
          filtr=trial.TrialFilter(
              min_id=2,
              max_id=3,
              ids=(1, 2, 3, 4),
              status=[trial.TrialStatus.REQUESTED]),
          answers=[False, True, False, False]))
  def test_filter(self, filtr: trial.TrialFilter, answers: Sequence[bool]):
    trials = (
        trial.Trial(id=1),  # ACTIVE
        trial.Trial(id=2, is_requested=True),  #  REQUESTED
        trial.Trial(id=3, stopping_reason='stopping'),  # STOPPING
        trial.Trial(id=4).complete(trial.Measurement()),  # COMPLETED
    )
    self.assertSequenceEqual([filtr(t) for t in trials], answers)


if __name__ == '__main__':
  absltest.main()
