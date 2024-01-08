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

import numpy as np
from vizier import pyvizier
from vizier._src.pyvizier.multimetric import safety
from absl.testing import absltest


class SafeAcquisitionTest(absltest.TestCase):

  def testSafeMeasurements(self):
    info1 = pyvizier.MetricInformation(
        name='safety1',
        goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE,
        safety_threshold=0.1)
    info2 = pyvizier.MetricInformation(
        name='safety2',
        goal=pyvizier.ObjectiveMetricGoal.MINIMIZE,
        safety_threshold=0.5)
    checker = safety.SafetyChecker(pyvizier.MetricsConfig([info1, info2]))
    safe_measurement = pyvizier.Measurement(metrics={
        'safety1': pyvizier.Metric(value=3.1),
        'safety2': pyvizier.Metric(value=0.3)
    })
    unsafe1_measurement = pyvizier.Measurement(metrics={
        'safety1': pyvizier.Metric(value=-3.1),
        'safety2': pyvizier.Metric(value=0.4)
    })
    unsafe2_measurement = pyvizier.Measurement(metrics={
        'safety1': pyvizier.Metric(value=3.1),
        'safety2': pyvizier.Metric(value=0.7)
    })
    no_safety_measurement = pyvizier.Measurement(
        metrics={'safety1': pyvizier.Metric(value=3.1)})

    self.assertListEqual(
        checker.are_measurements_safe([
            safe_measurement, unsafe1_measurement, unsafe2_measurement,
            no_safety_measurement
        ]), [True, False, False, True])

  def testSafeTrials(self):
    info1 = pyvizier.MetricInformation(
        name='obj', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    info2 = pyvizier.MetricInformation(
        name='safety',
        goal=pyvizier.ObjectiveMetricGoal.MINIMIZE,
        safety_threshold=0.5)
    checker = safety.SafetyChecker(pyvizier.MetricsConfig([info1, info2]))

    safe_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(metrics={
            'obj': 0.2,
            'safety': 0.3
        }))
    unsafe_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(metrics={
            'obj': 0.2,
            'safety': 0.7
        }))
    no_safety_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(metrics={'obj': 0.2}))

    # Empty Trial is assumed to be safe.
    self.assertListEqual(
        checker.are_trials_safe(
            [safe_trial, unsafe_trial, no_safety_trial,
             pyvizier.Trial()]), [True, False, True, True])

  def testNoSafety(self):
    info1 = pyvizier.MetricInformation(
        name='obj', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    checker = safety.SafetyChecker(pyvizier.MetricsConfig([info1]))

    safe_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(metrics={
            'obj': 0.2,
            'safety': 0.3
        }))
    unsafe_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(metrics={
            'obj': 0.2,
            'safety': 0.7
        }))

    # With no safety metrics, all Trials are safe..
    self.assertListEqual(
        checker.are_trials_safe([safe_trial, unsafe_trial,
                                 pyvizier.Trial()]), [True, True, True])

  def testWarpUnsafeTrials(self):
    info1 = pyvizier.MetricInformation(
        name='obj', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
    )
    info2 = pyvizier.MetricInformation(
        name='safety',
        goal=pyvizier.ObjectiveMetricGoal.MINIMIZE,
        safety_threshold=0.5,
    )
    checker = safety.SafetyChecker(pyvizier.MetricsConfig([info1, info2]))

    safe_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(
            metrics={'obj': 0.2, 'safety': 0.3}
        )
    )
    unsafe_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(
            metrics={'obj': 0.2, 'safety': 0.7, 'extra': 0.3}
        )
    )
    no_safety_trial = pyvizier.Trial(
        final_measurement=pyvizier.Measurement(metrics={'obj': 0.2})
    )

    # Empty safety measurement is assumed to be safe.
    warped_trials = checker.warp_unsafe_trials(
        [safe_trial, unsafe_trial, no_safety_trial]
    )

    self.assertEqual(warped_trials[0], safe_trial)
    self.assertEqual(warped_trials[2], no_safety_trial)

    # Unsafe Trial is warped.
    self.assertEqual(
        warped_trials[1].final_measurement.metrics['obj'].value, -np.inf
    )
    # Extra metrics should be untouched.
    self.assertEqual(
        warped_trials[1].final_measurement.metrics['extra'].value, 0.3
    )


if __name__ == '__main__':
  absltest.main()
