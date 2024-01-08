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

"""Tests for vizier.pyvizier.shared.base_study_config."""

import numpy as np
from vizier._src.pyvizier.shared import base_study_config
from absl.testing import absltest
from absl.testing import parameterized


class ObjectiveMetricGoalTest(absltest.TestCase):

  def test_basics(self):
    self.assertTrue(base_study_config.ObjectiveMetricGoal.MAXIMIZE.is_maximize)
    self.assertFalse(base_study_config.ObjectiveMetricGoal.MAXIMIZE.is_minimize)
    self.assertTrue(base_study_config.ObjectiveMetricGoal.MINIMIZE.is_minimize)
    self.assertFalse(base_study_config.ObjectiveMetricGoal.MINIMIZE.is_maximize)


class MetricTypeTest(absltest.TestCase):

  def test_basics(self):
    self.assertTrue(base_study_config.MetricType.SAFETY.is_safety)
    self.assertTrue(base_study_config.MetricType.OBJECTIVE.is_objective)


class MetricInformationTest(absltest.TestCase):

  def testMinMaxValueDefault(self):
    info = base_study_config.MetricInformation(
        goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE)
    self.assertEqual(info.min_value, -np.inf)
    self.assertEqual(info.max_value, np.inf)

  def testMinMaxValueSet(self):
    info = base_study_config.MetricInformation(
        goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE,
        min_value=-1.,
        max_value=1.)
    self.assertEqual(info.min_value, -1.)
    self.assertEqual(info.max_value, 1.)

  def testMinMaxBadValueInit(self):
    with self.assertRaises(ValueError):
      base_study_config.MetricInformation(
          goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE,
          min_value=1.,
          max_value=-1.)

  def testMinMaxBadValueSet(self):
    info = base_study_config.MetricInformation(
        goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE,
        min_value=-1.,
        max_value=1.)
    with self.assertRaises(ValueError):
      info.min_value = 2.
    with self.assertRaises(ValueError):
      info.max_value = -2.


class MetricsConfigTest(parameterized.TestCase):

  def testBasics(self):
    config = base_study_config.MetricsConfig()
    config.append(
        base_study_config.MetricInformation(
            name='max1', goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE))
    config.extend([
        base_study_config.MetricInformation(
            name='max_safe1',
            goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE,
            safety_threshold=0.0),
        base_study_config.MetricInformation(
            name='max2', goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE),
        base_study_config.MetricInformation(
            name='min1', goal=base_study_config.ObjectiveMetricGoal.MINIMIZE),
        base_study_config.MetricInformation(
            name='min_safe2',
            goal=base_study_config.ObjectiveMetricGoal.MINIMIZE,
            safety_threshold=0.0,
            desired_min_safe_trials_fraction=0.1)
    ])
    self.assertLen(config, 5)
    self.assertLen(config.of_type(base_study_config.MetricType.OBJECTIVE), 3)
    self.assertLen(config.of_type(base_study_config.MetricType.SAFETY), 2)
    self.assertLen(
        config.exclude_type(base_study_config.MetricType.OBJECTIVE), 2
    )

  def testDuplicateNames(self):
    config = base_study_config.MetricsConfig()
    config.append(
        base_study_config.MetricInformation(
            name='max1', goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE))
    with self.assertRaises(ValueError):
      config.append(
          base_study_config.MetricInformation(
              name='max1', goal=base_study_config.ObjectiveMetricGoal.MAXIMIZE))


if __name__ == '__main__':
  absltest.main()
