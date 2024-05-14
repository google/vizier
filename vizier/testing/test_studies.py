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

"""Test study generator."""

from typing import List
import numpy as np
from vizier import pyvizier as vz


def flat_continuous_space_with_scaling() -> vz.SearchSpace:
  """Search space with float parameter types."""

  space = vz.SearchSpace()
  root = space.root
  root.add_float_param('lineardouble', -1., 2.)
  root.add_float_param('logdouble', 1e-4, 1e2, scale_type=vz.ScaleType.LOG)
  return space


def flat_categorical_space() -> vz.SearchSpace:
  """Search space with categorical parameter types."""

  space = vz.SearchSpace()
  root = space.root
  root.add_categorical_param('categorical_0', ['a', 'aa', 'aaa'])
  root.add_categorical_param('categorical_1', ['b', 'bb', 'bbb'])
  return space


def flat_continuous_space_with_scaling_trials(
    count: int = 1,
) -> list[vz.TrialSuggestion]:
  """Trials of search space with float parameter types."""
  trials = []
  for _ in range(count):
    trials.append(
        vz.Trial({
            'lineardouble': np.random.uniform(low=-1.0, high=2.0),
            'logdouble': np.random.uniform(low=1e-4, high=1e2),
        })
    )
  return trials


def flat_space_with_all_types() -> vz.SearchSpace:
  """Search space with all parameter types."""

  space = vz.SearchSpace()
  root = space.root
  root.add_float_param('lineardouble', -1., 2.)
  root.add_float_param('logdouble', 1e-4, 1e2, scale_type=vz.ScaleType.LOG)
  root.add_int_param('integer', -2, 2)
  root.add_categorical_param('categorical', ['a', 'aa', 'aaa'])
  root.add_bool_param('boolean')
  root.add_discrete_param('discrete_double', [-.5, 1.0, 1.2])
  root.add_discrete_param(
      'discrete_logdouble', [1e-5, 1e-2, 1e-1], scale_type=vz.ScaleType.LOG)
  root.add_discrete_param('discrete_int', [-1, 1, 2])

  return space


def conditional_automl_space() -> vz.SearchSpace:
  """Conditional space for a simple AutoML task."""
  space = vz.SearchSpace()
  root = space.select_root()
  root.add_categorical_param(
      'model_type', ['linear', 'dnn'], default_value='dnn'
  )

  dnn = root.select('model_type', ['dnn'])
  dnn.add_float_param(
      'learning_rate',
      0.0001,
      1.0,
      default_value=0.001,
      scale_type=vz.ScaleType.LOG,
  )

  linear = root.select('model_type', ['linear'])
  linear.add_float_param(
      'learning_rate', 0.1, 1.0, default_value=0.1, scale_type=vz.ScaleType.LOG
  )

  _ = dnn.add_categorical_param('optimizer_type', ['adam', 'evolution'])

  # Chained select() calls, path length of 1.
  root.select('model_type', ['dnn']).select(
      'optimizer_type', ['adam']
  ).add_float_param(
      'learning_rate', 0.1, 1.0, default_value=0.1, scale_type=vz.ScaleType.LOG
  )

  # Chained select() calls, path length of 2.
  ko = (
      root.select('model_type', ['dnn'])
      .select('optimizer_type', ['adam'])
      .add_bool_param('use_special_logic', default_value=False)
  )

  ko2 = ko.select_values(['True'])
  _ = ko2.add_float_param(
      'special_logic_parameter', 1.0, 3.0, default_value=2.1
  )
  return space


def metrics_objective_goals() -> List[vz.MetricInformation]:
  return [
      vz.MetricInformation('gain', goal=vz.ObjectiveMetricGoal.MAXIMIZE),
      vz.MetricInformation('loss', goal=vz.ObjectiveMetricGoal.MINIMIZE),
      vz.MetricInformation(
          'auc',
          goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          min_value=0.0,
          max_value=1.0),
      vz.MetricInformation(
          'crossentropy', goal=vz.ObjectiveMetricGoal.MINIMIZE, min_value=0.0),
  ]


def metrics_all_unconstrained() -> List[vz.MetricInformation]:
  return [
      vz.MetricInformation('gain', goal=vz.ObjectiveMetricGoal.MAXIMIZE),
      vz.MetricInformation('loss', goal=vz.ObjectiveMetricGoal.MINIMIZE),
      vz.MetricInformation(
          'gt2', goal=vz.ObjectiveMetricGoal.MAXIMIZE, safety_threshold=2.0),
      vz.MetricInformation(
          'lt2', goal=vz.ObjectiveMetricGoal.MINIMIZE, safety_threshold=2.0),
  ]


def metrics_all_constrained() -> List[vz.MetricInformation]:
  return [
      vz.MetricInformation(
          'auc',
          goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          min_value=0.0,
          max_value=1.0),
      vz.MetricInformation(
          'crossentropy', goal=vz.ObjectiveMetricGoal.MINIMIZE, min_value=0.0),
      vz.MetricInformation(
          'gt2',
          goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          safety_threshold=2.0,
          min_value=-1.0,
          max_value=5.0),
      vz.MetricInformation(
          'lt2',
          goal=vz.ObjectiveMetricGoal.MINIMIZE,
          safety_threshold=2.0,
          min_value=-1.0,
          max_value=5.0),
  ]
