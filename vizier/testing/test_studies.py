"""Test study generator."""

from typing import Collection
from vizier import pyvizier as vz


def flat_space_with_all_types() -> vz.SearchSpace:
  """Search space with all parameter types."""

  space = vz.SearchSpace()
  root = space.select_root()
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


def metrics_objective_goals() -> Collection[vz.MetricInformation]:
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


def metrics_all_unconstrained() -> Collection[vz.MetricInformation]:
  return [
      vz.MetricInformation('gain', goal=vz.ObjectiveMetricGoal.MAXIMIZE),
      vz.MetricInformation('loss', goal=vz.ObjectiveMetricGoal.MINIMIZE),
      vz.MetricInformation(
          'gt2', goal=vz.ObjectiveMetricGoal.MAXIMIZE, safety_threshold=2.0),
      vz.MetricInformation(
          'lt2', goal=vz.ObjectiveMetricGoal.MINIMIZE, safety_threshold=2.0),
  ]


def metrics_all_constrained() -> Collection[vz.MetricInformation]:
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
