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

"""Essential classes for defining a blackbox optimization problem.

Contains `ProblemStatement` and its components that are cross platform
compatible.
"""
import collections
from typing import Collection
import enum
from typing import Callable, Iterable, Iterator, List, Optional, Type, TypeVar, Union
import attr
import numpy as np
from vizier._src.pyvizier.shared import common
from vizier._src.pyvizier.shared import parameter_config

################### PyTypes ###################
ScaleType = parameter_config.ScaleType
ExternalType = parameter_config.ExternalType
# A sequence of possible internal parameter values.
MonotypeParameterSequence = parameter_config.MonotypeParameterSequence
_T = TypeVar('_T')


################### Helper Classes ###################
def _min_leq_max(instance: 'MetricInformation', _, value: float):
  if value > instance.max_value:
    raise ValueError(
        f'min_value={value} cannot exceed max_value={instance.max_value}.'
    )


def _max_geq_min(instance: 'MetricInformation', _, value: float):
  if value < instance.min_value:
    raise ValueError(
        f'min_value={instance.min_value} cannot exceed max_value={value}.'
    )


# Values should NEVER be removed from ObjectiveMetricGoal, only added.
class ObjectiveMetricGoal(enum.IntEnum):
  """Valid Values for MetricInformation.Goal."""

  MAXIMIZE = 1
  MINIMIZE = 2

  # pylint: disable=comparison-with-callable
  @property
  def is_maximize(self) -> bool:
    return self == self.MAXIMIZE

  @property
  def is_minimize(self) -> bool:
    return self == self.MINIMIZE


class MetricType(enum.Enum):
  """Type of the metric.

  OBJECTIVE: Objective to be maximized / minimized.
  SAFETY: Objective to be kept above / below a certain threshold.
  """

  OBJECTIVE = 'OBJECTIVE'
  SAFETY = 'SAFETY'  # Soft constraint

  # pylint: disable=comparison-with-callable
  @property
  def is_safety(self) -> bool:
    return self == MetricType.SAFETY

  @property
  def is_objective(self) -> bool:
    return self == MetricType.OBJECTIVE


@attr.define(frozen=False, init=True, slots=True)
class MetricInformation:
  """MetricInformation provides optimization metrics configuration."""

  # The name of this metric. An empty string is allowed for single-metric
  # optimizations.
  name: str = attr.field(
      init=True, default='', validator=attr.validators.instance_of(str)
  )

  goal: ObjectiveMetricGoal = attr.field(
      init=True,
      # pylint: disable=g-long-lambda
      converter=ObjectiveMetricGoal,
      validator=attr.validators.instance_of(ObjectiveMetricGoal),
      on_setattr=[attr.setters.convert, attr.setters.validate],
      kw_only=True,
  )

  # The following should be used to configure this as a safety metric.
  # safety_threshold must always be set (to a float) for safety metrics.
  safety_threshold: Optional[float] = attr.field(
      init=True,
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(float)),
      kw_only=True,
  )
  # Safety_std_threshold is DEPRECATED and is here for backward compatibility.
  # To configure how cautious the optimization should be, please defer to
  # desired_min_safe_trials_fraction. This corresponds to the allowed
  # probability threshold (as a function of the z-score) of
  # violating the safety_threshold.
  safety_std_threshold: Optional[float] = attr.field(
      init=True,
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(float)),
      kw_only=True,
  )
  # Desired minimum fraction of safe trials (over total number of trials)
  # that should be targeted by the algorithm during the entire duration of the
  # study (best effort). A value of 0.0 is the same as None. If unset, or set
  # to 0.0, then Vizier has no constraint on the number/fraction of unsafe
  # trials it can suggest.
  desired_min_safe_trials_fraction: Optional[float] = attr.field(
      init=True,
      default=None,
      validator=[
          attr.validators.optional(attr.validators.instance_of(float)),
          attr.validators.optional(attr.validators.le(1.0)),
          attr.validators.optional(attr.validators.ge(0.0)),
      ],
      kw_only=True,
  )

  # Minimum value of this metric can be optionally specified.
  min_value: float = attr.field(
      init=True,
      default=None,
      # FYI: Converter is applied before validator.
      converter=lambda x: float(x) if x is not None else -np.inf,
      validator=[attr.validators.instance_of(float), _min_leq_max],
      kw_only=True,
  )

  # Maximum value of this metric can be optionally specified.
  max_value: float = attr.field(
      init=True,
      default=None,
      # FYI: Converter is applied before validator.
      converter=lambda x: float(x) if x is not None else np.inf,
      validator=[attr.validators.instance_of(float), _max_geq_min],
      on_setattr=attr.setters.validate,
      kw_only=True,
  )

  def min_value_or(self, default_value_fn: Callable[[], float]) -> float:
    """Returns the minimum value if finite, or default_value_fn().

    Avoids the common pitfalls of using
      `metric.min_value or default_value`
    which would incorrectly use the default_value when min_value == 0, and
    requires default_value to have been computed.

    Args:
      default_value_fn: Default value if min_value is not finite. This function
        does not run at all if min_value is finite.
    """
    if np.isfinite(self.min_value):
      return self.min_value
    else:
      return default_value_fn()

  def max_value_or(self, default_value_fn: Callable[[], float]) -> float:
    """Returns the minimum value if finite, or default_value_fn().

    Avoids the common pitfalls of using
      `metric.max_value or default_value`
    which would incorrectly use the default_value when max_value == 0, and
    requires default_value to have been computed.

    Args:
      default_value_fn: Default value if max_value is not finite. This function
        does not run at all if max_value is configured.
    """
    if np.isfinite(self.max_value):
      return self.max_value
    else:
      return default_value_fn()

  @property
  def range(self) -> float:
    """Range of the metric. Can be infinite."""
    return self.max_value - self.min_value

  @property
  def type(self) -> MetricType:
    if self.safety_threshold is not None:
      return MetricType.SAFETY
    else:
      return MetricType.OBJECTIVE

  def flip_goal(self) -> 'MetricInformation':
    """Flips the goal in-place and returns the reference to self."""
    if self.goal == ObjectiveMetricGoal.MAXIMIZE:
      self.goal = ObjectiveMetricGoal.MINIMIZE
    else:
      self.goal = ObjectiveMetricGoal.MAXIMIZE
    return self


@attr.define(frozen=False, init=True, slots=True)
class MetricsConfig(Collection[MetricInformation]):
  """Container for metrics.

  Metric names should be unique.
  """

  _metrics: List[MetricInformation] = attr.ib(
      init=True,
      factory=list,
      converter=list,
      validator=attr.validators.deep_iterable(
          member_validator=attr.validators.instance_of(MetricInformation),
          iterable_validator=attr.validators.instance_of(Iterable),
      ),
  )

  def item(self) -> MetricInformation:
    if len(self._metrics) != 1:
      raise ValueError(
          'item() may only be called when there is exactly one '
          'metric (there are %d).'
          % len(self._metrics)
      )
    return self._metrics[0]

  def _assert_names_are_unique(self) -> None:
    counts = collections.Counter(metric.name for metric in self._metrics)
    if len(counts) != len(self._metrics):
      for name, count in counts.items():
        if count > 1:
          raise ValueError(f'Duplicate metric name: {name} in {self._metrics}')

  def __attrs_post_init__(self):
    self._assert_names_are_unique()

  def __iter__(self) -> Iterator[MetricInformation]:
    return iter(self._metrics)

  def __contains__(self, x: object) -> bool:
    return x in self._metrics

  def __len__(self) -> int:
    return len(self._metrics)

  def __add__(self, metrics: Iterable[MetricInformation]) -> 'MetricsConfig':
    return MetricsConfig(self._metrics + list(metrics))

  def of_type(
      self, include: Union[MetricType, Iterable[MetricType]]
  ) -> 'MetricsConfig':
    """Filters the Metrics by type."""
    if isinstance(include, MetricType):
      include = (include,)
    return MetricsConfig(m for m in self._metrics if m.type in include)

  def exclude_type(
      self, exclude: Union[MetricType, Iterable[MetricType]]
  ) -> 'MetricsConfig':
    """Filters out the Metrics by type."""
    if isinstance(exclude, MetricType):
      exclude = (exclude,)
    return MetricsConfig(m for m in self._metrics if m.type not in exclude)

  def append(self, metric: MetricInformation):
    self._metrics.append(metric)
    self._assert_names_are_unique()

  def extend(self, metrics: Iterable[MetricInformation]):
    for metric in metrics:
      self.append(metric)

  @property
  def is_single_objective(self) -> bool:
    """Returns True if only one objective metric is configured."""
    return len(self.of_type(MetricType.OBJECTIVE)) == 1

  @property
  def is_safety_metric(self) -> bool:
    """Returns True if at least one safety metric is configured."""
    return True if self.of_type(MetricType.SAFETY) else False


################### Main Class ###################
@attr.define(frozen=False, init=True, slots=True)
class ProblemStatement:
  """Defines a blackbox optimization problem.

  `ProblemStatement` contains the minimal information that defines the search
  problem. It is inherited by platform-specific classes that carry additional
  platform-specific configurations including which algorithm to use.

  Each of OSS, Vertex, and Google Vizier has their own implementations of
  `StudyConfig` that inherit from `ProblemStatement`.

  Pythia `Policy` interface uses `ProblemStatement` as opposed to `StudyConfig`
  so that the same algorithm code can be used across platforms.
  """

  search_space: parameter_config.SearchSpace = attr.ib(
      init=True,
      factory=parameter_config.SearchSpace,
      validator=attr.validators.instance_of(parameter_config.SearchSpace),
      on_setattr=[attr.setters.convert, attr.setters.validate],
  )
  # TODO: This name/type combo is confusing.
  metric_information: MetricsConfig = attr.ib(
      init=True,
      factory=MetricsConfig,
      converter=MetricsConfig,
      validator=attr.validators.instance_of(MetricsConfig),
      on_setattr=[attr.setters.convert, attr.setters.validate],
      kw_only=True,
  )

  metadata: common.Metadata = attr.field(
      init=True,
      kw_only=True,
      factory=common.Metadata,
      validator=attr.validators.instance_of(common.Metadata),
      on_setattr=[attr.setters.convert, attr.setters.validate],
  )

  @property
  def debug_info(self) -> str:
    return ''

  @classmethod
  def from_problem(cls: Type[_T], problem: 'ProblemStatement') -> _T:
    """Converts a ProblemStatement to a subclass instance.

    Note that this method is useful in subclasses but not so much in
    `ProblemStatement` itself. `ProblemStatement.from_problem` simply generates
    a (shallow) copy of `problem`.

    Args:
      problem:

    Returns:
      A subclass instance filled with shallow copies of `ProblemStatement`
      fields.
    """
    return cls(
        search_space=problem.search_space,
        metric_information=problem.metric_information,
        metadata=problem.metadata,
    )

  def to_problem(self) -> 'ProblemStatement':
    """Converts to a ProblemStatement which is the parent class of `self`.

    Note that this method is useful in subclasses but not so much in
    `ProblemStatement` itself. `ProblemStatement.to_problem` simply generates
    a (shallow) copy of `problem`.

    Returns:
      `ProblemStatement` filled with shallow copies of `self.
    """
    return ProblemStatement(
        search_space=self.search_space,
        metric_information=self.metric_information,
        metadata=self.metadata,
    )

  @property
  def is_single_objective(self) -> bool:
    """Returns True if only one objective metric is configured."""
    return self.metric_information.is_single_objective

  @property
  def single_objective_metric_name(self) -> Optional[str]:
    """Returns the name of the single-objective metric, if set.

    Returns:
      String: name of the single-objective metric.
      None: if this is not a single-objective study.
    """
    if self.is_single_objective:
      return self.metric_information.of_type(MetricType.OBJECTIVE).item().name
    return None

  @property
  def is_safety_metric(self) -> bool:
    """Returns True if at least one safety metric is configured."""
    return self.metric_information.is_safety_metric
