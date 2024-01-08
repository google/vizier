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

"""Wrapper classes for Trial protos and other messages in them."""

import collections
from collections import abc
import copy
import dataclasses
import datetime
import enum
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Union, FrozenSet

from absl import logging
import attr
import numpy as np

from vizier._src.pyvizier.shared import common

ParameterValueTypes = Union[str, int, float, bool]

# TODO: These constants should be deleted.
TRUE_VALUE = 'True'
FALSE_VALUE = 'False'


class ParameterType(enum.Enum):
  """Valid Values for ParameterConfig.type."""
  DOUBLE = 'DOUBLE'
  INTEGER = 'INTEGER'
  CATEGORICAL = 'CATEGORICAL'
  DISCRETE = 'DISCRETE'
  CUSTOM = 'CUSTOM'

  def is_numeric(self) -> bool:
    return self in [self.DOUBLE, self.INTEGER, self.DISCRETE]

  def is_continuous(self) -> bool:
    return self == self.DOUBLE

  def _raise_type_error(self, value: ParameterValueTypes) -> None:
    raise TypeError(f'Type {self} is not compatible with value: {value}')

  def assert_correct_type(self, value: ParameterValueTypes) -> None:
    if self.is_numeric() and float(value) != value:
      self._raise_type_error(value)

    # TODO: Accepting boolean into categorical is unintuitive.
    elif (self
          == ParameterType.CATEGORICAL) and (not isinstance(value,
                                                            (str, bool))):
      self._raise_type_error(value)

    if self == self.INTEGER and int(value) != value:
      self._raise_type_error(value)


# TODO: Trial class should not depend on these.
class ExternalType(enum.Enum):
  """Valid Values for ParameterConfig.external_type."""
  INTERNAL = 'INTERNAL'
  BOOLEAN = 'BOOLEAN'
  INTEGER = 'INTEGER'
  FLOAT = 'FLOAT'


# Values should NEVER be removed from the enums below, only added.
class TrialStatus(enum.Enum):
  """Values for Trial.Status."""
  UNKNOWN = 'UNKNOWN'
  REQUESTED = 'REQUESTED'
  ACTIVE = 'ACTIVE'
  COMPLETED = 'COMPLETED'
  STOPPING = 'STOPPING'


@attr.s(frozen=True, init=True, slots=True, kw_only=False)
class Metric:
  """Enhanced immutable wrapper for vizier_pb2.Metric proto.

  It has an optional field "std" for internal usage. This field gets lost
  when the object is converted to proto.
  """

  def _std_not_negative(self, _, stddev: Optional[float]) -> bool:
    if (stddev is not None) and (not stddev >= 0):
      raise ValueError(
          'Standard deviation must be a non-negative finite number.')

  value: float = attr.ib(
      converter=float,
      init=True,
      validator=[attr.validators.instance_of(float)],
      kw_only=False)
  std: Optional[float] = attr.ib(
      converter=lambda x: float(x) if x is not None else None,
      validator=[
          attr.validators.optional(attr.validators.instance_of(float)),
          _std_not_negative
      ],
      init=True,
      default=None,
      kw_only=True)


# Use when you want to preserve the shapes or reduce if-else statements.
# e.g. `metrics.get('metric_name', NaNMetric).value` to get NaN or the actual
# value.
NaNMetric = Metric(value=np.nan)


# TODO: This class should be deleted in the future.
@attr.s(auto_attribs=True, frozen=True, init=True, slots=True, repr=False)
class ParameterValue:
  """Immutable wrapper for vizier_pb2.Parameter.value, which is a oneof field.

  Has accessors (properties) that cast the value into the type according
  to StudyConfiguration class behavior. In particular, 'true' and FALSE_VALUE
  are
  treated as special strings that are cast to a numeric value of 1 and 0,
  respectively, and boolean value of True and False, repectively.
  """

  value: ParameterValueTypes = attr.ib(
      init=True,
      validator=[
          attr.validators.instance_of((str, int, float, bool)),
      ])

  def cast_as_internal(self,
                       internal_type: ParameterType) -> ParameterValueTypes:
    """Cast to the internal type."""
    internal_type.assert_correct_type(self.value)

    if internal_type in (ParameterType.DOUBLE, ParameterType.DISCRETE):
      return self.as_float
    elif internal_type == ParameterType.INTEGER:
      return self.as_int
    elif internal_type == ParameterType.CATEGORICAL:
      return self.as_str
    else:
      raise RuntimeError(f'Unknown type {internal_type}')

  def cast(
      self,
      external_type: ExternalType,
  ) -> ParameterValueTypes:
    """Returns ParameterValue cast to external_type.

    Args:
      external_type:

    Returns:
      self.value if external_type is INTERNAL.
      self.as_bool if external_type is BOOLEAN.
      self.as_int if external_type is INTEGER.
      self.as_float if external_type is FLOAT.

    Raises:
      ValueError: If external_type is not valid.
    """
    if external_type == ExternalType.INTERNAL:
      return self.value
    elif external_type == ExternalType.BOOLEAN:
      return self.as_bool
    elif external_type == ExternalType.INTEGER:
      return self.as_int
    elif external_type == ExternalType.FLOAT:
      return self.as_float
    else:
      raise ValueError(
          'Unknown external type enum value: {}.'.format(external_type))

  @property
  def as_float(self) -> Optional[float]:
    """Returns the value cast to float."""
    if self.value == TRUE_VALUE:
      return 1.0
    elif self.value == FALSE_VALUE:
      return 0.0
    try:
      # Note str -> float conversion exists for benchmark use.
      return float(self.value)
    except ValueError:
      return None

  @property
  def as_int(self) -> Optional[int]:
    """Returns the value cast to int."""
    if self.value == TRUE_VALUE:
      return 1
    elif self.value == FALSE_VALUE:
      return 0
    try:
      # Note str -> int conversion exists for benchmark use.
      return int(self.value)
    except ValueError:
      return None

  @property
  def as_str(self) -> Optional[str]:
    """Returns str-typed value or 'True'/'False' if value is bool."""
    if isinstance(self.value, bool):
      if self.value:
        return TRUE_VALUE
      else:
        return FALSE_VALUE
    elif isinstance(self.value, str):
      return self.value
    # Note str conversion exists for benchmark use. Use __str__ instead.
    return str(self.value)

  @property
  def as_bool(self) -> Optional[bool]:
    """Returns the value as bool following StudyConfiguration's behavior.

    Returns: True if value is TRUE_VALUE or 1. False if value is
      FALSE_VALUE or 0. For all other cases, returns None.
      For string type, this behavior is consistent with how
      StudyConfiguration.AddBooleanParameter's. For other types, this
      guarantees that self.value == self.as_bool
    """
    if isinstance(self.value, str):
      if self.value == TRUE_VALUE:
        return True
      elif self.value == FALSE_VALUE:
        return False
    else:
      if self.value == 1.0:
        return True
      elif self.value == 0.0:
        return False
    return None

  def __str__(self) -> str:
    return str(self.value)

  def __repr__(self) -> str:
    return repr(self.value)


class _MetricDict(collections.UserDict, Mapping[str, Metric]):
  """Dictionary of string to metrics."""

  def get_value(self, key: str, default: float | None) -> float | None:
    if key in self.data:
      return self.data[key].value
    else:
      return default

  def __setitem__(self, key: str, value: Union[float, Metric]):
    if isinstance(value, Metric):
      self.data.__setitem__(key, value)
    else:
      self.data.__setitem__(key, Metric(value=value))

  def as_float_dict(self) -> dict[str, float]:
    return {k: m.value for k, m in self.data.items()}


@attr.s(auto_attribs=True, frozen=False, init=True, slots=True)
class Measurement:
  """A collection of metrics with a timestamp & checkpoint.

  metrics: Named, floating-point metrics.  A typical example would be the
    accuracy of a machine-learning model.  Typically, all the metrics mentioned
    in the MetricInformation class would be listed here; other metrics may be
    listed but would not normally be used by Vizier.

  elapsed_secs: (optional) The length of time it took to evaluate the Trial to
    reach this Measurement, in seconds.  This may be used by some Vizier
    algorithms.

  steps: (optional)  A positive integer roughly proportional to the amount of
    work spent.  When training a ML system, often this is a count of training
    steps/epochs.  When supplied, $steps should be consistent with the order of
    Measurements, but they need not be consecutive values.

  checkpoint_path: (optional) A slash-separated pathname.  Typically used when
    training a ML model; it would normally be a pathname for the checkpoint that
    produced the Measurement.  Implementations may limit the length of this
    string, but at least 2048 bytes will be allowed.
  """

  def _value_is_finite(self, _, value):
    if not (np.isfinite(value) and value >= 0):
      raise ValueError('Must be finite and non-negative.')

  # Should be used as a regular Dict.
  metrics: _MetricDict = attr.ib(
      init=True,
      converter=lambda d: _MetricDict(**d),
      default=_MetricDict(),
      validator=attr.validators.instance_of(_MetricDict),
      on_setattr=[attr.setters.convert, attr.setters.validate])

  elapsed_secs: float = attr.ib(
      converter=float,
      init=True,
      default=0,
      validator=[attr.validators.instance_of(float), _value_is_finite],
      on_setattr=[attr.setters.convert, attr.setters.validate],
      kw_only=True)

  # TODO: Change type annotation to int.
  steps: float = attr.ib(
      converter=int,
      init=True,
      default=0,
      validator=[attr.validators.instance_of(int), _value_is_finite],
      on_setattr=[attr.setters.convert, attr.setters.validate],
      kw_only=True)

  checkpoint_path: str = attr.ib(
      init=True,
      default='',
      validator=[attr.validators.instance_of(str)],
      kw_only=True,
  )


def _to_local_time(
    dt: Optional[datetime.datetime]) -> Optional[datetime.datetime]:
  """Converter for initializing timestamps in Trial class."""
  return dt.astimezone() if dt else None


# TODO: This class should have group() method that
# groups list parameters under the same key.
@attr.define(init=False, frozen=True, eq=True)
class ParameterDict(abc.MutableMapping):
  """Parameter dictionary.

  Maps the parameter names to their values. Works like a regular
  dict[str, ParameterValue] for the most part, except one can directly assign
  values of type `ParameterValueType`. So,
    ParameterDict(a=3) and
    ParameterDict(a=ParameterValue(3)) are equivalent.


  To access the raw value directly, use get_value() or as_dict():
    d.get_value('a') == d.get('a').value
    d.as_dict()['a'] == d.get_value('a')
  """

  _items: MutableMapping[str, ParameterValue] = attr.field(
      init=False, factory=dict)

  def as_dict(self) -> Dict[str, ParameterValueTypes]:
    """Returns the dict of parameter names to raw values."""
    return {k: self.get_value(k) for k in self._items}

  def __init__(self, iterable: Any = tuple(), **kwargs):
    self.__attrs_init__()
    self.update(iterable, **kwargs)

  def __setitem__(self, key: str, value: Union[ParameterValue,
                                               ParameterValueTypes]):
    if isinstance(value, ParameterValue):
      self._items[key] = value
    else:
      self._items[key] = ParameterValue(value)

  def __delitem__(self, key: str):
    del self._items[key]

  def __getitem__(self, key: str) -> ParameterValue:
    return self._items[key]

  def __len__(self) -> int:
    return len(self._items)

  def __iter__(self):
    return iter(self._items)

  def get_value(
      self,
      key: str,
      default: Optional[ParameterValueTypes] = None
  ) -> Optional[ParameterValueTypes]:
    """Returns the raw value of the given parameter name."""
    pv = self.get(key, default)
    if isinstance(pv, ParameterValue):
      return pv.value
    else:
      return pv


@attr.define(auto_attribs=True, frozen=False, init=True, slots=True)
class TrialSuggestion:
  """Freshly suggested trial.

  Suggestion can be converted to Trial object which has more functionalities.
  """

  parameters: ParameterDict = attr.field(
      init=True,
      factory=ParameterDict,
      converter=ParameterDict,
      validator=attr.validators.instance_of(ParameterDict))  # pytype: disable=wrong-arg-types

  metadata: common.Metadata = attr.field(
      init=True,
      kw_only=True,
      factory=common.Metadata,
      validator=attr.validators.instance_of(common.Metadata))

  def to_trial(self, uid: int = 0) -> 'Trial':
    """Assign an id and make it a Trial object.

    Usually SuggetedTrial objects are shorted-lived and not exposed to end
    users. This method is for non-service usage of trial suggestions in
    benchmarks, tests, colabs, etc.

    Args:
      uid: Trial id.

    Returns:
      Trial object.
    """
    return Trial(id=uid, parameters=self.parameters, metadata=self.metadata)


@attr.define(auto_attribs=True, frozen=False, init=True, slots=True)
class Trial(TrialSuggestion):
  """A Vizier Trial."""
  id: int = attr.ib(
      init=True,
      kw_only=True,
      default=0,
      validator=attr.validators.instance_of(int),
  )

  is_requested: bool = attr.ib(
      init=True,
      kw_only=True,
      default=False,
      validator=attr.validators.instance_of(bool))

  assigned_worker: Optional[str] = attr.ib(
      init=True,
      kw_only=True,
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(str)),
  )

  stopping_reason: Optional[str] = attr.ib(
      init=True,
      kw_only=True,
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(str)),
  )

  _infeasibility_reason: Optional[str] = attr.ib(
      init=True,
      kw_only=True,
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(str)),
  )

  description: Optional[str] = attr.ib(
      init=True,
      kw_only=True,
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(str)),
  )

  related_links: Dict[str, str] = attr.ib(
      init=True,
      kw_only=True,
      factory=dict,
      validator=attr.validators.deep_mapping(
          key_validator=attr.validators.instance_of(str),
          value_validator=attr.validators.instance_of(str),
          mapping_validator=attr.validators.instance_of(dict)),
  )  # pytype: disable=wrong-arg-types

  final_measurement: Optional[Measurement] = attr.ib(
      init=True,
      kw_only=True,
      default=None,
      validator=attr.validators.optional(
          attr.validators.instance_of(Measurement)),
  )

  measurements: List[Measurement] = attr.ib(
      init=True,
      kw_only=True,
      factory=list,
      validator=attr.validators.deep_iterable(
          member_validator=attr.validators.instance_of(Measurement),
          iterable_validator=attr.validators.instance_of(list)),
  )

  creation_time: Optional[datetime.datetime] = attr.ib(
      init=True,
      factory=datetime.datetime.now,
      converter=_to_local_time,
      kw_only=True,
      repr=lambda v: v.strftime('%x %X') if v is not None else 'None',
      validator=attr.validators.optional(
          attr.validators.instance_of(datetime.datetime)
      ),
  )

  completion_time: Optional[datetime.datetime] = attr.ib(
      init=True,
      kw_only=True,
      default=None,
      eq=False,
      repr=lambda v: v.strftime('%x %X') if v is not None else 'None',
      converter=_to_local_time,
      validator=attr.validators.optional(
          attr.validators.instance_of(datetime.datetime)
      ),
  )

  def __attrs_post_init__(self):
    if self.completion_time is None and (
        self.final_measurement is not None or self.infeasibility_reason
    ):
      self.completion_time = self.creation_time

  @property
  def duration(self) -> Optional[datetime.timedelta]:
    """Returns the duration of this Trial if it is completed, or None."""
    if self.completion_time:
      return self.completion_time - self.creation_time
    else:
      return None

  @property
  def status(self) -> TrialStatus:
    """Status.

    COMPLETED: Trial has final measurement or is declared infeasible.
    ACTIVE: Trial is being evaluated.
    STOPPING: Trial is being evaluated, but was decided to be not worth further
      evaluating.
    REQUESTED: Trial is queued for future suggestions.
    """
    if self.final_measurement is not None or self.infeasible:
      return TrialStatus.COMPLETED
    elif self.stopping_reason is not None:
      return TrialStatus.STOPPING
    elif self.is_requested:
      return TrialStatus.REQUESTED
    else:
      return TrialStatus.ACTIVE

  @property
  def is_completed(self) -> bool:
    """Returns True if this Trial is completed."""
    if self.status == TrialStatus.COMPLETED:
      if self.completion_time is None:
        logging.warning('Invalid Trial state: status is COMPLETED, but a '
                        ' completion_time was not set')
      return True
    elif self.completion_time is not None:
      if self.status is None:
        logging.warning('Invalid Trial state: status is not set to COMPLETED, '
                        'but a completion_time is set')
      return True
    return False

  @property
  def infeasible(self) -> bool:
    """Returns True if this Trial is infeasible."""
    return self._infeasibility_reason is not None

  @property
  def infeasibility_reason(self) -> Optional[str]:
    """Returns this Trial's infeasibility reason, if set."""
    return self._infeasibility_reason

  def complete(self,
               measurement: Measurement,
               *,
               infeasibility_reason: Optional[str] = None,
               inplace: bool = True) -> 'Trial':
    """Completes the trial and returns it.

    Args:
      measurement: Measurement to complete the trial with.
      infeasibility_reason: If set, completes the trial as infeasible. If the
        trial was already infeasible and infeasibility_reason is not set, the
        trial remains infeasible.
      inplace: If True, Trial is modified in place. If False, which is the
        default, then the operation is performed and it returns a copy of the
        object.

    Returns:
      Completed Trial.
    """
    if inplace:
      # Use setattr. If we assign to self.final_measurement, then hyperref
      # mechanisms think this line is where `final_measurement` property
      # is defined, instead of where we declare attr.ib.
      self.__setattr__('final_measurement', copy.deepcopy(measurement))
      if infeasibility_reason is not None:
        self.__setattr__('_infeasibility_reason', infeasibility_reason)
      self.completion_time = _to_local_time(datetime.datetime.now())
      return self
    else:
      clone = copy.deepcopy(self)
      return clone.complete(
          measurement, inplace=True, infeasibility_reason=infeasibility_reason)

  @property
  def final_measurement_or_die(self) -> Measurement:
    if self.final_measurement is None:
      raise ValueError(f'Trial is missing final_measurement: {self}')
    return self.final_measurement


# Define aliases.
CompletedTrial = Trial
PendingTrial = Trial
CompletedTrialWithMeasurements = Trial
PendingTrialWithMeasurements = Trial


@attr.define(kw_only=True)
class TrialFilter:
  """Trial filter.

  All filters are by default 'AND' conditions.

  Attributes:
    ids: If set, requires the trial's id to be in the set.
    min_id: If set, requires the trial's id to be at least this number.
    max_id: If set, requires the trial's id to be at most this number.
    status: If set, requires the trial's status to be in the set.
  """
  ids: Optional[FrozenSet[int]] = attr.field(
      default=None,
      converter=lambda x: frozenset(x) if x is not None else None,
      validator=attr.validators.optional(
          attr.validators.deep_iterable(
              attr.validators.instance_of(int),
              attr.validators.instance_of(frozenset))))
  min_id: Optional[int] = attr.field(default=None)
  max_id: Optional[int] = attr.field(default=None)
  status: Optional[FrozenSet[TrialStatus]] = attr.field(
      default=None,
      converter=lambda x: frozenset(x) if x is not None else None,
      validator=attr.validators.optional(
          attr.validators.deep_iterable(
              attr.validators.instance_of(TrialStatus),
              attr.validators.instance_of(frozenset))))

  # TODO: Add "search_space" argument

  def __call__(self, trial: Trial) -> bool:
    if self.ids is not None:
      if trial.id not in self.ids:
        return False
    if self.min_id is not None:
      if trial.id < self.min_id:
        return False
    if self.max_id is not None:
      if trial.id > self.max_id:
        return False
    if self.status is not None:
      if trial.status not in self.status:
        return False
    return True


@dataclasses.dataclass(frozen=True)
class MetadataDelta:
  """Carries cumulative delta for a batch metadata update.

  Attributes:
    on_study: Updates to be made on study-level metadata.
    on_trials: Maps trial id to updates.
  """

  on_study: common.Metadata = dataclasses.field(default_factory=common.Metadata)

  on_trials: Dict[int, common.Metadata] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(common.Metadata))

  def __bool__(self):
    """Returns True if this carries any Metadata items."""
    if self.on_study:
      return True
    for v in self.on_trials.values():
      if v:
        return True
    return False

  def on_trial(self, trial_id: int) -> common.Metadata:
    """Enables easy assignment to a single Trial."""
    return self.on_trials[trial_id]

  def assign(self,
             namespace: str,
             key: str,
             value: common.MetadataValue,
             *,
             trial: Optional[Trial] = None,
             trial_id: Optional[int] = None):
    """Assigns metadata.

    Args:
      namespace: Namespace of the metadata. See common.Metadata doc for more
        details.
      key:
      value:
      trial: If specified, `trial_id` must be None. It behaves the same as when
        `trial_id=trial.id`, and additionally, the metadata is added to `trial`.
      trial_id: If specified, `trial` must be None. If both `trial` and
        `trial_id` are None, then the key-value pair will be assigned to the
        study.

    Raises:
      ValueError:
    """
    if trial is None and trial_id is None:
      self.on_study.ns(namespace)[key] = value
    elif trial is not None and trial_id is not None:
      raise ValueError(
          'At most one of `trial` and `trial_id` can be specified.')
    elif trial is not None:
      self.on_trials[trial.id].ns(namespace)[key] = value
      trial.metadata.ns(namespace)[key] = value
    elif trial_id is not None:
      self.on_trials[trial_id].ns(namespace)[key] = value
