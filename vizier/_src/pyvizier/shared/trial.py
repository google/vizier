"""Wrapper classes for Trial protos and other messages in them.

Example usage:
  trial = Trial.from_proto(trial_proto)
  print('This trial's auc is: ', trial.final_measurement.metrics['auc'].value)
  print('This trial had parameter "n_hidden_layers": ',
        trial.parameters['n_hidden_layers'].value)
"""

import collections
from collections import abc as cabc
import copy
import datetime
import enum
from typing import Any, Dict, List, MutableMapping, Optional, Union, FrozenSet

from absl import logging
import attr
import numpy as np

from vizier._src.pyvizier.shared import common

ParameterValueTypes = Union[str, int, float, bool]
OrderedDict = collections.OrderedDict
Metadata = common.Metadata


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

  It has an additional field "std" for internal usage. This field gets lost
  when the object is converted to proto.
  """

  def _std_not_negative(self, _, stddev):
    if stddev < 0:
      raise ValueError(
          'Standard deviation must be a non-negative finite number.')

  value: float = attr.ib(
      converter=float,
      init=True,
      validator=[attr.validators.instance_of(float)],
      kw_only=False)
  std: float = attr.ib(
      converter=float,
      validator=[attr.validators.instance_of(float), _std_not_negative],
      init=True,
      default=0.0,
      kw_only=True)


# Use when you want to preserve the shapes or reduce if-else statements.
# e.g. `metrics.get('metric_name', NaNMetric).value` to get NaN or the actual
# value.
NaNMetric = Metric(value=np.nan)


@attr.s(auto_attribs=True, frozen=True, init=True, slots=True, repr=False)
class ParameterValue:
  """Immutable wrapper for vizier_pb2.Parameter.value, which is a oneof field.

  Has accessors (properties) that cast the value into the type according
  to StudyConfiguration class behavior. In particular, 'true' and 'false' are
  treated as special strings that are cast to a numeric value of 1 and 0,
  respectively, and boolean value of True and False, repectively.
  """

  value: ParameterValueTypes = attr.ib(
      init=True,
      validator=[
          attr.validators.instance_of((str, int, float, bool)),
      ])

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
    if self.value == 'true':
      return 1.0
    elif self.value == 'false':
      return 0.0
    try:
      # Note str -> float conversion exists for benchmark use.
      return float(self.value)
    except ValueError:
      return None

  @property
  def as_int(self) -> Optional[int]:
    """Returns the value cast to int."""
    if self.value == 'true':
      return 1
    elif self.value == 'false':
      return 0
    try:
      # Note str -> int conversion exists for benchmark use.
      return int(self.value)
    except ValueError:
      return None

  @property
  def as_str(self) -> Optional[str]:
    """Returns str-typed value or lowercase 'true'/'false' if value is bool."""
    if isinstance(self.value, bool):
      return str(self.value).lower()
    elif isinstance(self.value, str):
      return self.value
    # Note str conversion exists for benchmark use. Use __str__ instead.
    return str(self.value)

  @property
  def as_bool(self) -> Optional[bool]:
    """Returns the value as bool following StudyConfiguration's behavior.

    Returns: True if value is 'true' or 1. False if value is
      'false' or 0. For all other cases, returns None.
      For string type, this behavior is consistent with how
      StudyConfiguration.AddBooleanParameter's. For other types, this
      guarantees that self.value == self.as_bool
    """
    if isinstance(self.value, str):
      if self.value.lower() == 'true':
        return True
      elif self.value.lower() == 'false':
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
    return str(self.value)


class _MetricDict(collections.UserDict):

  def __setitem__(self, key: str, value: Union[float, Metric]):
    if isinstance(value, Metric):
      self.data.__setitem__(key, value)
    else:
      self.data.__setitem__(key, Metric(value=value))


@attr.s(auto_attribs=True, frozen=False, init=True, slots=True)
class Measurement:
  """Collection of metrics with a timestamp."""

  def _value_is_finite(self, _, value):
    if not (np.isfinite(value) and value >= 0):
      raise ValueError('Must be finite and non-negative.')

  # Should be used as a regular Dict.
  metrics: MutableMapping[str, Metric] = attr.ib(
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

  steps: float = attr.ib(
      converter=int,
      init=True,
      default=0,
      validator=[attr.validators.instance_of(int), _value_is_finite],
      on_setattr=[attr.setters.convert, attr.setters.validate],
      kw_only=True)


def _to_local_time(
    dt: Optional[datetime.datetime]) -> Optional[datetime.datetime]:
  """Converter for initializing timestamps in Trial class."""
  return dt.astimezone() if dt else None


@attr.define(init=False, frozen=True, eq=True)
class ParameterDict(cabc.MutableMapping):
  """Parameter dictionary.

  Maps the parameter names to their values. Works like a regular
  dict[str, ParameterValue] for the most part, except one can directly assign
  values of type `ParameterValueType`. So,
    ParameterDict(a=3) and
    ParameterDict(a=ParameterValue(3)) are equivalent.

  To access the raw value directly, use get_value() method.
    d.get('a').value == d.get_value('a')
  """

  _items: MutableMapping[str, ParameterValue] = attr.field(
      init=False, factory=dict)

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

  metadata: Metadata = attr.field(
      init=True,
      kw_only=True,
      factory=Metadata,
      validator=attr.validators.instance_of(Metadata))

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
  """Wrapper for learning_vizier.service.Trial proto."""
  id: int = attr.ib(
      init=True,
      kw_only=True,
      default=0,
      validator=attr.validators.instance_of(int),
  )

  _is_requested: bool = attr.ib(
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
      default=list(),
      validator=attr.validators.deep_iterable(
          member_validator=attr.validators.instance_of(Measurement),
          iterable_validator=attr.validators.instance_of(list)),
  )

  creation_time: Optional[datetime.datetime] = attr.ib(
      init=True,
      default=datetime.datetime.now(),
      converter=_to_local_time,
      kw_only=True,
      repr=lambda v: v.strftime('%x %X') if v is not None else 'None',
      validator=attr.validators.optional(
          attr.validators.instance_of(datetime.datetime)),
  )

  completion_time: Optional[datetime.datetime] = attr.ib(
      init=True,
      kw_only=True,
      default=None,
      repr=lambda v: v.strftime('%x %X') if v is not None else 'None',
      converter=_to_local_time,
      validator=attr.validators.optional(
          attr.validators.instance_of(datetime.datetime)),
  )

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
    elif self._is_requested:
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
               inplace: bool = True) -> 'Trial':
    """Completes the trial and returns it.

    Args:
      measurement: Measurement to complete the trial with.
      inplace: If True, Trial is modified in place. If False,  which is the
        default, then the operation is performed and it returns a copy of the
        object

    Returns:
      Completed Trial.
    """
    if inplace:
      # Use setattr. If we assign to self.final_measurement, then hyperref
      # mechanisms think this line is where `final_measurement` property
      # is defined, instead of where we declare attr.ib.
      self.__setattr__('final_measurement', copy.deepcopy(measurement))
      self.completion_time = _to_local_time(datetime.datetime.now())
      return self
    else:
      clone = copy.deepcopy(self)
      return clone.complete(measurement, inplace=True)


# Define aliases.
CompletedTrial = Trial
PendingTrial = Trial
CompletedTrialWithMeasurements = Trial
PendingTrialWithMeasurements = Trial


@attr.define
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
