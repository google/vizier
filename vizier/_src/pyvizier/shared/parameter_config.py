# Copyright 2022 Google LLC.
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

"""ParameterConfig wraps ParameterConfig and ParameterSpec protos."""

import collections
import copy
import enum
import math
import re
from typing import Iterable, Generator, List, Optional, Sequence, Tuple, Union, overload

from absl import logging
import attr

from vizier._src.pyvizier.shared import trial

ExternalType = trial.ExternalType


class ParameterType(enum.Enum):
  """Valid Values for ParameterConfig.type."""
  DOUBLE = 'DOUBLE'
  INTEGER = 'INTEGER'
  CATEGORICAL = 'CATEGORICAL'
  DISCRETE = 'DISCRETE'

  def is_numeric(self) -> bool:
    return self in [self.DOUBLE, self.INTEGER, self.DISCRETE]

  def is_continuous(self) -> bool:
    return self == self.DOUBLE


class ScaleType(enum.Enum):
  """Valid Values for ParameterConfig.scale_type."""
  LINEAR = 'LINEAR'
  LOG = 'LOG'
  REVERSE_LOG = 'REVERSE_LOG'
  UNIFORM_DISCRETE = 'UNIFORM_DISCRETE'

  def is_nonlinear(self) -> bool:
    return self in [self.LOG, self.REVERSE_LOG]


# A sequence of possible internal parameter values.
MonotypeParameterSequence = Union[Sequence[Union[int, float]], Sequence[str]]
MonotypeParameterList = Union[List[Union[int, float]], List[str]]


def _validate_bounds(bounds: Union[Tuple[int, int], Tuple[float, float]]):
  """Validates the bounds."""
  if len(bounds) != 2:
    raise ValueError('Bounds must have length 2. Given: {}'.format(bounds))
  lower = bounds[0]
  upper = bounds[1]
  if not all([math.isfinite(v) for v in (lower, upper)]):
    raise ValueError(
        'Both "lower" and "upper" must be finite. Given: (%f, %f)' %
        (lower, upper))
  if lower > upper:
    raise ValueError(
        'Lower cannot be greater than upper: given lower={} upper={}'.format(
            lower, upper))


def _get_feasible_points_and_bounds(
    feasible_values: Sequence[float]
) -> Tuple[List[float], Union[Tuple[int, int], Tuple[float, float]]]:
  """Validates and converts feasible values to floats."""
  if not all([math.isfinite(p) for p in feasible_values]):
    raise ValueError('Feasible values must all be finite. Given: {}' %
                     feasible_values)

  feasible_points = list(sorted(feasible_values))
  bounds = (feasible_points[0], feasible_points[-1])
  return feasible_points, bounds


def _get_categories(categories: Sequence[str]) -> List[str]:
  """Returns the categories."""
  return sorted(list(categories))


def _get_default_value(
    param_type: ParameterType,
    default_value: Union[float, int, str]) -> Union[float, int, str]:
  """Validates and converts the default_value to the right type."""
  if (param_type in (ParameterType.DOUBLE, ParameterType.DISCRETE) and
      (isinstance(default_value, float) or isinstance(default_value, int))):
    return float(default_value)
  elif (param_type == ParameterType.INTEGER and
        (isinstance(default_value, float) or isinstance(default_value, int))):
    if isinstance(default_value, int):
      return default_value
    else:
      # Check if the float rounds nicely.
      default_int_value = round(default_value)
      if not math.isclose(default_value, default_int_value):
        raise ValueError('default_value for an INTEGER parameter should be an '
                         'integer, got float: [{}]'.format(default_value))
      return default_int_value
  elif (param_type == ParameterType.CATEGORICAL and
        isinstance(default_value, str)):
    return default_value
  raise ValueError(
      'default_value has an incorrect type. ParameterType has type {}, '
      'but default_value has type {}'.format(param_type.name,
                                             type(default_value)))


#######################
# Experimental features
#######################
class FidelityMode(enum.Enum):
  """Decides how the fidelity config should be interpreated.

  SEQUENTIAL: A high fidelity measurement can be "warm-started" from a lower
    fidelity measurement. Currently, no algorithms can take advatange of it, and
    Vizier behaves exactly like NON_SEQUENTIAL case. This is for tracking
    purposes only.

  NOT_SEQUENTIAL: Each fidelity is separately measured. Example: Fidelity
    is the fraction of dataset to train on.

  STEPS: Fidelity determines the maximum value for Measurement.steps reported
    to Vizier. There is one-to-one correspondence between steps and fidelity.
    A high fideltiy Trial's measurements contain lower fidelity evaluations.
    When this is enabled, suggestion models do not use
    Trials' final_measurement. Instead, it reads the measurements whose
    "steps" exactly match one of the fidelities, and treats them as if they
    were separate Trials. Example: Fidelity is the number of total epochs
    to train on.
  """
  SEQUENTIAL = 'SEQUENTIAL'
  NOT_SEQUENTIAL = 'NOT_SEQUENTIAL'
  STEPS = 'STEPS'


@attr.define
class FidelityConfig:
  mode: FidelityMode = attr.field(converter=FidelityMode)
  cost_ratio: Sequence[float] = attr.field(
      converter=tuple, default=tuple(), kw_only=True)


########################
# Experimental features end here
########################


@attr.s(auto_attribs=True, frozen=True, init=True, slots=True)
class ParameterConfig:
  """A Vizier ParameterConfig.

  Use ParameterConfig.factory to create a valid instance.
  """
  _name: str = attr.ib(
      init=True, validator=attr.validators.instance_of(str), kw_only=True)
  _type: ParameterType = attr.ib(
      init=True,
      validator=attr.validators.instance_of(ParameterType),
      repr=lambda v: v.name if v is not None else 'None',
      kw_only=True)
  # Only one of _feasible_values, _bounds will be set at any given time.
  _bounds: Optional[Union[Tuple[int, int], Tuple[float, float]]] = attr.ib(
      init=True,
      validator=attr.validators.optional(
          attr.validators.deep_iterable(
              member_validator=attr.validators.instance_of((int, float)),
              iterable_validator=attr.validators.instance_of(tuple))),
      kw_only=True)
  _feasible_values: Optional[MonotypeParameterList] = attr.ib(
      init=True,
      validator=attr.validators.optional(
          attr.validators.deep_iterable(
              member_validator=attr.validators.instance_of((int, float, str)),
              iterable_validator=attr.validators.instance_of((list, tuple)))),
      kw_only=True)
  _scale_type: Optional[ScaleType] = attr.ib(
      init=True,
      validator=attr.validators.optional(
          attr.validators.instance_of(ScaleType)),
      repr=lambda v: v.name if v is not None else 'None',
      kw_only=True)
  _default_value: Optional[Union[float, int, str]] = attr.ib(
      init=True,
      validator=attr.validators.optional(
          attr.validators.instance_of((float, int, str))),
      kw_only=True)
  _external_type: ExternalType = attr.ib(
      init=True,
      converter=lambda v: v or ExternalType.INTERNAL,
      validator=attr.validators.optional(
          attr.validators.instance_of(ExternalType)),
      repr=lambda v: v.name if v is not None else 'None',
      kw_only=True)
  # Parent values for this ParameterConfig. If set, then this is a child
  # ParameterConfig.
  _matching_parent_values: Optional[MonotypeParameterList] = attr.ib(
      init=True,
      validator=attr.validators.optional(
          attr.validators.deep_iterable(
              member_validator=attr.validators.instance_of((int, float, str)),
              iterable_validator=attr.validators.instance_of((list, tuple)))),
      kw_only=True)
  # Children ParameterConfig. If set, then this is a parent ParameterConfig.
  _child_parameter_configs: Optional[List['ParameterConfig']] = attr.ib(
      init=True, kw_only=True)

  # Experimental feature.
  fidelity_config: Optional[FidelityConfig] = attr.ib(
      init=True,
      default=None,
      kw_only=True,
  )

  # Pytype treats instances of EnumTypeWrapper as types, but they can't be
  # evaluated at runtime, so a Union[] of proto enums has to be a forward
  # reference below.
  @classmethod
  def factory(
      cls,
      name: str,
      *,
      bounds: Optional[Union[Tuple[int, int], Tuple[float, float]]] = None,
      feasible_values: Optional[MonotypeParameterSequence] = None,
      children: Optional[Sequence[Tuple[MonotypeParameterSequence,
                                        'ParameterConfig']]] = None,
      fidelity_config: Optional[FidelityConfig] = None,
      scale_type: Optional[ScaleType] = None,
      default_value: Optional[Union[float, int, str]] = None,
      external_type: Optional[ExternalType] = ExternalType.INTERNAL
  ) -> 'ParameterConfig':
    """Factory method.

    Args:
      name: The parameter's name. Cannot be empty.
      bounds: REQUIRED for INTEGER or DOUBLE type. Specifies (min, max). The
        type of (min, max) determines the created ParameterConfig's type.
      feasible_values: REQUIRED for DISCRETE or CATEGORICAL type. The elements'
        type determines the created ParameterConfig's type.
      children: sequence of tuples formatted as: (matching_parent_values,
        ParameterConfig). ONLY THE TYPES ARE VALIDATED. If the child
        ParameterConfig protos already have parent values set, they will be
        overridden by the provided matching_parent_values.
      fidelity_config: Fidelity config.  NOT VALIDATED.
      scale_type: Scaling to be applied. NOT VALIDATED.
      default_value: A default value for the Parameter.
      external_type: An annotation indicating the type this parameter should be
        cast to.

    Returns:
      A ParameterConfig object which wraps a partially validated proto.

    Raises:
      ValueError: Exactly one of feasible_values and bounds must be convertible
        to Boolean true. Bounds and numeric feasible_values must be finite.
        Bounds and feasible_values, if provided, must consist of
        elements of the same type.
      TypeError: If children's matching_parent_values are not compatible with
        the ParameterConfig being created.
    """
    if not name:
      raise ValueError('Parameter name cannot be empty.')

    if bool(feasible_values) == bool(bounds):
      raise ValueError(
          'While creating Parameter with name={}: exactly one of '
          '"feasible_values" or "bounds" must be provided, but given '
          'feasible_values={} and bounds={}.'.format(name, feasible_values,
                                                     bounds))
    if feasible_values:
      if len(set(feasible_values)) != len(feasible_values):
        counter = collections.Counter(feasible_values)
        duplicate_dict = {k: v for k, v in counter.items() if v > 1}
        raise ValueError(
            'Feasible values cannot have duplicates: {}'.format(duplicate_dict))
      if all(isinstance(v, (float, int)) for v in feasible_values):
        inferred_type = ParameterType.DISCRETE
        feasible_values, bounds = _get_feasible_points_and_bounds(
            feasible_values)
      elif all(isinstance(v, str) for v in feasible_values):
        inferred_type = ParameterType.CATEGORICAL
        feasible_values = _get_categories(feasible_values)
      else:
        raise ValueError(
            'Feasible values must all be numeric or strings. Given {}'.format(
                feasible_values))
    else:  # bounds were specified.
      if isinstance(bounds[0], int) and isinstance(bounds[1], int):
        inferred_type = ParameterType.INTEGER
        _validate_bounds(bounds)
      elif isinstance(bounds[0], float) and isinstance(bounds[1], float):
        inferred_type = ParameterType.DOUBLE
        _validate_bounds(bounds)
      else:
        raise ValueError(
            'Bounds must both be integers or doubles. Given: {}'.format(bounds))

    if default_value is not None:
      default_value = _get_default_value(inferred_type, default_value)

    pc = cls(
        name=name,
        type=inferred_type,
        bounds=bounds,
        feasible_values=feasible_values,
        scale_type=scale_type,
        default_value=default_value,
        fidelity_config=fidelity_config,
        external_type=external_type,
        matching_parent_values=None,
        child_parameter_configs=None)
    if children:
      pc = pc.add_children(children)
    return pc

  @property
  def name(self) -> str:
    return self._name

  @property
  def type(self) -> ParameterType:
    return self._type

  @property
  def external_type(self) -> ExternalType:
    return self._external_type

  @property
  def scale_type(self) -> Optional[ScaleType]:
    return self._scale_type

  @property
  def bounds(self) -> Union[Tuple[float, float], Tuple[int, int]]:
    """Returns the bounds, if set, or raises a ValueError."""
    if self.type == ParameterType.CATEGORICAL:
      raise ValueError('Accessing bounds of a categorical parameter: %s' %
                       self.name)
    return self._bounds

  @property
  def matching_parent_values(self) -> MonotypeParameterList:
    """Returns the matching parent values, if this is a child parameter."""
    if not self._matching_parent_values:
      return []
    return copy.copy(self._matching_parent_values)

  @property
  def child_parameter_configs(self) -> List['ParameterConfig']:
    if not self._child_parameter_configs:
      return []
    return copy.deepcopy(self._child_parameter_configs)

  def _del_child_parameter_configs(self):
    """Deletes the current child ParameterConfigs."""
    object.__setattr__(self, '_child_parameter_configs', None)

  @property
  def clone_without_children(self) -> 'ParameterConfig':
    """Returns the clone of self, without child_parameter_configs."""
    clone = copy.deepcopy(self)
    clone._del_child_parameter_configs()  # pylint: disable='protected-access'
    return clone

  @property
  def feasible_values(self) -> Union[List[int], List[float], List[str]]:
    if self.type in (ParameterType.DISCRETE, ParameterType.CATEGORICAL):
      if not self._feasible_values:
        return []
      return copy.copy(self._feasible_values)
    elif self.type == ParameterType.INTEGER:
      return list(range(self.bounds[0], self.bounds[1] + 1))
    raise ValueError('feasible_values is invalid for type: %s' % self.type)

  @property
  def default_value(self) -> Optional[Union[int, float, str]]:
    """Returns the default value, or None if not set."""
    return self._default_value

  def _set_matching_parent_values(self,
                                  parent_values: MonotypeParameterSequence):
    """Sets the given matching parent values in this object, without validation.

    Args:
      parent_values: Parent values for which this child ParameterConfig is
        active. Existing values will be replaced.
    """
    object.__setattr__(self, '_matching_parent_values', list(parent_values))

  def _set_child_parameter_configs(self, children: List['ParameterConfig']):
    """Sets the given child ParameterConfigs in this object, without validation.

    Args:
      children: The children to set in this object. Existing children will be
        replaced.
    """
    object.__setattr__(self, '_child_parameter_configs', children)

  def add_children(
      self, new_children: Sequence[Tuple[MonotypeParameterSequence,
                                         'ParameterConfig']]
  ) -> 'ParameterConfig':
    """Clones the ParameterConfig and adds new children to it.

    Args:
      new_children: A sequence of tuples formatted as: (matching_parent_values,
        ParameterConfig). If the child ParameterConfig have pre-existing parent
        values, they will be overridden.

    Returns:
      A parent parameter config, with children set.

    Raises:
      ValueError: If the child configs are invalid
      TypeError: If matching parent values are invalid
    """
    parent = copy.deepcopy(self)
    if not new_children:
      return parent

    for child_pair in new_children:
      if len(child_pair) != 2:
        raise ValueError('Each element in new_children must be a tuple of '
                         '(Sequence of valid parent values,  ParameterConfig),'
                         ' given: {}'.format(child_pair))

    logging.debug('add_children: new_children=%s', new_children)
    child_parameter_configs = parent.child_parameter_configs
    for unsorted_parent_values, child in new_children:
      parent_values = sorted(unsorted_parent_values)
      child_copy = copy.deepcopy(child)
      if parent.type == ParameterType.DISCRETE:
        if not all(isinstance(v, (float, int)) for v in parent_values):
          raise TypeError('Parent is DISCRETE-typed, but a child is specifying '
                          'one or more non float/int parent values: child={} '
                          ', parent_values={}'.format(child, parent_values))
        child_copy._set_matching_parent_values(parent_values)  # pylint: disable='protected-access'
      elif parent.type == ParameterType.CATEGORICAL:
        if not all(isinstance(v, str) for v in parent_values):
          raise TypeError('Parent is CATEGORICAL-typed, but a child is '
                          'specifying one or more non float/int parent values: '
                          'child={}, parent_values={}'.format(
                              child, parent_values))
        child_copy._set_matching_parent_values(parent_values)  # pylint: disable='protected-access'
      elif parent.type == ParameterType.INTEGER:
        # Allow {int, float}->float conversion but block str->float conversion.
        int_values = [int(v) for v in parent_values]
        if int_values != parent_values:
          raise TypeError(
              'Parent is INTEGER-typed, but a child is specifying one or more '
              'non-integral parent values: {}'.format(parent_values))
        child_copy._set_matching_parent_values(int_values)  # pylint: disable='protected-access'
      else:
        raise ValueError('DOUBLE type cannot have child parameters')
      child_parameter_configs.extend([child_copy])
    parent._set_child_parameter_configs(child_parameter_configs)  # pylint: disable='protected-access'
    return parent

  def continuify(self) -> 'ParameterConfig':
    """Returns a newly created DOUBLE parameter with the same range."""
    if self.type == ParameterType.DOUBLE:
      return copy.deepcopy(self)
    elif not ParameterType.is_numeric(self.type):
      raise ValueError(
          'Cannot convert a non-numeric parameter to DOUBLE: {}'.format(self))
    elif self._child_parameter_configs:
      raise ValueError(
          'Cannot convert a parent parameter to DOUBLE: {}'.format(self))

    scale_type = self.scale_type
    if scale_type == ScaleType.UNIFORM_DISCRETE:
      logging.log_every_n(
          logging.WARNING,
          'Converting a UNIFORM_DISCRETE scaled discrete parameter '
          'to DOUBLE: %s', 10, self)
      scale_type = None

    default_value = self.default_value
    if default_value is not None:
      default_value = float(default_value)
    return ParameterConfig.factory(
        self.name,
        bounds=(float(self.bounds[0]), float(self.bounds[1])),
        scale_type=scale_type,
        default_value=default_value)

  @classmethod
  def merge(cls, one: 'ParameterConfig',
            other: 'ParameterConfig') -> 'ParameterConfig':
    """Merge two ParameterConfigs.

    Args:
      one: ParameterConfig with no child parameters.
      other: Must have the same type as one, and may not have child parameters.

    Returns:
      For Categorical, Discrete or Integer ParameterConfigs, the resulting
      config will be the union of all feasible values.
      For Double ParameterConfigs, the resulting config will have [min_value,
      max_value] set to the smallest and largest bounds.

    Raises:
      ValueError: If any of the input configs has child parameters, or if
        the two parameters have different types.
    """
    if one.child_parameter_configs or other.child_parameter_configs:
      raise ValueError(
          'Cannot merge parameters with child_parameter_configs: %s and %s' %
          one, other)
    if one.type != other.type:
      raise ValueError('Type conflicts between {} and {}'.format(
          one.type.name, other.type.name))
    if one.scale_type != other.scale_type:
      logging.warning('Scale type conflicts while merging %s and %s', one,
                      other)

    if one.type in (ParameterType.CATEGORICAL, ParameterType.DISCRETE):
      new_feasible_values = list(
          set(one.feasible_values + other.feasible_values))
      return ParameterConfig.factory(
          name=one.name,
          feasible_values=new_feasible_values,
          scale_type=one.scale_type)
    elif one.type in (ParameterType.INTEGER, ParameterType.DOUBLE):
      original_min, original_max = one.bounds
      other_min, other_max = other.bounds
      new_bounds = (min(original_min, other_min), max(original_max, other_max))
      return ParameterConfig.factory(
          name=one.name, bounds=new_bounds, scale_type=one.scale_type)
    raise ValueError('Unknown type {}. This is currently'
                     'an unreachable code.'.format(one.type))

  def traverse(
      self,
      show_children: bool = False) -> Generator['ParameterConfig', None, None]:
    """DFS Generator for parameter configs.

    Args:
      show_children: If True, every generated ParameterConfig has
        child_parameter_configs. For example, if 'foo' has two child configs
        'bar1' and 'bar2', then traversing 'foo' with show_children=True
        generates (foo, with bar1,bar2 as children), (bar1), and (bar2). If
        show_children=False, it generates (foo, without children), (bar1), and
        (bar2).

    Yields:
      DFS on all parameter configs.
    """
    if show_children:
      yield self
    else:
      yield self.clone_without_children
    for child in self.child_parameter_configs:
      yield from child.traverse(show_children)

  def contains(
      self, value: Union[trial.ParameterValueTypes,
                         trial.ParameterValue]) -> bool:
    """Check if the `value` is a valid value for this parameter config."""
    if not isinstance(value, trial.ParameterValue):
      value = trial.ParameterValue(value)

    if self.type == ParameterType.DOUBLE:
      return self.bounds[0] <= value.as_float and value.as_float <= self.bounds[
          1]
    elif self.type == ParameterType.INTEGER:
      if value.as_int != value.as_float:
        return False
      return self.bounds[0] <= value.as_int and value.as_int <= self.bounds[1]
    elif self.type == ParameterType.DISCRETE:
      return value.as_float in self.feasible_values
    elif self.type == ParameterType.CATEGORICAL:
      return value.as_str in self.feasible_values
    else:
      raise NotImplementedError(f'Cannot determine whether {value} is feasible'
                                f'for Unknown parameter type {self.type}.\n'
                                f'Full config: {repr(self)}')

  @property
  def num_feasible_values(self) -> Union[float, int]:
    if self.type == ParameterType.DOUBLE:
      return float('inf')
    elif self.type == ParameterType.INTEGER:
      return self.bounds[1] - self.bounds[0] + 1
    else:
      return len(self.feasible_values)


@attr.s(frozen=True, init=True, slots=True, kw_only=True)
class _PathSegment:
  """Selection of a parameter name and one of its values."""
  # A ParameterConfig name.
  name: str = attr.ib(
      init=True, validator=attr.validators.instance_of(str), kw_only=True)

  # A ParameterConfig value.
  value: Union[int, float, str] = attr.ib(
      init=True,
      validator=attr.validators.instance_of((int, float, str)),
      kw_only=True)


class _PathSelector(Sequence[_PathSegment]):
  """Immutable sequence of path segments."""

  def __init__(self, iterable: Iterable[_PathSegment] = tuple()):
    self._paths = tuple(iterable)

  @overload
  def __getitem__(self, s: slice) -> '_PathSelector':
    ...

  @overload
  def __getitem__(self, i: int) -> _PathSegment:
    ...

  def __getitem__(self, index):
    item = self._paths[index]
    if isinstance(item, _PathSegment):
      return item
    else:
      return _PathSelector(item)

  def __len__(self) -> int:
    """Returns the number of elements in the container."""
    return len(self._paths)

  def __add__(
      self, other: Union[Sequence[_PathSegment],
                         _PathSegment]) -> '_PathSelector':
    if isinstance(other, _PathSegment):
      other = [other]
    return _PathSelector(self._paths + tuple(other))

  def __repr__(self) -> str:
    """Returns the path as a string."""
    path = '/'.join(['{}={}'.format(p.name, p.value) for p in self._paths])
    return f'{path}'


class InvalidParameterError(Exception):
  """Error thrown when parameter values are invalid."""


################### Main Classes ###################
@attr.s(frozen=True, init=True, slots=True, kw_only=True)
class SearchSpaceSelector:
  """A Selector for all, or part of a SearchSpace."""

  # List of ParameterConfig objects referenced by this selector.
  # This is a reference to a list of objects owned by SearchSpace (and will
  # typically include the entire SearchSpace).
  _configs: List[ParameterConfig] = attr.ib(
      init=True,
      factory=list,
      # Verify that this is a list of ParameterConfig objects.
      validator=attr.validators.deep_iterable(
          member_validator=attr.validators.instance_of(ParameterConfig),
          iterable_validator=attr.validators.instance_of(list)),
      kw_only=True)

  # _selected_path and _selected_name control how parameters are added to the
  # search space.
  #
  # 1) If _selected_path is empty, and _selected_name is empty, parameters
  #    are added to the root of the search space.
  # 2) If _selected_path is empty, and _selected_name is non-empty, parameters
  #    will be added as child parameters to all root and child parameters
  #    with name ==_selected_name.
  # 3) If both _selected_path and _selected_name are specified, parameters will
  #    be added as child parameters to the parameter specified by the path and
  #    the name.
  # 4) If _selected_path is non-empty, and _selected_name is empty, this is an
  #    error.

  # An ordered list of _PathSelector objects which uniquely identifies a path
  # in a conditional tree.
  _selected_path: _PathSelector = attr.ib(
      init=True,
      default=_PathSelector(),  # We don't need factory since it's immutable.
      converter=_PathSelector,
      kw_only=True)

  # A ParameterConfig name.
  # If there is a _selected_name, then there have to also be _selected_values
  # below, and new parameters are added to the parent(s) selected by
  # _selected_path and _selected_name.
  _selected_name: str = attr.ib(
      init=True,
      default='',
      validator=attr.validators.instance_of(str),
      kw_only=True)

  # List of ParameterConfig values from _configs.
  # If there are _selected_values, then there have to also be _selected_name
  # above.
  _selected_values: MonotypeParameterSequence = attr.ib(
      init=True,
      factory=list,
      validator=attr.validators.deep_iterable(
          member_validator=attr.validators.instance_of((int, float, str)),
          iterable_validator=attr.validators.instance_of(list)),
      kw_only=True)

  @property
  def parameter_name(self) -> str:
    """Returns the selected parameter name."""
    return self._selected_name

  @property
  def parameter_values(self) -> MonotypeParameterSequence:
    """Returns the selected parameter values.

    Callers should not mutate the returned values.

    Returns:
      Values for the parameter selected by this selector.
    """
    return self._selected_values

  def add_float_param(self,
                      name: str,
                      min_value: float,
                      max_value: float,
                      *,
                      default_value: Optional[float] = None,
                      scale_type: Optional[ScaleType] = ScaleType.LINEAR,
                      index: Optional[int] = None) -> 'SearchSpaceSelector':
    """Adds floating point parameter config(s) to the search space.

    If select_all() was previously called for this selector, so it contains
    selected parent values, the parameter configs will be added as child
    parameters to the selected parameter configs, and a reference to this
    selector is returned.

    If no parent values are selected, the parameter config(s) will be added at
    the same level as currently selected parameters, and a reference to the
    newly added parameters is returned.

    Args:
      name: The parameter's name. Cannot be empty.
      min_value: Inclusive lower bound for the parameter.
      max_value: Inclusive upper bound for the parameter.
      default_value: A default value for the Parameter.
      scale_type: Scaling to be applied. NOT VALIDATED.
      index: Specifies the multi-dimensional index for this parameter. E.g. if
        name='rate' and index=0, then a single ParameterConfig with name
        'rate[0]' is added. `index` should be >= 0.

    Returns:
      SearchSpaceSelector(s) for the newly added parameter(s):
      One SearchSpaceSelector if one parameter was added, or a list of
      SearchSpaceSelector if multiple parameters were added.

    Raises:
      ValueError: If `index` is invalid (e.g. negative).
    """
    bounds = (float(min_value), float(max_value))
    param_names = self._get_parameter_names_to_create(name=name, index=index)

    new_params = []
    for param_name in param_names:
      new_pc = ParameterConfig.factory(
          name=param_name,
          bounds=bounds,
          scale_type=scale_type,
          default_value=default_value)
      new_params.append(new_pc)
    return self._add_parameters(new_params)[0]

  def add_int_param(
      self,
      name: str,
      min_value: int,
      max_value: int,
      *,
      default_value: Optional[int] = None,
      scale_type: Optional[ScaleType] = None,
      index: Optional[int] = None,
      experimental_fidelity_config: Optional[FidelityConfig] = None,
  ) -> 'SearchSpaceSelector':
    """Adds integer parameter config(s) to the search space.

    If select_all() was previously called for this selector, so it contains
    selected parent values, the parameter configs will be added as child
    parameters to the selected parameter configs, and a reference to this
    selector is returned.

    If no parent values are selected, the parameter config(s) will be added at
    the same level as currently selected parameters, and a reference to the
    newly added parameters is returned.

    Args:
      name: The parameter's name. Cannot be empty.
      min_value: Inclusive lower bound for the parameter.
      max_value: Inclusive upper bound for the parameter.
      default_value: A default value for the Parameter.
      scale_type: Scaling to be applied. NOT VALIDATED.
      index: Specifies the multi-dimensional index for this parameter. E.g. if
        name='hidden_units' and index=0, then a single ParameterConfig with name
        'hidden_units[0]' is added. `index` should be >= 0.
      experimental_fidelity_config: EXPERIMENTAL. See FidelityConfig doc.

    Returns:
      SearchSpaceSelector for the newly added parameter.

    Raises:
      ValueError: If min_value or max_value are not integers.
      ValueError: If `index` is invalid (e.g. negative).
    """
    int_min_value = int(min_value)
    if not math.isclose(min_value, int_min_value):
      raise ValueError('min_value for an INTEGER parameter should be an integer'
                       ', got: [{}]'.format(min_value))
    int_max_value = int(max_value)
    if not math.isclose(max_value, int_max_value):
      raise ValueError('max_value for an INTEGER parameter should be an integer'
                       ', got: [{}]'.format(min_value))
    bounds = (int_min_value, int_max_value)

    param_names = self._get_parameter_names_to_create(name=name, index=index)

    new_params = []
    for param_name in param_names:
      new_pc = ParameterConfig.factory(
          name=param_name,
          bounds=bounds,
          scale_type=scale_type,
          fidelity_config=experimental_fidelity_config,
          default_value=default_value)
      new_params.append(new_pc)
    return self._add_parameters(new_params)[0]

  def add_discrete_param(
      self,
      name: str,
      feasible_values: Union[Sequence[float], Sequence[int]],
      *,
      default_value: Optional[Union[float, int]] = None,
      scale_type: Optional[ScaleType] = ScaleType.LINEAR,
      index: Optional[int] = None,
      auto_cast: Optional[bool] = True,
      experimental_fidelity_config: Optional[FidelityConfig] = None,
  ) -> 'SearchSpaceSelector':
    """Adds ordered numeric parameter config(s) with a finite set of values.

    IMPORTANT: If a parameter is discrete, its values are assumed to have
    ordered semantics. Thus, you should not use discrete parameters for
    unordered values such as ids. In this case, see add_categorical_param()
    below.

    If select_all() was previously called for this selector, so it contains
    selected parent values, the parameter configs will be added as child
    parameters to the selected parameter configs, and a reference to this
    selector is returned.

    If no parent values are selected, the parameter config(s) will be added at
    the same level as currently selected parameters, and a reference to the
    newly added parameters is returned.

    Args:
      name: The parameter's name. Cannot be empty.
      feasible_values: The set of feasible values for this parameter.
      default_value: A default value for the Parameter.
      scale_type: Scaling to be applied. NOT VALIDATED.
      index: Specifies the multi-dimensional index for this parameter. E.g. if
        name='batch_size' and index=0, then a single ParameterConfig with name
        'batch_size[0]' is added. `index` should be >= 0.
      auto_cast: If False, the external type will be set to INTEGER if all
        values are castable to an integer without losing precision. If True, the
        external type will be set to float.
      experimental_fidelity_config: EXPERIMENTAL. See FidelityConfig doc.

    Returns:
      SearchSpaceSelector for the newly added parameter.

    Raises:
      ValueError: If `index` is invalid (e.g. negative).
    """
    param_names = self._get_parameter_names_to_create(name=name, index=index)

    external_type = ExternalType.FLOAT
    if auto_cast:
      # If all feasible values are convertible to ints without loss of
      # precision, annotate the external type as INTEGER. This will cast
      # [0., 1., 2.] into [0, 1, 2] when parameter values are returned in
      # clients.
      if all([v == round(v) for v in feasible_values]):
        external_type = ExternalType.INTEGER

    new_params = []
    for param_name in param_names:
      new_pc = ParameterConfig.factory(
          name=param_name,
          feasible_values=sorted(feasible_values),
          scale_type=scale_type,
          fidelity_config=experimental_fidelity_config,
          default_value=default_value,
          external_type=external_type)
      new_params.append(new_pc)
    return self._add_parameters(new_params)[0]

  def add_categorical_param(
      self,
      name: str,
      feasible_values: Sequence[str],
      *,
      default_value: Optional[str] = None,
      scale_type: Optional[ScaleType] = None,
      index: Optional[int] = None) -> 'SearchSpaceSelector':
    """Adds unordered string-valued parameter config(s) to the search space.

    IMPORTANT: If a parameter is categorical, its values are assumed to be
    unordered. If the `feasible_values` have ordering, use add_discrete_param()
    above, since it will improve Vizier's model quality.

    If select_all() was previously called for this selector, so it contains
    selected parent values, the parameter configs will be added as child
    parameters to the selected parameter configs, and a reference to this
    selector is returned.

    If no parent values are selected, the parameter config(s) will be added at
    the same level as currently selected parameters, and a reference to the
    newly added parameters is returned.

    Args:
      name: The parameter's name. Cannot be empty.
      feasible_values: The set of feasible values for this parameter.
      default_value: A default value for the Parameter.
      scale_type: Scaling to be applied. NOT VALIDATED.
      index: Specifies the multi-dimensional index for this parameter. E.g. if
        name='id' and index=0, then a single ParameterConfig with name 'id[0]'
        is added. `index` should be >= 0.

    Returns:
      SearchSpaceSelector for the newly added parameter.

    Raises:
      ValueError: If `index` is invalid (e.g. negative).
    """
    param_names = self._get_parameter_names_to_create(name=name, index=index)

    new_params = []
    for param_name in param_names:
      new_pc = ParameterConfig.factory(
          name=param_name,
          feasible_values=sorted(feasible_values),
          scale_type=scale_type,
          default_value=default_value)
      new_params.append(new_pc)
    return self._add_parameters(new_params)[0]

  def add_bool_param(self,
                     name: str,
                     feasible_values: Optional[Sequence[bool]] = None,
                     *,
                     default_value: Optional[bool] = None,
                     scale_type: Optional[ScaleType] = None,
                     index: Optional[int] = None) -> 'SearchSpaceSelector':
    """Adds boolean-valued parameter config(s) to the search space.

    If select_all() was previously called for this selector, so it contains
    selected parent values, the parameter configs will be added as child
    parameters to the selected parameter configs, and a reference to this
    selector is returned.

    If no parent values are selected, the parameter config(s) will be added at
    the same level as currently selected parameters, and a reference to the
    newly added parameters is returned.

    Args:
      name: The parameter's name. Cannot be empty.
      feasible_values: An optional list of feasible boolean values, i.e. one of
        the following: [True], [False], [True, False], [False, True].
      default_value: A default value for the Parameter.
      scale_type: Scaling to be applied. NOT VALIDATED.
      index: Specifies the multi-dimensional index for this parameter. E.g. if
        name='match' and index=0, then a single ParameterConfig with name
        'match[0]' is added. `index` should be >= 0.

    Returns:
      SearchSpaceSelector for the newly added parameter.

    Raises:
      ValueError: If `feasible_values` has invalid values.
      ValueError: If `index` is invalid (e.g. negative).
    """
    allowed_values = (None, (True, False), (False, True), (True,), (False,))
    if feasible_values is not None and tuple(
        feasible_values) not in allowed_values:
      raise ValueError('feasible_values must be one of %s; got: %s.' %
                       (allowed_values, feasible_values))
    # Boolean parameters are represented as categorical parameters internally.
    bool_to_string = lambda x: 'True' if x else 'False'
    if feasible_values is None:
      categories = ('True', 'False')
    else:
      categories = [bool_to_string(x) for x in feasible_values]
    feasible_values = sorted(categories, reverse=True)

    if default_value is not None:
      default_value = bool_to_string(default_value)

    param_names = self._get_parameter_names_to_create(name=name, index=index)

    new_params = []
    for param_name in param_names:
      new_pc = ParameterConfig.factory(
          name=param_name,
          feasible_values=sorted(feasible_values),
          scale_type=scale_type,
          default_value=default_value,
          external_type=ExternalType.BOOLEAN)
      new_params.append(new_pc)
    return self._add_parameters(new_params)[0]

  def select(
      self,
      parameter_name: str,
      parameter_values: Optional[MonotypeParameterSequence] = None
  ) -> 'SearchSpaceSelector':
    """Selects a single parameter specified by path and parameter_name.

    This method should be called to select a parent parameter, before calling
    `add_*_param` methods to create child parameters.

    Given a selector to the root of the search space:
    root = pyvizier.SearchSpace().root

    1) To select a parameter at the root of the search space, with parent values
      for child parameters:
      model = root.select('model_type', ['dnn'])
      model.add_float_param('hidden_units', ...)
    2) To select a parameter at the root of the search space, and defer parent
      value selection to later calls:
      model = root.select('model_type')
      # Add `hidden_units` and `optimizer_type` as `dnn` children.
      model.select_values(['dnn']).add_float_param('hidden_units', ...)
      model.select_values(['dnn']).add_categorical_param(
        'optimizer_type', ['adam', 'adagrad'])
      # Add `optimizer_type` and `activation` as `linear` children.
      model.select_values(['linear']).add_categorical_param(
        'optimizer_type', ['adam', 'ftrl'])
      model.select_values(['linear']).add_categorical_param('activation', ...)
    3) To select a parameter in a conditional search space, specify a path, by
      chaining select() calls:
      optimizer = root.select('model_type', ['linear']).select('optimizer_type')
      optimizer.select_values('adam').add_float_param('learning_rate', 0.001,..)
      optimizer.select_values('ftrl').add_float_param('learning_rate', 0.1,..)

      # OR pre-select the parent parameter values:
      optimizer = root.select('model_type', ['linear']).select(
        'optimizer_type', ['adam'])
      optimizer.add_float_param('learning_rate', 0.001,...)
    4) If there is *only one* parameter with the given name, then it is possible
      to select it without specifying the path, using:
      selectors = root.select_all('activation')
      # 'activation' exists only under model_type='linear'.
      assert len(selectors) == 1
      activation = selectors[0]

    Args:
      parameter_name:
      parameter_values: Optional parameter values for this selector, which will
        be used to add child parameters, or traverse a conditional tree.

    Returns:
      A new SearchSpaceSelector.
    """
    # Make sure parameter_name exists in the conditional parameters tree.
    # parameter_values will be validated only when a child parameter is added.
    if not self._parameter_exists(parameter_name):
      raise ValueError('No parameter with name {} exists in this SearchSpace')

    path = []
    selected_values = []
    if parameter_values is not None:
      if not isinstance(parameter_values, (list, tuple)):
        raise ValueError('parameter_values should be a list or tuple, given '
                         '{} with type {}'.format(parameter_values,
                                                  type(parameter_values)))
      selected_values = parameter_values

    if self._selected_name:
      # There is already a parameter name selected, so this is a chained select
      # call.
      if not self._selected_values:
        raise ValueError('Cannot call select() again before parameter values '
                         'are selected: parameter {} was previously selected, '
                         ' with the path: {}, but no values were selected for '
                         'it'.format(self.parameter_name, self.path_string))
      # Return a new selector, with the currently selected parameter added to
      # the path.
      new_path_segment = [
          _PathSegment(
              name=self._selected_name, value=self._selected_values[0])
      ]
      path = self._selected_path + new_path_segment
      if not self._path_exists(path):
        raise ValueError('Path {} does not exist in this SearchSpace: '
                         '{}'.format((path), self))

    return SearchSpaceSelector(
        configs=self._configs,
        selected_path=path,
        selected_name=parameter_name,
        selected_values=selected_values)

  def select_values(
      self,
      parameter_values: MonotypeParameterSequence) -> 'SearchSpaceSelector':
    """Selects values for a pre-selected parameter.

    This method should be called to select parent parameter(s) value(s), before
    calling `add_*_param` methods to create child parameters.

    This method must be called AFTER select().
    This method mutates this selector.

    Args:
      parameter_values: Parameter values for this selector, which will be used
        to add child parameters.

    Returns:
      SearchSpaceSelector
    """
    if not self._selected_name:
      raise ValueError('No parameter is selected. Call select() first.')
    if not parameter_values:
      raise ValueError(
          'parameter_values cannot be empty. Specify at least one value.')
    if not isinstance(parameter_values, (list, tuple)):
      raise ValueError('parameter_values should be a list or tuple, given '
                       '{} with type {}'.format(parameter_values,
                                                type(parameter_values)))
    # TODO: Allow to directly select boolean parent parameters.
    object.__setattr__(self, '_selected_values', parameter_values)
    return self

  def select_all(
      self, parameter_name: str, parameter_values: MonotypeParameterSequence
  ) -> List['SearchSpaceSelector']:
    """Select one or more parent parameters, with the same name.

    This method should be called to select parent parameter(s), before calling
    `add_*_param` methods to create child parameters.
    Multiple parent parameters with the same name are possible in a conditional
    search space. See go/conditional-parameters for more details.

    1) If the conditional search space has two parameters with the same
    name, 'optimizer_type', given a selector to the root of the search space,
    select_all() can be used to simultaneously add child parameters to both
    'optimizer_type` parameters:

    root = pyvizier.SearchSpace().root
    model.select_values(['dnn']).add_categorical_param(
        'optimizer_type', ['adam', 'adagrad'])
    model.select_values(['linear']).add_categorical_param(
        'optimizer_type', ['adam', 'ftrl'])
    # Add a 'learning_rate' parameter to both 'adam' optimizers:
    optimizers = model.select_all('optimizer_type', parent_values=['adam'])
    optimizers.add_float_param('learning_rate', ...)

    2) If there is *only one* parameter with the given name, then it is also
      possible to use select_all() to select it:
      root = pyvizier.SearchSpace().root
      model.select_values(['dnn']).add_categorical_param('activation', ...)
      # Select the single parameter with the name 'activation':
      selectors = root.select_all('activation')
      assert len(selectors) == 1
      activation = selector[0]

    Args:
      parameter_name:
      parameter_values: Optional parameter values for this selector, which will
        be used to add child parameters.

    Returns:
      List of SearchSpaceSelector
    """
    # TODO: Raise an error if this selector already has selected_name.
    # Make sure parameter_name exists in the conditional parameters tree.
    if not self._parameter_exists(parameter_name):
      raise ValueError('No parameter with name {} exists in this SearchSpace')

    if parameter_values is not None:
      if not isinstance(parameter_values, (list, tuple)):
        raise ValueError('parameter_values should be a list or tuple, given '
                         '{} with type {}'.format(parameter_values,
                                                  type(parameter_values)))
    # TODO: Complete this method.
    raise NotImplementedError()

  def _path_exists(self, path: _PathSelector) -> bool:
    """Checks if the path exists in the conditional tree."""
    for parent in self._configs:
      if (path[0].name == parent.name and
          path[0].value in parent.feasible_values):
        if len(path) == 1:
          # No need to recurse.
          return True
        return self._path_exists_inner(parent, path[1:])
    return False

  @classmethod
  def _path_exists_inner(cls, current_root: ParameterConfig,
                         current_path: _PathSelector) -> bool:
    """Returns true if the path exists, starting at root_parameter."""
    child_idx = None
    for idx, child in enumerate(current_root.child_parameter_configs):
      if (current_path[0].name == child.name and
          current_path[0].value in child.feasible_values):
        child_idx = idx
        break
    if child_idx is None:
      # No match is found. This path does not exist.
      return False
    if len(current_path) == 1:
      # This is the end of the path.
      return True
    # Keep traversing.
    return cls._path_exists_inner(
        current_root.child_parameter_configs[child_idx], current_path[1:])

  def _parameter_exists(self, parameter_name: str) -> bool:
    """Checks if there exists at least one parameter with this name.

    Note that this method checks existence in the entire search space.

    Args:
      parameter_name:

    Returns:
      bool: Exists.
    """
    found = False
    for parent in self._configs:
      for pc in parent.traverse(show_children=False):
        if pc.name == parameter_name:
          found = True
          break
    return found

  @classmethod
  def _get_parameter_names_to_create(cls,
                                     *,
                                     name: str,
                                     length: Optional[int] = None,
                                     index: Optional[int] = None) -> List[str]:
    """Returns the names of all parameters which should be created.

    Args:
      name: The base parameter name.
      length: Specifies the length of a multi-dimensional parameters. If larger
        than 1, then multiple ParameterConfigs are added. E.g. if name='rate'
        and length=2, then two ParameterConfigs with names 'rate[0]', 'rate[1]'
        are added. Cannot be specified together with `index`.
      index: Specifies the multi-dimensional index for this parameter. Cannot be
        specified together with `length`. E.g. if name='rate' and index=1, then
        a single ParameterConfig with name 'rate[1]' is added.

    Returns:
      List of parameter names to create.

    Raises:
      ValueError: If `length` or `index` are invalid.
    """
    if length is not None and index is not None:
      raise ValueError('Only one of `length` and `index` can be specified. Got'
                       ' length={}, index={}'.format(length, index))
    if length is not None and length < 1:
      raise ValueError('length must be >= 1. Got length={}'.format(length))
    if index is not None and index < 0:
      raise ValueError('index must be >= 0. Got index={}'.format(index))

    param_names = []
    if length is None and index is None:
      # Add one parameter with no multi-dimensional index.
      param_names.append(name)
    elif index is not None:
      # Add one parameter with a multi-dimensional index.
      param_names.append(cls._multi_dimensional_parameter_name(name, index))
    elif length is not None:
      # `length > 0' is synthatic sugar for multi multi-dimensional parameter.
      # Each multi-dimensional parameter is encoded as a list of separate
      # parameters with names equal to `name[index]` (index is zero based).
      for i in range(length):
        param_names.append(cls._multi_dimensional_parameter_name(name, i))
    return param_names

  @classmethod
  def _multi_dimensional_parameter_name(cls, name: str, index: int) -> str:
    """Returns the indexed parameter name."""
    return '{}[{}]'.format(name, index)

  @classmethod
  def parse_multi_dimensional_parameter_name(
      cls, name: str) -> Optional[Tuple[str, int]]:
    """Returns the base name for a multi-dimensional parameter name.

    Args:
      name: A parameter name.

    Returns:
      (base_name, index): if name='hidden_units[10]', base_name='hidden_units'
        and index=10.
      Returns None if name is not in the format 'base_name[idx]'.
    """
    regex = r'(?P<name>[^()]*)\[(?P<index>\d+)\]$'
    pattern = re.compile(regex)
    matches = pattern.match(name)
    if matches is None:
      return None
    return (matches.groupdict()['name'], int(matches.groupdict()['index']))

  @property
  def path_string(self) -> str:
    """Returns the selected path as a string."""
    return str(self._selected_path)

  def _add_parameters(
      self, parameters: List[ParameterConfig]) -> List['SearchSpaceSelector']:
    """Adds ParameterConfigs either to the root, or as child parameters.

    Args:
      parameters: The parameters to add to the search space.

    Returns:
      A list of SearchSpaceSelectors, one for each parameters added.
    """
    if self._selected_name and not self._selected_values:
      raise ValueError(
          'Cannot add child parameters to parameter {}: parent values were '
          'not selected. Call select_values() first.'.format(
              self._selected_name))
    if not self._selected_name and self._selected_values:
      raise ValueError(
          'Cannot add child parameters: no parent name is selected.'
          ' Call select() or select_all() first.')
    if self._selected_path and not self._selected_name:
      raise ValueError(
          'Cannot add child parameters: path is specified ({}), but no parent'
          ' name is specified. Call select() or select_all() first'.format(
              self.path_string))

    selectors: List['SearchSpaceSelector'] = []
    if not self._selected_path and not self._selected_name:
      # If _selected_path is empty, and _selected_name is empty, parameters
      # are added to the root of the search space.
      logging.info('Adding parameters to root of search space: %s', parameters)
      self._configs.extend(parameters)
      # Return Selectors for the newly added parameters.
      for param in parameters:
        selectors.append(
            SearchSpaceSelector(
                configs=self._configs,
                selected_path=[],
                selected_name=param.name,
                selected_values=[]))
    elif not self._selected_path and self._selected_name:
      # If _selected_path is empty, and _selected_name is not empty, parameters
      # will be added as child parameters to *all* root and child parameters
      # with name ==_selected_name.
      logging.info(
          'Adding child parameters to all matching parents with '
          'name "%s": %s', self._selected_name, parameters)
      found_at_least_one_match = False
      for idx, root_param in enumerate(self._configs):
        updated_param, new_selectors = self._recursive_add_child_parameters(
            self._configs, _PathSelector(), root_param, self._selected_name,
            self._selected_values, parameters)
        if new_selectors:
          found_at_least_one_match = True
        # Update the root ParameterConfig in place.
        self._configs[idx] = updated_param
        selectors.extend(new_selectors)
      if not found_at_least_one_match:
        logging.warning('Failed to find a matching parent with name %s',
                        self._selected_name)
    else:
      # If both _selected_path and _selected_name are specified, parameters will
      # be added as child parameters to the parameter specified by the path and
      # the name.
      logging.info(
          'Adding child parameters to parent with path "%s" '
          ', name "%s", values: %s: %s', (self._selected_path),
          self._selected_name, self._selected_values, parameters)
      idx, updated_param, new_selectors = self._add_parameters_at_selected_path(
          root_configs=self._configs,
          complete_path=self._selected_path,
          parent_name=self._selected_name,
          parent_values=self._selected_values,
          new_children=parameters)
      # Update the root ParameterConfig in place.
      self._configs[idx] = updated_param
      selectors.extend(new_selectors)

    if not selectors:
      raise ValueError(
          'Cannot add child parameters: the path ({}), is not valid.'.format(
              self.path_string))
    return selectors

  @classmethod
  def _recursive_add_child_parameters(
      cls, configs: List[ParameterConfig], path: _PathSelector,
      root: ParameterConfig, parent_name: str,
      parent_values: MonotypeParameterSequence,
      new_children: List[ParameterConfig]
  ) -> Tuple[ParameterConfig, List['SearchSpaceSelector']]:
    """Recursively adds new children to all matching parameters.

    new_children are potentially added to root, and all matching child
    parameters with name==parent_name.

    Args:
      configs: A list of configs to include in returned SearchSpaceSelectors,
        this list is not modified or used for anything else.
      path: The path to include in returned SearchSpaceSelectors.
      root: Parent parameter to start the recursion at.
      parent_name: new_children are added to all parameter with this name.
      parent_values: new_children are added with these parent values.
      new_children: Child parameter configs to add.

    Returns:
      (An updated root with all of its children updated, list of selectors to
       any parameters which may have been added)
    """
    updated_children: List[Tuple[MonotypeParameterSequence,
                                 ParameterConfig]] = []
    selectors: List['SearchSpaceSelector'] = []

    logging.debug(
        '_recursive_add_child_parameters called with path=%s, '
        'root=%s, parent_name=%s, parent_values=%s', path, root, parent_name,
        parent_values)

    if root.name == parent_name:
      # Add new children to this root. If this is a leaf parameter,
      # e.g. it has no children, this is where the recursion ends.
      for child in new_children:
        updated_children.append((parent_values, child))
        # For the path, select one parent value, since for the path, the exact
        # value does not matter, as long as it's valid.
        root_path_fragment = [
            _PathSegment(name=root.name, value=parent_values[0])
        ]
        selectors.append(
            SearchSpaceSelector(
                configs=configs,
                selected_path=path + root_path_fragment,
                selected_name=child.name,
                selected_values=[]))
    # Recursively update existing children, if any.
    for child in root.child_parameter_configs:
      # For the path, select one parent value, since for the path, the exact
      # value does not matter, as long as it's valid.
      root_path_fragment = [
          _PathSegment(name=root.name, value=child.matching_parent_values[0])
      ]
      updated_child, new_selectors = cls._recursive_add_child_parameters(
          configs, path + root_path_fragment, child, parent_name, parent_values,
          new_children)
      updated_children.append(
          (updated_child.matching_parent_values, updated_child))
      selectors += new_selectors
    # Update all children (existing and potentially new) in the root.
    return root.clone_without_children.add_children(updated_children), selectors

  @classmethod
  def _add_parameters_at_selected_path(
      cls, root_configs: List[ParameterConfig], complete_path: _PathSelector,
      parent_name: str, parent_values: MonotypeParameterSequence,
      new_children: List[ParameterConfig]
  ) -> Tuple[int, ParameterConfig, List['SearchSpaceSelector']]:
    """Adds new children to the parameter specified by the path and parent_name.

    Args:
      root_configs: A list of configs to include in returned
        SearchSpaceSelectors, this list is not modified. These are expected to
        be the configs at the root of the search space.
      complete_path: The path to include in the returned SearchSpaceSelectors.
      parent_name: new_children are added to all parameter with this name.
      parent_values: new_children are added with these parent values.
      new_children: Child parameter configs to add.

    Returns:
      (Root index in root_configs,
       an updated root with all of its children updated,
       list of selectors to any parameters which may have been added)

    Raises:
      RuntimeError:
      ValueError:
    """
    if not complete_path:
      # This is an internal error, since the caller should never specify an
      # empty current_path.
      raise RuntimeError('Internal error: got empty complete_path')

    # This is the beginning of the recursion. Select a root to recurse at.
    current_root: Optional[ParameterConfig] = None
    root_idx: int = 0
    for root_idx, root_param in enumerate(root_configs):
      if complete_path[0].name == root_param.name:
        current_root = root_param
        break
    if current_root is None:
      raise ValueError('Invalid path: {}: failed to traverse the path: failed'
                       ' to find a matching root for parameter name "{}".'
                       ' Root parameter names: {}'.format(
                           (complete_path), complete_path[0].name,
                           [pc.name for pc in root_configs]))

    updated_root, selectors = cls._add_parameters_at_selected_path_inner(
        root_configs=root_configs,
        complete_path=complete_path,
        current_root=current_root,
        current_path=complete_path[1:],
        parent_name=parent_name,
        parent_values=parent_values,
        new_children=new_children)
    return (root_idx, updated_root, selectors)

  @classmethod
  def _add_parameters_at_selected_path_inner(
      cls, root_configs: List[ParameterConfig], complete_path: _PathSelector,
      current_root: ParameterConfig, current_path: _PathSelector,
      parent_name: str, parent_values: MonotypeParameterSequence,
      new_children: List[ParameterConfig]
  ) -> Tuple[ParameterConfig, List['SearchSpaceSelector']]:
    """Adds new children to the parameter specified by the path and parent_name.

    Args:
      root_configs: A list of configs to include in returned
        SearchSpaceSelectors, this list is not modified. These are expected to
        be the configs at the root of the search space.
      complete_path: The path to include in the returned SearchSpaceSelectors.
      current_root: Parent parameter to start the recursion at.
      current_path: The path to the parent parameter from current_root. This is
        used in the recursion.
      parent_name: new_children are added to all parameter with this name.
      parent_values: new_children are added with these parent values.
      new_children: Child parameter configs to add.

    Returns:
      (An updated root with all of its children updated,
       List of selectors to all added parameters)

    Raises:
      RuntimeError:
      ValueError:
    """
    updated_children: List[Tuple[MonotypeParameterSequence,
                                 ParameterConfig]] = []
    selectors: List['SearchSpaceSelector'] = []

    if not current_path:
      # This is the end of the path. End the recursion.
      # parent_name should be a child of current_root
      child_idx = None
      for idx, child in enumerate(current_root.child_parameter_configs):
        if parent_name == child.name:
          child_idx = idx
          last_parent_path = [
              _PathSegment(name=parent_name, value=parent_values[0])
          ]
          new_path = complete_path + last_parent_path
          updated_child, selectors = cls._add_child_parameters(
              root_configs, new_path, child, parent_values, new_children)
          break
      if child_idx is None:
        raise ValueError('Invalid parent_name: after traversing the path "{}", '
                         'failed to find a child parameter with name "{}".'
                         ' Current root="{}"'.format((complete_path),
                                                     parent_name, current_root))

      # Update current_root with the updated child.
      for idx, child in enumerate(current_root.child_parameter_configs):
        if idx == child_idx:
          updated_children.append(
              (updated_child.matching_parent_values, updated_child))
        else:
          updated_children.append((child.matching_parent_values, child))
      return (
          current_root.clone_without_children.add_children(updated_children),
          selectors)

    # Traverse the path: find which child matches the next path selection.
    child_idx = None
    for idx, child in enumerate(current_root.child_parameter_configs):
      if (current_path[0].name == child.name and
          current_path[0].value in child.feasible_values):
        child_idx = idx
        break
    if child_idx is None:
      raise ValueError('Invalid path: "{}": failed to traverse the path: failed'
                       ' to find a matching child for path selector "{}".'
                       ' Current root="{}", current_path="{}"'.format(
                           (complete_path), (current_path[:1]),
                           current_root.name, (current_path)))

    updated_child, selectors = cls._add_parameters_at_selected_path_inner(
        root_configs=root_configs,
        complete_path=complete_path,
        current_root=current_root.child_parameter_configs[child_idx],
        current_path=current_path[1:],
        parent_name=parent_name,
        parent_values=parent_values,
        new_children=new_children)
    # Update current_root with the updated child, leave the selectors untouched.
    for idx, child in enumerate(current_root.child_parameter_configs):
      if idx == child_idx:
        updated_children.append(
            (updated_child.matching_parent_values, updated_child))
      else:
        updated_children.append((child.matching_parent_values, child))
    return (current_root.clone_without_children.add_children(updated_children),
            selectors)

  @classmethod
  def _add_child_parameters(
      cls, selector_configs: List[ParameterConfig],
      selector_path: _PathSelector, parent: ParameterConfig,
      parent_values: MonotypeParameterSequence,
      new_children: List[ParameterConfig]
  ) -> Tuple[ParameterConfig, List['SearchSpaceSelector']]:
    """Adds new children to the parent parameter and returns selectors.

    Args:
      selector_configs: A list of configs to include in returned
        SearchSpaceSelectors, this list is not modified. These are expected to
        be the configs at the root of the search space.
      selector_path: The path to include in the returned SearchSpaceSelectors.
      parent: Parent parameter to add children to.
      parent_values: new_children are added with these parent values.
      new_children: Child parameter configs to add.

    Returns:
      (An updated root with all of its children updated,
       List of selectors to all added parameters)

    Raises:
      RuntimeError:
      ValueError:
    """
    updated_children: List[Tuple[MonotypeParameterSequence,
                                 ParameterConfig]] = []
    selectors: List['SearchSpaceSelector'] = []

    # Add existing children.
    for child in parent.child_parameter_configs:
      updated_children.append((child.matching_parent_values, child))
    # Add new child parameter configs.
    for child in new_children:
      updated_children.append((parent_values, child))
      selectors.append(
          SearchSpaceSelector(
              configs=selector_configs,
              selected_path=selector_path,
              selected_name=child.name,
              selected_values=[]))
    # Add all children (existing and potentially new) to the parent.
    return (parent.clone_without_children.add_children(updated_children),
            selectors)


@attr.s(frozen=True, init=True, slots=True, kw_only=True)
class SearchSpace:
  """A builder and wrapper for StudyConfig.parameter_configs."""

  _parameter_configs: List[ParameterConfig] = attr.ib(init=False, factory=list)

  @classmethod
  def _factory(
      cls,
      parameter_configs: Optional[List[ParameterConfig]] = None
  ) -> 'SearchSpace':
    """Creates a new SearchSpace containing the provided parameter configs.

    Args:
      parameter_configs:

    Returns:
      SearchSpace
    """
    if parameter_configs is None:
      parameter_configs = []
    space = cls()
    object.__setattr__(space, '_parameter_configs', list(parameter_configs))
    return space

  @property
  def parameters(self) -> List[ParameterConfig]:
    """Returns the parameter configs in this search space."""
    return self._parameter_configs

  def select_root(self) -> SearchSpaceSelector:
    # Deprecated function.
    # TODO: Remove this from downstream user code.
    return self.root

  @property
  def root(self) -> SearchSpaceSelector:
    """Returns a selector for the root of the search space.

    Parameters can be added to the search space using the returned
    SearchSpaceSelector.
    """
    return SearchSpaceSelector(configs=self._parameter_configs)

  @property
  def is_conditional(self) -> bool:
    """Returns True if search_space contains any conditional parameters."""
    return any([p.child_parameter_configs for p in self._parameter_configs])

  def contains(self, parameters: trial.ParameterDict) -> bool:
    try:
      self.assert_contains(parameters)
      return True
    except InvalidParameterError:
      return False

  def assert_contains(self, parameters: trial.ParameterDict) -> bool:
    """Throws an error if parameters is not a valid point in the space.

    Args:
      parameters:

    Returns:
      Always returns True unless an exception is Raised.

    Raises:
      InvalidParameterError: If parameters are invalid.
      NotImplementedError: If parameter type is unknown
    """
    if self.is_conditional:
      raise NotImplementedError('Not implemented for conditional space.')
    if len(parameters) != len(self._parameter_configs):
      set1 = set(pc.name for pc in self._parameter_configs)
      set2 = set(parameters)
      raise InvalidParameterError(
          f'Search space has {len(self._parameter_configs)} parameters '
          f'but only {len(parameters)} were given. '
          f'Missing in search space: {set2 - set1}. '
          f'Missing in parameters: {set1 - set2}.')
    for pc in self._parameter_configs:
      if pc.name not in parameters:
        raise InvalidParameterError(f'{pc.name} is missing in {parameters}.')
      elif not pc.contains(parameters[pc.name]):
        raise InvalidParameterError(
            f'{parameters[pc.name]} is not feasible in {pc}')
    return True
