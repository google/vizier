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

"""ParameterConfig wraps ParameterConfig and ParameterSpec protos."""

import collections
from typing import Iterable, Set as AbstractSet, Sized
import copy
import enum
import json
import math
import re
from typing import Generator, Iterator, List, Optional, Sequence, Tuple, Union, overload

from absl import logging
import attr
from vizier._src.pyvizier.shared import trial


ExternalType = trial.ExternalType
ParameterType = trial.ParameterType


class ScaleType(enum.Enum):
  """Valid Values for ParameterConfig.scale_type."""

  LINEAR = 'LINEAR'
  LOG = 'LOG'
  REVERSE_LOG = 'REVERSE_LOG'
  UNIFORM_DISCRETE = 'UNIFORM_DISCRETE'

  def is_nonlinear(self) -> bool:
    return self in [self.LOG, self.REVERSE_LOG]


# A sequence of possible internal parameter values.
ParameterValueTypes = trial.ParameterValueTypes
MonotypeParameterSequence = Union[Sequence[Union[int, float]], Sequence[str]]
MonotypeParameterList = Union[List[Union[int, float]], List[str]]


def _validate_bounds(bounds: Union[Tuple[int, int], Tuple[float, float]]):
  """Validates the bounds."""
  if len(bounds) != 2:
    raise ValueError(f'Bounds must have length 2. Given: {bounds}')
  lower = bounds[0]
  upper = bounds[1]
  if not all([math.isfinite(v) for v in (lower, upper)]):
    raise ValueError(
        f'Both "lower" and "upper" must be finite. Given: ({lower}, {upper})'
    )
  if lower > upper:
    raise ValueError(
        f'Lower cannot be greater than upper: given lower={lower} upper={upper}'
    )


def _get_feasible_points_and_bounds(
    feasible_values: Sequence[float],
) -> Tuple[List[float], Union[Tuple[int, int], Tuple[float, float]]]:
  """Validates and converts feasible values to floats."""
  if not all([math.isfinite(p) for p in feasible_values]):
    raise ValueError(
        f'Feasible values must all be finite. Given: {feasible_values}'
    )

  feasible_points = list(sorted(feasible_values))
  bounds = (feasible_points[0], feasible_points[-1])
  return feasible_points, bounds


def _get_categories(categories: Sequence[str]) -> List[str]:
  """Returns the categories."""
  return sorted(list(categories))


def _get_default_value(
    param_type: ParameterType, default_value: Union[float, int, str]
) -> Union[float, int, str]:
  """Validates and converts the default_value to the right type."""
  if param_type in (ParameterType.DOUBLE, ParameterType.DISCRETE) and (
      isinstance(default_value, float) or isinstance(default_value, int)
  ):
    return float(default_value)
  elif param_type == ParameterType.INTEGER and (
      isinstance(default_value, float) or isinstance(default_value, int)
  ):
    if isinstance(default_value, int):
      return default_value
    else:
      # Check if the float rounds nicely.
      default_int_value = round(default_value)
      if not math.isclose(default_value, default_int_value):
        raise ValueError(
            'default_value for an INTEGER parameter should be an '
            f'integer, got float: [{default_value}]'
        )
      return default_int_value
  elif param_type == ParameterType.CATEGORICAL and isinstance(
      default_value, str
  ):
    return default_value
  elif param_type == ParameterType.CUSTOM:
    return default_value
  raise ValueError(
      'default_value has an incorrect type. '
      f'ParameterType has type {param_type.name}, '
      f'but default_value has type {type(default_value)}'
  )


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
      converter=tuple, default=tuple(), kw_only=True
  )


########################
# Experimental features end here
########################


@attr.s(auto_attribs=True, frozen=False, init=True, slots=True, eq=True)
class ParameterConfig:
  """A Vizier ParameterConfig.

  Please use ParameterConfig.factory() to create an instance instead of calling
  the constructor directly.
  """

  _name: str = attr.ib(
      init=True, validator=attr.validators.instance_of(str), kw_only=True
  )
  _type: ParameterType = attr.ib(
      init=True,
      validator=attr.validators.instance_of(ParameterType),
      repr=lambda v: v.name if v is not None else 'None',
      kw_only=True,
  )
  # Only one of _feasible_values, _bounds will be set at any given time.
  _bounds: Optional[Union[Tuple[int, int], Tuple[float, float]]] = attr.ib(
      init=True,
      validator=attr.validators.optional(
          attr.validators.deep_iterable(
              member_validator=attr.validators.instance_of((int, float)),
              iterable_validator=attr.validators.instance_of(tuple),
          )
      ),
      kw_only=True,
  )
  _feasible_values: Optional[MonotypeParameterList] = attr.ib(
      init=True,
      validator=attr.validators.optional(
          attr.validators.deep_iterable(
              member_validator=attr.validators.instance_of((int, float, str)),
              iterable_validator=attr.validators.instance_of((list, tuple)),
          )
      ),
      kw_only=True,
  )
  _scale_type: Optional[ScaleType] = attr.ib(
      init=True,
      validator=attr.validators.optional(
          attr.validators.instance_of(ScaleType)
      ),
      repr=lambda v: v.name if v is not None else 'None',
      kw_only=True,
  )
  _default_value: Optional[Union[float, int, str]] = attr.ib(
      init=True,
      validator=attr.validators.optional(
          attr.validators.instance_of((float, int, str))
      ),
      kw_only=True,
  )
  _external_type: ExternalType = attr.ib(
      init=True,
      converter=lambda v: v or ExternalType.INTERNAL,
      validator=attr.validators.optional(
          attr.validators.instance_of(ExternalType)
      ),
      repr=lambda v: v.name if v is not None else 'None',
      kw_only=True,
  )

  # TODO: Make this a defaultdict and public.
  _children: dict[Union[float, int, str, bool], 'SearchSpace'] = attr.ib(
      init=True,
      factory=dict,
      # For equality checks, drop any empty search spaces.
      eq=lambda d: {k: v for k, v in d.items() if v.parameters},
      repr=lambda d: json.dumps(d, indent=2, default=repr),
  )

  # TODO: Deprecate this field.
  _matching_parent_values: MonotypeParameterSequence = attr.ib(
      init=True, default=tuple(), kw_only=True, eq=False
  )

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
      children: Optional[
          Sequence[Tuple[MonotypeParameterSequence, 'ParameterConfig']]
      ] = None,
      fidelity_config: Optional[FidelityConfig] = None,
      scale_type: Optional[ScaleType] = None,
      default_value: Optional[Union[float, int, str]] = None,
      external_type: Optional[ExternalType] = ExternalType.INTERNAL,
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

    if bool(feasible_values) and bool(bounds):
      raise ValueError(
          'While creating Parameter with name={}: one or none of '
          '"feasible_values" or "bounds" must be provided, but given '
          'feasible_values={} and bounds={}.'.format(
              name, feasible_values, bounds
          )
      )
    if feasible_values:
      if len(set(feasible_values)) != len(feasible_values):
        counter = collections.Counter(feasible_values)
        duplicate_dict = {k: v for k, v in counter.items() if v > 1}
        raise ValueError(
            'Feasible values cannot have duplicates: {}'.format(duplicate_dict)
        )
      if all(isinstance(v, (float, int)) for v in feasible_values):
        inferred_type = ParameterType.DISCRETE
        feasible_values, bounds = _get_feasible_points_and_bounds(
            feasible_values
        )
      elif all(isinstance(v, str) for v in feasible_values):
        inferred_type = ParameterType.CATEGORICAL
        feasible_values = _get_categories(feasible_values)
      else:
        raise ValueError(
            'Feasible values must all be numeric or strings. Given {}'.format(
                feasible_values
            )
        )
    elif bounds:  # bounds were specified.
      if isinstance(bounds[0], int) and isinstance(bounds[1], int):
        inferred_type = ParameterType.INTEGER
        _validate_bounds(bounds)
      elif isinstance(bounds[0], float) and isinstance(bounds[1], float):
        inferred_type = ParameterType.DOUBLE
        _validate_bounds(bounds)
      else:
        raise ValueError(
            'Bounds must both be integers or doubles. Given: {}'.format(bounds)
        )
    else:
      inferred_type = ParameterType.CUSTOM

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
    )
    if children:
      pc = pc._add_children(children)
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
      raise ValueError(
          'Accessing bounds of a categorical parameter: %s' % self.name
      )
    if self._bounds is None:
      raise ValueError(f'Accessing bounds when not set for {self.name}')
    return self._bounds

  @property
  def _child_parameter_configs(self) -> Iterator['ParameterConfig']:
    for subspace in self._children.values():
      for param in subspace.parameters:
        yield param

  # TODO: TO BE DEPRECATED. If we want to continue supporting multiple
  # matching parent values, expose "def compact_subspaces(self)" that returns
  # Iterator[tuple[MonotypeValueSequence, ParameterConfig]]
  @property
  def matching_parent_values(self) -> MonotypeParameterList:
    """Returns the matching parent values, if this is a child parameter."""
    if not self._matching_parent_values:
      return []
    return list(self._matching_parent_values)

  # TODO: TO BE DEPRECATED. Replace with
  # def subspaces() -> Iterator[Value, 'SearchSpace'] which lets users
  # iterate over all search spaces.
  @property
  def child_parameter_configs(self) -> List['ParameterConfig']:
    return copy.deepcopy(list(self._child_parameter_configs))

  def subspaces(
      self,
  ) -> Iterable[Tuple[ParameterValueTypes, 'SearchSpace']]:
    return self._children.items()

  # TODO: TO BE DEPRECATED.
  def _del_child_parameter_configs(self):
    """Deletes the current child ParameterConfigs."""
    self._children.clear()

  # TODO: Equivalent code should look like:
  # copied = copy.deepcopy(config)
  # for feasible_value in copied.feasible_values():
  #   copied.subspace(feasible_value).clear()
  @property
  def clone_without_children(self) -> 'ParameterConfig':
    """Returns the clone of self, without child_parameter_configs."""
    clone = copy.deepcopy(self)
    clone._del_child_parameter_configs()  # pylint: disable='protected-access'
    return clone

  @property
  def feasible_values(self) -> Union[List[int], List[float], List[str]]:
    """Sorted feasible values, or a ValueError if config is continuous."""
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

  @property
  def deterministic_value(self) -> Optional[Union[int, float, str]]:
    """Returns the value if ParameterConfig only allows one value."""
    if self.type in [ParameterType.DOUBLE, ParameterType.INTEGER]:
      min_val, max_val = self.bounds
      if min_val == max_val:
        return trial.ParameterValue(min_val).cast_as_internal(self.type)
    else:
      feasible_values = self.feasible_values
      if len(feasible_values) == 1:
        return trial.ParameterValue(self.feasible_values[0]).cast_as_internal(
            self.type
        )
    return None

  # TODO: TO BE DEPRECATED. Used by factory() only.
  def _add_children(
      self,
      new_children: Sequence[
          Tuple[MonotypeParameterSequence, 'ParameterConfig']
      ],
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
        raise ValueError(
            'Each element in new_children must be a tuple of '
            '(Sequence of valid parent values,  ParameterConfig),'
            ' given: {}'.format(child_pair)
        )

    logging.debug('_add_children: new_children=%s', new_children)
    for unsorted_parent_values, child in new_children:
      parent_values = sorted(unsorted_parent_values)
      for parent_value in parent_values:
        parent.subspace(parent_value).add(copy.deepcopy(child))
    return parent

  def continuify(self) -> 'ParameterConfig':
    """Returns a newly created DOUBLE parameter with the same range."""
    if self.type == ParameterType.DOUBLE:
      return copy.deepcopy(self)
    elif not self.type.is_numeric():
      raise ValueError(
          'Cannot convert a non-numeric parameter to DOUBLE: {}'.format(self)
      )
    elif list(self._child_parameter_configs):
      raise ValueError(
          'Cannot convert a parent parameter to DOUBLE: {}'.format(self)
      )

    scale_type = self.scale_type
    if scale_type == ScaleType.UNIFORM_DISCRETE:
      logging.log_every_n(
          logging.WARNING,
          (
              'Converting a UNIFORM_DISCRETE scaled discrete parameter '
              'to DOUBLE: %s'
          ),
          10,
          self,
      )
      scale_type = None

    default_value = self.default_value
    if default_value is not None:
      default_value = float(default_value)
    return ParameterConfig.factory(
        self.name,
        bounds=(float(self.bounds[0]), float(self.bounds[1])),
        scale_type=scale_type,
        default_value=default_value,
    )

  @classmethod
  def merge(
      cls, one: 'ParameterConfig', other: 'ParameterConfig'
  ) -> 'ParameterConfig':
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
          'Cannot merge parameters with child_parameter_configs: %s and %s'
          % one,
          other,
      )
    if one.type != other.type:
      raise ValueError(
          'Type conflicts between {} and {}'.format(
              one.type.name, other.type.name
          )
      )
    if one.scale_type != other.scale_type:
      logging.warning(
          'Scale type conflicts while merging %s and %s', one, other
      )

    if one.type in (ParameterType.CATEGORICAL, ParameterType.DISCRETE):
      new_feasible_values = list(
          set(one.feasible_values + other.feasible_values)
      )
      return ParameterConfig.factory(
          name=one.name,
          feasible_values=new_feasible_values,
          scale_type=one.scale_type,
      )
    elif one.type in (ParameterType.INTEGER, ParameterType.DOUBLE):
      original_min, original_max = one.bounds
      other_min, other_max = other.bounds
      new_bounds = (min(original_min, other_min), max(original_max, other_max))
      return ParameterConfig.factory(
          name=one.name, bounds=new_bounds, scale_type=one.scale_type
      )
    raise ValueError(
        'Unknown type {}. This is currentlyan unreachable code.'.format(
            one.type
        )
    )

  def traverse(
      self, show_children: bool = False
  ) -> Generator['ParameterConfig', None, None]:
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

  # TODO: Rename to `validate_value or is_feasible`
  def contains(
      self, value: Union[trial.ParameterValueTypes, trial.ParameterValue]
  ) -> bool:
    """Check if the `value` is a valid value for this parameter config."""
    if isinstance(value, trial.ParameterValue):
      # TODO: Extract the raw value.
      value = value.value
    try:
      self._assert_feasible(value)
    except (TypeError, ValueError):
      return False
    return True

  @property
  def num_feasible_values(self) -> Union[float, int]:
    if self.type == ParameterType.DOUBLE:
      return float('inf')
    elif self.type == ParameterType.INTEGER:
      return self.bounds[1] - self.bounds[0] + 1
    elif self.type == ParameterType.CUSTOM:
      return float('inf')
    else:
      return len(self.feasible_values)

  def _assert_bounds(self, value: trial.ParameterValueTypes) -> None:
    if not self.bounds[0] <= value <= self.bounds[1]:
      raise ValueError(
          f'Parameter {self.name} has bounds: {self.bounds}. Given: {value}'
      )

  def _assert_in_feasible_values(
      self, value: trial.ParameterValueTypes
  ) -> None:
    if value not in self._feasible_values:
      raise ValueError(
          f'Parameter {self.name} has feasible values: '
          f'{self.feasible_values}. '
          f'Given: {value}'
      )

  def _assert_feasible(self, value: trial.ParameterValueTypes) -> None:
    """Asserts that the value is feasible for this parameter config.

    Args:
      value:

    Raises:
      TypeError: Value does not match the config's type
      ValueError: Value is not feasible.
      RuntimeError: Other errors.
    """
    try:
      self.type.assert_correct_type(value)
    except TypeError as e:
      raise TypeError(
          f'Parameter {self.name} is not compatible with value: {value}'
      ) from e

    # TODO: We should be able to directly use "value" without
    # casting to the internal type.
    value = trial.ParameterValue(value)
    if self.type == ParameterType.DOUBLE:
      self._assert_bounds(value.as_float)
    elif self.type == ParameterType.INTEGER:
      self._assert_bounds(value.as_int)
    elif self.type == ParameterType.DISCRETE:
      self._assert_in_feasible_values(value.as_float)
    elif self.type == ParameterType.CATEGORICAL:
      self._assert_in_feasible_values(value.as_str)
    else:
      raise RuntimeError(
          f'Parameter {self.name} has unknown parameter type: {self.type}'
      )

  def get_subspace_deepcopy(self, value: ParameterValueTypes) -> 'SearchSpace':
    """Get a deep copy of the subspace.

    Validates the feasibility of value.

    Args:
      value: Must be a feasible value per this parameter config.

    Returns:
      Subspace conditioned on the value. Note that an empty search space is
      returned if the parameter config is continuous and thus cannot have
      a subspace.
    """
    if not math.isfinite(self.num_feasible_values):
      return SearchSpace()
    value = trial.ParameterValue(value).cast_as_internal(self.type)
    self._assert_feasible(value)
    return copy.deepcopy(self._children.get(value, SearchSpace()))

  def subspace(self, value: ParameterValueTypes) -> 'SearchSpace':
    """Selects the subspace for a specified parent value."""
    if not math.isfinite(self.num_feasible_values):
      raise TypeError('DOUBLE type cannot have child parameters')

    # TODO: We should be able to directly use "value".
    value = trial.ParameterValue(value).cast_as_internal(self.type)
    self._assert_feasible(value)
    if value not in self._children:
      self._children[value] = SearchSpace(parent_values=[value])
    return self._children[value]


@attr.define(init=False)
class ParameterConfigSelector(Iterable[ParameterConfig], Sized):
  """Holds a reference to ParameterConfigs."""

  # Selected configs.
  _selected: tuple[ParameterConfig] = attr.field(init=True, converter=tuple)

  def __iter__(self) -> Iterator[ParameterConfig]:
    return iter(self._selected)

  def __len__(self) -> int:
    return len(self._selected)

  def __init__(
      self, selected: Union[ParameterConfig, Iterable[ParameterConfig]], /
  ):
    if isinstance(selected, ParameterConfig):
      self.__attrs_init__(tuple([selected]))
    else:
      self.__attrs_init__(tuple(selected))

  def select_values(
      self, values: MonotypeParameterSequence
  ) -> 'SearchSpaceSelector':
    """Select values."""
    values = tuple(values)

    for value in values:
      for config in self._selected:
        if not config.contains(value):
          # Validate first so we don't create a lot of unnecessary empty
          # search space upon failure.
          raise ValueError(f'{value} is not feasible in {self}')

    spaces = []
    for value in values:
      for config in self._selected:
        spaces.append(config.subspace(value))
    return SearchSpaceSelector(spaces)

  def merge(self) -> 'ParameterConfigSelector':
    """Merge by taking the union of the parameter configs with the same name.

    Returns:
      The returned ParameterConfigSelector does not contain parameters with
      duplicate names. Their feasible set (either as a range or discrete set) is
      the union of all feasible sets under the same parameter name.
    """
    merged_configs = {}
    for parameter_config in self:
      name = parameter_config.name  # Alias
      existing_config = merged_configs.setdefault(name, parameter_config)
      merged_configs[name] = ParameterConfig.merge(
          existing_config, parameter_config
      )
    return ParameterConfigSelector(merged_configs.values())


class InvalidParameterError(ValueError):
  """Error thrown when parameter values are invalid."""


################### Main Classes ###################


@attr.define(init=False)
class SearchSpaceSelector:
  """Holds a reference to (sub) spaces."""

  # Selected (sub)-spaces.
  # TODO: Consider switching the order of SearchSpaceSelector and
  # SearchSpace.
  _selected: tuple['SearchSpace'] = attr.field(init=True)

  def __len__(self) -> int:
    return len(self._selected)

  def __init__(
      self, selected: Union['SearchSpace', Iterable['SearchSpace']], /
  ):
    if isinstance(selected, SearchSpace):
      self.__attrs_init__(tuple([selected]))
    else:
      self.__attrs_init__(tuple(selected))

  def add_float_param(
      self,
      name: str,
      min_value: float,
      max_value: float,
      *,
      default_value: Optional[float] = None,
      scale_type: Optional[ScaleType] = None,
      index: Optional[int] = None,
  ) -> 'ParameterConfigSelector':
    """Adds floating point parameter config(s) to the selected search space(s).

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
    if scale_type is None:
      scale_type = ScaleType.LINEAR
    bounds = (float(min_value), float(max_value))
    param_names = self._get_parameter_names_to_create(name=name, index=index)

    new_params = []
    for param_name in param_names:
      new_pc = ParameterConfig.factory(
          name=param_name,
          bounds=bounds,
          scale_type=scale_type,
          default_value=default_value,
      )
      new_params.append(new_pc)
    return self._add_parameters(new_params)

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
  ) -> 'ParameterConfigSelector':
    """Adds integer parameter config(s) to the selected search space(s).

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
      ParameterConfigSelector for the newly added parameter(s).

    Raises:
      ValueError: If min_value or max_value are not integers.
      ValueError: If `index` is invalid (e.g. negative).
    """
    int_min_value = int(min_value)
    if not math.isclose(min_value, int_min_value):
      raise ValueError(
          'min_value for an INTEGER parameter should be an integer'
          ', got: [{}]'.format(min_value)
      )
    int_max_value = int(max_value)
    if not math.isclose(max_value, int_max_value):
      raise ValueError(
          'max_value for an INTEGER parameter should be an integer'
          ', got: [{}]'.format(min_value)
      )
    bounds = (int_min_value, int_max_value)

    param_names = self._get_parameter_names_to_create(name=name, index=index)

    new_params = []
    for param_name in param_names:
      new_pc = ParameterConfig.factory(
          name=param_name,
          bounds=bounds,
          scale_type=scale_type,
          fidelity_config=experimental_fidelity_config,
          default_value=default_value,
      )
      new_params.append(new_pc)
    return self._add_parameters(new_params)

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
  ) -> 'ParameterConfigSelector':
    """Adds ordered numeric parameter config(s) with a finite set of values.

    IMPORTANT: If a parameter is discrete, its values are assumed to have
    ordered semantics. Thus, you should not use discrete parameters for
    unordered values such as ids. In this case, see add_categorical_param()
    below.

    Args:
      name: The parameter's name. Cannot be empty.
      feasible_values: The set of feasible values for this parameter.
      default_value: A default value for the Parameter.
      scale_type: Scaling to be applied. NOT VALIDATED.
      index: Specifies the multi-dimensional index for this parameter. E.g. if
        name='batch_size' and index=0, then a single ParameterConfig with name
        'batch_size[0]' is added. `index` should be >= 0.
      auto_cast: If True, the external type will be set to INTEGER if all values
        are castable to an integer without losing precision. If False, the
        external type will be set to float.
      experimental_fidelity_config: EXPERIMENTAL. See FidelityConfig doc.

    Returns:
      ParameterConfigSelector for the newly added parameter(s).

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
          external_type=external_type,
      )
      new_params.append(new_pc)
    return self._add_parameters(new_params)

  def add_categorical_param(
      self,
      name: str,
      feasible_values: Sequence[str],
      *,
      default_value: Optional[str] = None,
      scale_type: Optional[ScaleType] = None,
      index: Optional[int] = None,
  ) -> 'ParameterConfigSelector':
    """Adds string-valued parameter config(s) to the selected search space(s).

    IMPORTANT: If a parameter is categorical, its values are assumed to be
    unordered. If the `feasible_values` have ordering, use add_discrete_param()
    above, since it will improve Vizier's model quality.

    Args:
      name: The parameter's name. Cannot be empty.
      feasible_values: The set of feasible values for this parameter.
      default_value: A default value for the Parameter.
      scale_type: Scaling to be applied. NOT VALIDATED.
      index: Specifies the multi-dimensional index for this parameter. E.g. if
        name='id' and index=0, then a single ParameterConfig with name 'id[0]'
        is added. `index` should be >= 0.

    Returns:
      ParameterConfigSelector for the newly added parameter(s).

    Raises:
      ValueError: If `index` is invalid (e.g. negative), or `feasible_values`
        are invalid (not strings).
    """
    for value in feasible_values:
      if not isinstance(value, str):
        raise ValueError(f'feasible_values must be strings; got: {value}')

    param_names = self._get_parameter_names_to_create(name=name, index=index)

    new_params = []
    for param_name in param_names:
      new_pc = ParameterConfig.factory(
          name=param_name,
          feasible_values=sorted(feasible_values),
          scale_type=scale_type,
          default_value=default_value,
      )
      new_params.append(new_pc)
    return self._add_parameters(new_params)

  def add_custom_param(
      self,
      name: str,
      *,
      default_value: Optional[ParameterValueTypes] = None,
  ) -> 'ParameterConfigSelector':
    """Adds custom parameter config(s) to the selected search space(s).

    Args:
      name: The parameter's name. Cannot be empty.
      default_value: A default value for the Parameter. Generally should be set.

    Returns:
      ParameterConfigSelector for the newly added parameter(s).

    Raises:
      ValueError: If `index` is invalid (e.g. negative)
    """
    new_pc = ParameterConfig.factory(
        name=name,
        default_value=default_value,
    )
    return self._add_parameters([new_pc])

  def add_bool_param(
      self,
      name: str,
      feasible_values: Optional[Sequence[bool]] = None,
      *,
      default_value: Optional[bool] = None,
      scale_type: Optional[ScaleType] = None,
      index: Optional[int] = None,
  ) -> 'ParameterConfigSelector':
    """Adds boolean-valued parameter config(s) to the selected search space(s).

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
      ParameterConfigSelector for the newly added parameter(s).

    Raises:
      ValueError: If `feasible_values` has invalid values.
      ValueError: If `index` is invalid (e.g. negative).
    """
    allowed_values = (None, (True, False), (False, True), (True,), (False,))
    if (
        feasible_values is not None
        and tuple(feasible_values) not in allowed_values
    ):
      raise ValueError(
          'feasible_values must be one of %s; got: %s.'
          % (allowed_values, feasible_values)
      )
    # Boolean parameters are represented as categorical parameters internally.
    bool_to_string = lambda x: trial.TRUE_VALUE if x else trial.FALSE_VALUE
    if feasible_values is None:
      categories = (trial.TRUE_VALUE, trial.FALSE_VALUE)
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
          external_type=ExternalType.BOOLEAN,
      )
      new_params.append(new_pc)
    return self._add_parameters(new_params)

  @overload
  def select(
      self,
      parameter_name: str,
      parameter_values: None,
  ) -> ParameterConfigSelector:
    ...

  @overload
  def select(
      self, parameter_name: str, parameter_values: MonotypeParameterSequence
  ) -> 'SearchSpaceSelector':
    ...

  def select(
      self,
      parameter_name,
      parameter_values: Optional[MonotypeParameterSequence] = None,
  ):
    """Selects a parameter config or its subspace.

    This method is for constructing a _conditional_ search space.

    EXAMPLE: Suppose we have a selector to the root of the search space with one
    categorical parameter.
    root = pyvizier.SearchSpace().root
    root.add_categorical_param('model_type', ['dnn', 'linear'])

    1) Select a `ParameterConfig`:
      model = root.select('model_type')

    2) Select a subspace conditioned on `model_type == 'dnn'` and add
    a child parameter `hidden_units`:
      dnn_subspace = root.select('model_type', ['dnn'])
      dnn_subspace.add_int_param('hidden_layers', ...)

    or equivalently,
      dnn_subspace = root.select('model_type').select_values(['dnn'])
      dnn_subspace.add_int_param('hidden_layers', ...)

    3) Traverse your search space by chaining select() calls:
      root.select('model_type', ['dnn']).select('hidden_layers', [1, 2])

    4) Select more than one search space simultaneously:
      selected = root.select('model_type', ['dnn', 'linear'])
        .add_categorical_param('optimizer', ['adam', 'adagrad'])
      assert len(selected) == 4  # {dnn, linear} x {adam, adagard}

    Args:
      parameter_name:
      parameter_values: Optional parameter values for this selector, which will
        be used to add child parameters, or traverse a conditional tree.

    Returns:
      ParameterConfigSelector for `ParameterConfig`(s) if the values are not
        specified.
      SearchSpaceSelector for subspace(s) if parameter_values are specified.
    """
    if parameter_values is None:
      selected_configs = []
      for space in self._selected:
        selected_configs.append(space.get(parameter_name))
      return ParameterConfigSelector(selected_configs)
    else:
      selected_spaces = []
      for space in self._selected:
        selected_parameter = space.get(parameter_name)
        for value in parameter_values:
          selected_spaces.append(selected_parameter.subspace(value))
      return SearchSpaceSelector(selected_spaces)

  @classmethod
  def _get_parameter_names_to_create(
      cls,
      *,
      name: str,
      length: Optional[int] = None,
      index: Optional[int] = None,
  ) -> List[str]:
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
      raise ValueError(
          'Only one of `length` and `index` can be specified. Got'
          ' length={}, index={}'.format(length, index)
      )
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
      cls, name: str
  ) -> Optional[Tuple[str, int]]:
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

  # TODO: Add def extend(space: SearchSpace)
  def _add_parameters(
      self, parameters: Iterable[ParameterConfig]
  ) -> ParameterConfigSelector:
    """Adds deepcopy of the ParameterConfigs.

    Args:
      parameters: The parameters to add to the search space.

    Returns:
      A list of SearchSpaceSelectors, one for each parameters added.
    """
    parameters = list(parameters)
    logging.info(
        'Adding child parameters %s to %s subspaces ',
        set(p.name for p in parameters),
        len(self._selected),
    )
    added = []
    for parameter in parameters:
      for selected in self._selected:
        # Adds a deepcopy so that every ParameterConfig object is unique.
        added.append(selected.add(copy.deepcopy(parameter)))

    return ParameterConfigSelector(added)

  def select_all(self) -> ParameterConfigSelector:
    """Select all parameters at all levels."""
    all_parameter_configs = []
    for space in self._selected:
      for top_level_config in space.parameters:
        all_parameter_configs.extend(list(top_level_config.traverse()))

    return ParameterConfigSelector(all_parameter_configs)


@attr.define(frozen=False, init=True, slots=True, kw_only=True)
class SearchSpace:
  """[Cross-platform] Collection of ParameterConfigs.

  Vizier search space can be *conditional*.
  Parameter names are guaranteed to be unique in any subspace.

  Attribute:
    _parameter_configs: Maps parameter names to configs.
  """

  _parameter_configs: dict[str, ParameterConfig] = attr.field(
      init=False, factory=dict
  )

  # TODO: To be deprecated.
  _parent_values: MonotypeParameterSequence = attr.field(
      default=tuple(), converter=tuple, kw_only=True
  )

  @property
  def parameter_names(self) -> AbstractSet[str]:
    return self._parameter_configs.keys()

  def get(self, name: str) -> ParameterConfig:
    if name not in self._parameter_configs:
      raise KeyError(f'{name} is not in the search space.')
    return self._parameter_configs[name]

  def pop(self, name: str) -> ParameterConfig:
    return self._parameter_configs.pop(name)

  def add(
      self, parameter_config: ParameterConfig, *, replace: bool = False
  ) -> ParameterConfig:
    """Adds the ParameterConfig.

    For advanced users only. Takes a reference to Parameterconfig.
    Future edits will change the search space.

    Args:
      parameter_config:
      replace: Determines the behavior when there already exists a
        ParameterConfig with the same name. If set to True, replaces it. If set
        to False, raises ValueError.

    Returns:
      Reference to the ParameterConfig that was added to the search space.
    """
    name = parameter_config.name
    parameter_config._matching_parent_values = tuple(self._parent_values)  # pylint: disable=protected-access
    if (name in self._parameter_configs) and (not replace):
      raise ValueError(
          f'Duplicate name: {parameter_config.name} already exists.\n'
          f'Existing config: {parameter_config}\n'
          f'New config:{parameter_config}'
      )

    self._parameter_configs[name] = parameter_config
    return parameter_config

  # TODO: Change the return type to Iterator.
  @property
  def parameters(self) -> list[ParameterConfig]:
    """Returns the parameter configs in this search space."""
    return list(self._parameter_configs.values())

  def select_root(self) -> SearchSpaceSelector:
    # Deprecated function.
    # TODO: Remove this from downstream user code.
    return SearchSpaceSelector(self)

  @property
  def root(self) -> SearchSpaceSelector:
    """Returns a selector for the root of the search space.

    Parameters can be added to the search space using the returned
    SearchSpaceSelector.
    """
    return SearchSpaceSelector(self)

  @property
  def is_conditional(self) -> bool:
    """Returns True if search_space contains any conditional parameters."""
    return any([p.child_parameter_configs for p in self.parameters])

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
    if len(parameters) != len(self._parameter_configs.values()):
      set1 = set(pc.name for pc in self._parameter_configs.values())
      set2 = set(parameters)
      raise InvalidParameterError(
          f'Search space has {len(self._parameter_configs.values())} parameters'
          f' but only {len(parameters)} were given. Missing in search space:'
          f' {set2 - set1}. Missing in parameters: {set1 - set2}.'
      )
    for pc in self._parameter_configs.values():
      if pc.name not in parameters:
        raise InvalidParameterError(f'{pc.name} is missing in {parameters}.')
      elif not pc.contains(parameters[pc.name]):
        raise InvalidParameterError(
            f'{parameters[pc.name]} is not feasible in {pc}'
        )
    return True

  def num_parameters(self, param_type: Optional[ParameterType] = None) -> int:
    """Counts number of parameters with the param_type (if given)."""
    if param_type is None:
      return len(self.parameters)
    return [pc.type for pc in self.parameters].count(param_type)
