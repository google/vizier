"""ParameterConfig wraps ParameterConfig and ParameterSpec protos."""

import collections
import copy
import enum
import math
from typing import Generator, List, Optional, Sequence, Tuple, Union

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


class ScaleType(enum.Enum):
  """Valid Values for ParameterConfig.scale_type."""
  LINEAR = 'LINEAR'
  LOG = 'LOG'
  REVERSE_LOG = 'REVERSE_LOG'
  UNIFORM_DISCRETE = 'UNIFORM_DISCRETE'


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
        ParameterConfig). See
        cs/learning_vizier.service.ParameterConfig.child_parameter_configs for
        details. ONLY THE TYPES ARE VALIDATED. If the child ParameterConfig
        protos already have parent values set, they will be overridden by the
        provided matching_parent_values.
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
