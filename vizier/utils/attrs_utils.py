"""Utils for attrs."""

from typing import Any, Optional, Collection, Callable


def shape_equals(instance_to_shape: Callable[[Any], Collection[Optional[int]]]):
  """Creates a shape validator for attrs.

  For example, _shape_equals(lambda s : [3, None]) validates that the shape has
  length 2 and its first element is 3.

  Code Example:
  @attrs.define
  class TestAttr:
    x = attrs.field(validator=attrs_utils.shape_equals(lambda v: (3, v.d)))
    d = attrs.field()

  _TestAttr(np.zeros([3, 2]), 2)  # OK
  _TestAttr(np.zeros([3, 5]), None) # OK
  _TestAttr(np.zeros([3, 2]), 4)  # Raises ValueError

  Args:
    instance_to_shape: Takes instance as input and returns the desired shape for
      the instance. `None` is treated as "any number".

  Returns:
    A validator that can be passed into attrs.ib or attrs.field.
  """

  def validator(instance, attribute, value) -> None:
    shape = instance_to_shape(instance)

    def _validator_boolean():
      if len(value.shape) != len(shape):
        return False
      for s1, s2 in zip(value.shape, shape):
        if (s2 is not None) and (s1 != s2):
          return False
      return True

    if not _validator_boolean():
      raise ValueError(f'{attribute.name} has shape {value.shape} '
                       f'which does not match the expected shape {shape}')

  return validator
