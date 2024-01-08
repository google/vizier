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

"""Types library for vizier/_src/jax."""

from typing import Any, Generic, Iterable, Mapping, Optional, Sequence, TypeVar, Union
import equinox as eqx
from flax import struct
from flax.core import scope as flax_scope
import jax
from jax import numpy as jnp
from jax.typing import ArrayLike
import jaxtyping as jt
import numpy as np

# Integer dtype for categorical data.
INT_DTYPE = np.int32


# We define our own Array type since `jax.typing.Array` and `chex.Array` both
# include scalar types, which result in type errors when array
# methods/properties like `.shape` are accessed.
Array = Union[np.ndarray, jax.Array]


# TODO: Add a `concatenate` method.
class PaddedArray(eqx.Module):
  """Padded Array as a pytree.

  There is no validation done in `__init__`. In order to get a validated
  instance, Use a converter object or a factory method outside the jit scope.

  Attributes:
    padded_array:
    _original_shape: 1D array of length equal to `padded_array.ndims`. The k-th
      element must be less than or equal to `padded_array.shape[k]`.
    _nopadding_done: Must be set to True iff `_original_shape` is the same as
      padded_array.shape.
    _mask: Same shape as padded_array. True where padded_array's value is the
      original array value as opposed to the fill value.
  """

  # TODO: Rename "padded_array" to a shorter name like "padded".
  padded_array: jt.Shaped[jax.Array, '...'] = eqx.field(converter=jnp.asarray)
  fill_value: Any = eqx.field(converter=jnp.asarray)
  # TODO: Make `_original_shape` public.
  _original_shape: jt.Int[jax.Array, '_'] = eqx.field(converter=jnp.asarray)

  # TODO: Make _mask an inferred property. The current
  # implementation is not memory efficient.
  _mask: jt.Bool[jax.Array, '...'] = eqx.field(converter=jnp.asarray)
  _nopadding_done: bool = eqx.field(static=True, default=False, kw_only=True)

  @property
  def shape(self) -> tuple[int, ...]:
    """Returns the shape of the padded array."""
    return self.padded_array.shape

  def replace_array(self, array: jt.Shaped[jax.Array, '...']) -> 'PaddedArray':
    """Replaces the original array values, maintaining the padding."""
    return PaddedArray.from_array(array, self.shape, fill_value=self.fill_value)

  def replace_fill_value(self, fill_value: Any) -> 'PaddedArray':
    """Replaces the padded fill values."""
    # TODO: Consider optimizing when fill_value == self.fill_value.
    if self._nopadding_done:
      return self
    else:
      return PaddedArray(
          jnp.where(self._mask, self.padded_array, fill_value),
          fill_value=fill_value,
          _original_shape=self._original_shape,
          _mask=self._mask,
          _nopadding_done=self._nopadding_done,
      )

  @classmethod
  def as_padded(cls, array: jt.Shaped[jax.Array, '...']) -> 'PaddedArray':
    """Converts a regular array as PaddedArray type, with no actual padding.

    NOTE This is implemented as a separate method instead of setting default
    for `fill_value=np.nan` in `from_array` method. `jnp.pad`` automatically
    casts `nan` to `0` for integer arrays, which can cause unexpected
    behavior, and we want people to always explicitly set the fill value
    unless they know for sure that padding would not occur.

    Args:
      array:

    Returns:
      PaddedArray.
    """
    return PaddedArray.from_array(array, array.shape, fill_value=np.nan)

  @classmethod
  def from_array(
      cls,
      array: jt.Shaped[jax.Array, '...'],
      target_shape: Sequence[int],
      *,
      fill_value: Any
  ) -> 'PaddedArray':
    """Factory function to guarantee a self-consistent creation."""
    spec = [(0, dnew - d) for d, dnew in zip(array.shape, target_shape)]
    mask_array = jnp.pad(
        jnp.ones_like(array, dtype=bool), spec, constant_values=False
    )
    new_array = jnp.pad(array, spec, constant_values=fill_value)
    return PaddedArray(
        padded_array=new_array,
        fill_value=fill_value,
        _original_shape=array.shape,
        _mask=mask_array,
        _nopadding_done=array.shape == target_shape,
    )

  @property
  def is_missing(self) -> tuple[jt.Bool[jax.Array, '...']]:
    """Mask per dimension padded. Arrays have shape [N1], [N2], ..., [Nk]."""
    return tuple(
        jnp.arange(s1) >= s2
        for s1, s2 in zip(self.padded_array.shape, self._original_shape)
    )

  def unpad(self) -> jt.Shaped[jax.Array, '...']:
    """Can be used in jit scope only if original shape == padded shape."""
    if self._nopadding_done:
      return self.padded_array
    return jax.lax.slice(
        self.padded_array,
        [0] * self.padded_array.ndim,
        self._original_shape.tolist(),
    )


ArrayTree = Union[ArrayLike, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# An ArrayTree that allows None values.
ArrayTreeOptional = Union[
    ArrayLike,
    Iterable[Optional['ArrayTreeOptional']],
    Mapping[Any, Optional['ArrayTreeOptional']],
]
ParameterDict = flax_scope.Collection
ModelState = flax_scope.VariableDict


_T = TypeVar('_T')


@struct.dataclass
class ContinuousAndCategorical(Generic[_T]):
  continuous: _T
  categorical: _T


# TODO: Make it a private type inside jnp_converters.py.
ContinuousAndCategoricalArray = ContinuousAndCategorical[Array]

ModelInput = ContinuousAndCategorical[PaddedArray]


class ModelData(eqx.Module):
  features: ModelInput
  labels: PaddedArray


# Tuple representing a box constraint of the form (lower, upper) bounds.
Bounds = tuple[Optional[ArrayTreeOptional], Optional[ArrayTreeOptional]]


# TODO: Deprecate it in favor of
# PrecomputedPredictive for full predictive state including cholesky, and
# StochasticProcessWithCoroutine for computing log probs.
@struct.dataclass
class GPState:
  """State that changes at each iteration."""

  data: ModelData
  model_state: ModelState
