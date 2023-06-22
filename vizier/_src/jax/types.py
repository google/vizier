# Copyright 2023 Google LLC.
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

from typing import Any, Generic, Iterable, List, Mapping, Optional, TypeVar, Union

import attrs
import equinox as eqx
from flax import struct
from flax.core import scope as flax_scope
import jax
from jax import numpy as jnp
from jax.typing import ArrayLike
import numpy as np


# We define our own Array type since `jax.typing.Array` and `chex.Array` both
# include scalar types, which result in type errors when array
# methods/properties like `.shape` are accessed.
Array = Union[np.ndarray, jax.Array]


@attrs.define(frozen=True, kw_only=True)
class PaddedArray:
  # Array of shape [N1, ... Nk].
  padded_array: Array = attrs.field(init=True)
  # Mask per dimension padded. List of Arrays of shape [N1], [N2], ..., [Nk].
  is_missing: List[Array] = attrs.field(init=True)

  @property
  def shape(self):
    return self.padded_array.shape


MaybePaddedArray = Union[Array, PaddedArray]

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


ContinuousAndCategoricalArray = ContinuousAndCategorical[Array]


# Tuple representing a box constraint of the form (lower, upper) bounds.
Bounds = tuple[Optional[ArrayTreeOptional], Optional[ArrayTreeOptional]]

# TODO: Rename to FeatureType, and introduce
# Feature = Union[jax.Array, PaddedArray, ContinuousAndCategoricalArray]
Features = TypeVar('Features', Array, ContinuousAndCategoricalArray)


# TODO: Make this class an eqx.Module.
class StochasticProcessModelData(Generic[Features], eqx.Module):
  """Data that feed into GP."""

  features: Features
  labels: Array = eqx.field(converter=lambda x: jnp.asarray(x, dtype=x.dtype))
  label_is_missing: Optional[Array] = None
  dimension_is_missing: Optional[Features] = None


@struct.dataclass
class GPState(Generic[Features]):
  """State that changes at each iteration."""

  data: StochasticProcessModelData[Features]
  model_state: ModelState
