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

from typing import Any, Iterable, Mapping, Optional, Union

from flax.core import scope as flax_scope
import jax
from jax.typing import ArrayLike
import numpy as np


# We define our own Array type since `jax.typing.Array` and `chex.Array` both
# include scalar types, which result in type errors when array
# methods/properties like `.shape` are accessed.
Array = Union[np.ndarray, jax.Array]

ArrayTree = Union[ArrayLike, Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]

# An ArrayTree that allows None values.
ArrayTreeOptional = Union[
    ArrayLike,
    Iterable[Optional['ArrayTreeOptional']],
    Mapping[Any, Optional['ArrayTreeOptional']],
]
ParameterDict = flax_scope.Collection
ModelState = flax_scope.VariableDict
