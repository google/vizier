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

"""Library for padding inputs and arrays in order to reduce chances of recompilation."""

import enum
import math
from typing import Sequence

import attrs
import numpy as np
from vizier._src.jax import types


class PaddingType(enum.Enum):
  NONE = 1
  MULTIPLES_OF_10 = 2
  POWERS_OF_2 = 3


def _padded_dimensions(
    dims: Sequence[int], padding_types: Sequence[PaddingType]
) -> tuple[int, ...]:
  """Returns the padded shape according to `padding_types`."""
  new_dims = []
  for dim, padding_type in zip(dims, padding_types):
    if padding_type == PaddingType.NONE:
      new_dims.append(dim)
    elif padding_type == PaddingType.MULTIPLES_OF_10:
      new_dims.append(int(math.ceil(dim / 10.0)) * 10)
    elif padding_type == PaddingType.POWERS_OF_2:
      if dim == 0:
        new_dims.append(0)
      else:
        new_dims.append(int(2 ** (math.ceil(math.log(dim, 2)))))
    else:
      raise ValueError(f'{padding_type} unexpected.')
  return tuple(new_dims)


@attrs.define(frozen=True, kw_only=True, hash=True, eq=True)
class PaddingSchedule:
  """Convenience class for creating `PaddedArray`."""

  _num_trials: PaddingType = PaddingType.NONE
  _num_features: PaddingType = PaddingType.NONE
  _num_metrics: PaddingType = PaddingType.NONE

  _int_default: int = -1
  _float_default: float = np.nan

  def _pad_trailing_dims(
      self, array: np.ndarray, padding_types: Sequence[PaddingType]
  ) -> types.PaddedArray:
    """Pads features in to a `PaddedArray`."""
    assert len(padding_types) == len(array.shape)

    original_shape = array.shape[-len(padding_types) :]
    padded_shape = array.shape[: -len(padding_types)] + _padded_dimensions(
        original_shape, padding_types
    )

    if np.issubdtype(array.dtype, np.integer):
      fill_value = self._int_default
    else:
      fill_value = np.nan
    return types.PaddedArray.from_array(
        array, padded_shape, fill_value=fill_value
    )

  def pad_features(self, features: types.Array) -> types.PaddedArray:
    """Pads features in to a `PaddedArray`."""
    return self._pad_trailing_dims(
        features, [self._num_trials, self._num_features]
    )

  def pad_labels(
      self,
      labels: types.Array,
  ) -> types.PaddedArray:
    """Pads labels in to a `PaddedArray`."""
    return self._pad_trailing_dims(
        labels, [self._num_trials, self._num_metrics]
    )
