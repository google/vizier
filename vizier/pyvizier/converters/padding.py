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

"""Library for padding inputs and arrays in order to reduce chances of recompilation."""

import enum
import math
from typing import List, Tuple
import attrs
import numpy as np
from vizier._src.jax import types


class PaddingType(enum.Enum):
  NONE = 1
  MULTIPLES_OF_10 = 2
  POWERS_OF_2 = 3


@attrs.define(frozen=True, kw_only=True)
class PaddingSchedule:
  num_trials: PaddingType = attrs.field(init=True)
  num_features: PaddingType = attrs.field(init=True)


def padded_dimensions(
    dims: List[int], padding_types: List[PaddingType]
) -> List[int]:
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
  return new_dims


def pad_features(
    features: np.ndarray, padding_schedule: PaddingSchedule
) -> types.PaddedArray:
  """Pads features in to a `PaddedArray`."""
  num_trials, num_features = features.shape[-2], features.shape[-1]
  padding_types = [
      padding_schedule.num_trials,
      padding_schedule.num_features,
  ]
  new_num_trials, new_num_features = padded_dimensions(  # pylint: disable=unbalanced-tuple-unpacking
      [num_trials, num_features], padding_types
  )
  features_is_missing = np.array(
      [False] * num_features + [True] * (new_num_features - num_features)
  )
  trials_is_missing = np.array(
      [False] * num_trials + [True] * (new_num_trials - num_trials)
  )
  new_features = np.pad(
      features,
      ((0, new_num_trials - num_trials), (0, new_num_features - num_features)),
      constant_values=np.nan,
  )
  return types.PaddedArray(
      padded_array=new_features,
      is_missing=[trials_is_missing, features_is_missing],
  )


def pad_labels(
    labels: np.ndarray, padding_schedule: PaddingSchedule
) -> types.PaddedArray:
  """Pads labels in to a `PaddedArray`."""
  num_trials = labels.shape[-2]
  (new_num_trials,) = padded_dimensions(  # pylint:disable=unbalanced-tuple-unpacking
      [num_trials], [padding_schedule.num_trials]
  )
  trials_is_missing = np.array(
      [False] * num_trials + [True] * (new_num_trials - num_trials)
  )
  new_labels = np.pad(
      labels, ((0, new_num_trials - num_trials), (0, 0)), constant_values=np.nan
  )
  return types.PaddedArray(
      padded_array=new_labels, is_missing=[trials_is_missing]
  )


def pad_features_and_labels(
    features: np.ndarray,
    labels: np.ndarray,
    padding_schedule: PaddingSchedule,
) -> Tuple[types.PaddedArray, types.PaddedArray]:
  """Pads `features` and `labels` according to a `padding_schedule`.

  Args:
    features: `np.ndarray` of shape `[N, D]`.
    labels: `np.ndarray` of shape `[N, T]`.
    padding_schedule: `PaddingSchedule` namedtuple.

  Returns:
    A tuple of `PaddedArray`s representing:
    * padded_features
    * padded_labels
  """
  new_features = pad_features(features, padding_schedule)
  new_labels = pad_labels(labels, padding_schedule)
  return new_features, new_labels
