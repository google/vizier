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

"""Converter for padding trials."""

from typing import Sequence, Tuple

import numpy as np

from vizier import pyvizier
from vizier.pyvizier.converters import core
from vizier.pyvizier.converters import padding


class PaddedTrialToArrayConverter:
  """Converts trials to arrays and pads / masks them."""

  _experimental_override = 'I am aware that this code may break at any point.'

  def __init__(
      self,
      impl: core.TrialToArrayConverter,
      experimental_override: str = '',
      *,
      padding_schedule: padding.PaddingSchedule = padding.PaddingSchedule(
          num_trials=padding.PaddingType.NONE,
          num_features=padding.PaddingType.NONE,
      ),
  ):
    """SHOULD NOT BE USED! Use factory classmethods e.g. from_study_config."""

    if experimental_override != self._experimental_override:
      raise ValueError(
          'Set "experimental_override" if you want to call __init__ directly. '
          'Otherwise, use TrialToArrayConverter.from_study_config.'
      )
    self._impl = impl
    self._padding_schedule = padding_schedule

  def to_features(
      self, trials: Sequence[pyvizier.Trial]
  ) -> padding.PaddedArray:
    """Returns the features array with dimension: (n_trials, n_features)."""
    # Pad up the features.
    features = self._impl.to_features(trials)
    return padding.pad_features(features, self._padding_schedule)

  def to_labels(self, trials: Sequence[pyvizier.Trial]) -> padding.PaddedArray:
    """Returns the labels array with dimenion: (n_trials, n_metrics)."""
    # Pad up the labels.
    labels = self._impl.to_labels(trials)
    return padding.pad_labels(labels, self._padding_schedule)

  def to_xy(
      self, trials: Sequence[pyvizier.Trial]
  ) -> Tuple[padding.PaddedArray, padding.PaddedArray]:
    return self.to_features(trials), self.to_labels(trials)

  def to_parameters(self, arr: np.ndarray) -> Sequence[pyvizier.ParameterDict]:
    """Convert to nearest feasible parameter value. NaNs are preserved."""
    # Undo the padding of the parameters in this array.
    if self._padding_schedule is not None:
      # Get the original shape for the features without padding and
      # remove them.
      n_features = sum([spec.num_dimensions for spec in self.output_specs])
      arr = arr[..., :n_features]
    return self._impl.to_parameters(arr)

  @classmethod
  def from_study_config(
      cls,
      study_config: pyvizier.ProblemStatement,
      *,
      scale: bool = True,
      padding_schedule: padding.PaddingSchedule = padding.PaddingSchedule(
          num_trials=padding.PaddingType.NONE,
          num_features=padding.PaddingType.NONE,
      ),
      pad_oovs: bool = True,
      max_discrete_indices: int = 0,
      flip_sign_for_minimization_metrics: bool = True,
      dtype=np.float64,
  ) -> 'PaddedTrialToArrayConverter':
    """From study config.

    Args:
      study_config:
      scale: If True, scales the parameters to [0, 1] range.
      padding_schedule: Pads features and labels according to a padding
        schedule. This is to reduce the number of shapes a designer may see, and
        thus reduce JIT retracing.
      pad_oovs: If True, add an extra dimension for out-of-vocabulary values for
        non-CONTINUOUS parameters.
      max_discrete_indices: For DISCRETE and INTEGER parameters that have more
        than this many feasible values will be continuified. When generating
        suggestions, values are rounded to the nearest feasible value. Note this
        default is different from the default in DefaultModelInputConverter.
      flip_sign_for_minimization_metrics: If True, flips the metric signs so
        that every metric maximizes.
      dtype: dtype

    Returns:
      PaddedTrialToArrayConverter
    """
    converter = core.TrialToArrayConverter.from_study_config(
        study_config,
        scale=scale,
        pad_oovs=pad_oovs,
        max_discrete_indices=max_discrete_indices,
        flip_sign_for_minimization_metrics=flip_sign_for_minimization_metrics,
        dtype=dtype,
    )
    return cls(
        converter, cls._experimental_override, padding_schedule=padding_schedule
    )

  @property
  def output_specs(self) -> Sequence[core.NumpyArraySpec]:
    return self._impl.output_specs

  @property
  def metric_specs(self) -> Sequence[pyvizier.MetricInformation]:
    return self._impl.metric_specs

  @property
  def padding_schedule(self) -> padding.PaddingSchedule:
    return self._padding_schedule
