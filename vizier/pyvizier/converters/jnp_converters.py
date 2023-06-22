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

"""Converters that use JAX."""

from typing import Sequence, Tuple

import attr
import numpy as np
from vizier import pyvizier
from vizier._src.jax import types
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

  def to_features(self, trials: Sequence[pyvizier.Trial]) -> types.PaddedArray:
    """Returns the features array with dimension: (n_trials, n_features)."""
    # Pad up the features.
    features = self._impl.to_features(trials)
    return padding.pad_features(features, self._padding_schedule)

  def to_labels(self, trials: Sequence[pyvizier.Trial]) -> types.PaddedArray:
    """Returns the labels array with dimenion: (n_trials, n_metrics)."""
    # Pad up the labels.
    labels = self._impl.to_labels(trials)
    return padding.pad_labels(labels, self._padding_schedule)

  def to_xy(
      self, trials: Sequence[pyvizier.Trial]
  ) -> Tuple[types.PaddedArray, types.PaddedArray]:
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


@attr.define
class TrialToContinuousAndCategoricalConverter:
  """EXPERIMENTAL: A converter that returns `ContinuousAndCategoricalArray`s.

  Unlike TrialtoNumpyDict converters, `to_features` returns a structure of
  floating and integer data. CATEGORICAL and DISCRETE parameters are encoded as
  integers and not one-hot embedded.

  IMPORTANT: Use a factory method (currently, there is one: `from_study_config`)
  instead of `__init__`.

  WARNING: This class is not exchangable with `TrialToNumpyDict`, and does not
  have functions that return shape or metric informations. Use it at your own
  risk.
  """

  _impl: core.DefaultTrialConverter

  def to_features(self, trials) -> types.ContinuousAndCategoricalArray:
    """Returns a structure of arrays with first dimension `n_trials`."""
    features = self._impl.to_features(trials)

    continuous = []
    categorical = []
    for converter, v in zip(self._impl.parameter_converters, features.values()):
      spec = converter.output_spec
      if spec.type == core.NumpyArraySpecType.CONTINUOUS:
        continuous.append(v)
      elif spec.type == core.NumpyArraySpecType.DISCRETE:
        categorical.append(v)
      else:
        raise ValueError(
            f'Expected spec to be CONTINUOUS or DISCRETE, saw {spec.type}'
        )
    if continuous:
      continuous_array = np.concatenate(continuous, axis=-1)
    else:
      continuous_array = np.zeros([len(trials), 0], dtype=np.float64)
    if categorical:
      categorical_array = np.concatenate(categorical, axis=-1)
    else:
      categorical_array = np.zeros([len(trials), 0], dtype=np.int32)
    return types.ContinuousAndCategoricalArray(
        continuous_array, categorical_array
    )

  @property
  def dtype(self) -> types.ContinuousAndCategorical[np.dtype]:
    empty_features = self.to_features([])
    return types.ContinuousAndCategorical(
        empty_features.continuous.dtype, empty_features.categorical.dtype
    )

  def to_labels(self, trials: Sequence[pyvizier.Trial]) -> np.ndarray:
    """Returns the labels array with dimenion: (n_trials, n_metrics)."""
    return core.dict_to_array(self._impl.to_labels(trials))

  def to_xy(
      self, trials
  ) -> Tuple[types.ContinuousAndCategoricalArray, np.ndarray]:
    return self.to_features(trials), self.to_labels(trials)

  def to_parameters(
      self, feat: types.ContinuousAndCategoricalArray
  ) -> Sequence[pyvizier.ParameterDict]:
    """Convert to nearest feasible parameter value. NaNs are preserved."""
    feat_dict = {}
    cont_ind = 0
    cat_ind = 0
    for converter in self._impl.parameter_converters:
      spec = converter.output_spec
      if spec.type == core.NumpyArraySpecType.CONTINUOUS:
        feat_dict[spec.name] = feat.continuous[:, cont_ind : cont_ind + 1]
        cont_ind += 1
      elif spec.type == core.NumpyArraySpecType.DISCRETE:
        feat_dict[spec.name] = feat.categorical[:, cat_ind : cat_ind + 1]
        cat_ind += 1
      else:
        raise ValueError(
            f'Expected spec to be CONTINUOUS or DISCRETE, saw {spec.type}'
        )

    return self._impl.to_parameters(core.DictOf2DArrays(feat_dict))

  @classmethod
  def from_study_config(
      cls,
      study_config: pyvizier.ProblemStatement,
      *,
      scale: bool = True,
      max_discrete_indices: int = 0,
      flip_sign_for_minimization_metrics: bool = True,
      dtype=np.float64,
  ) -> 'TrialToContinuousAndCategoricalConverter':
    """From study config.

    Args:
      study_config:
      scale: If True, scales the parameters to [0, 1] range.
      max_discrete_indices: For DISCRETE and INTEGER parameters that have more
        than this many feasible values will be continuified. When generating
        suggestions, values are rounded to the nearest feasible value. Note this
        default is different from the default in DefaultModelInputConverter.
      flip_sign_for_minimization_metrics: If True, flips the metric signs so
        that every metric maximizes.
      dtype: dtype

    Returns:
      TrialToContinuousAndCategoricalConverter
    """

    def create_input_converter(parameter):
      return core.DefaultModelInputConverter(
          parameter,
          scale=scale,
          max_discrete_indices=max_discrete_indices,
          onehot_embed=False,
          float_dtype=dtype,
      )

    def create_output_converter(metric):
      return core.DefaultModelOutputConverter(
          metric,
          flip_sign_for_minimization_metrics=flip_sign_for_minimization_metrics,
          dtype=dtype,
      )

    sc = study_config  # alias, to keep pylint quiet in the next line.
    converter = core.DefaultTrialConverter(
        [create_input_converter(p) for p in sc.search_space.parameters],
        [create_output_converter(m) for m in sc.metric_information],
    )
    return cls(converter)

  @property
  def output_specs(
      self,
  ) -> types.ContinuousAndCategorical[Sequence[core.NumpyArraySpec]]:
    """Output `NumpyArraySpec`s."""
    continuous = []
    categorical = []
    for converter in self._impl.parameter_converters:
      spec = converter.output_spec
      if spec.type == core.NumpyArraySpecType.CONTINUOUS:
        continuous.append(spec)
      elif spec.type == core.NumpyArraySpecType.DISCRETE:
        categorical.append(spec)
      else:
        raise ValueError(
            f'Expected spec to be CONTINUOUS or DISCRETE, saw {spec.type}'
        )
    return types.ContinuousAndCategorical(continuous, categorical)

  @property
  def metric_specs(self) -> Sequence[pyvizier.MetricInformation]:
    return [mc.metric_information for mc in self._impl.metric_converters]
