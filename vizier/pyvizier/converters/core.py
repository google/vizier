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

"""Abstractions and default converters."""

import abc
import copy
import dataclasses
import enum
from typing import Any, Callable, Collection, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Type, Union

from absl import logging
import attr
import numpy as np
from vizier import pyvizier

# The study identifier for cross-study learning must be stored in
# Trial.Metadata and StudyConfig.Metadata with this key.
# TODO: Use metadata namespace instead.
STUDY_ID_FIELD = 'metalearn_study_id'


class NumpyArraySpecType(enum.Enum):
  """Type information for NumpyArraySpec.

  CONTINUOUS: Continuous parameter
  DISCRETE: Discrete/integer/categorical parameter
  ONEHOT_EMBEDDING: One-hot embedding of DISCRETE.
  OBJECT: Object type that can hold any value.
  """

  CONTINUOUS = 'CONTINUOUS'
  DISCRETE = 'DISCRETE'
  ONEHOT_EMBEDDING = 'ONEHOT_EMBEDDING'
  OBJECT = 'OBJECT'

  @classmethod
  def default_factory(
      cls, pc: pyvizier.ParameterConfig
  ) -> 'NumpyArraySpecType':
    """SpecType when encoding discretes as integer indices."""
    if pc.type == pyvizier.ParameterType.DOUBLE:
      return NumpyArraySpecType.CONTINUOUS
    elif pc.type in (
        pyvizier.ParameterType.DISCRETE,
        pyvizier.ParameterType.CATEGORICAL,
        pyvizier.ParameterType.INTEGER,
    ):
      return NumpyArraySpecType.DISCRETE
    elif pc.type == pyvizier.ParameterType.CUSTOM:
      return NumpyArraySpecType.OBJECT
    raise ValueError(f'Unknown type {pc.type}')

  @classmethod
  def embedding_factory(
      cls, pc: pyvizier.ParameterConfig
  ) -> 'NumpyArraySpecType':
    """SpecType when encoding discretes as onehot embedding."""
    if pc.type == pyvizier.ParameterType.DOUBLE:
      return NumpyArraySpecType.CONTINUOUS
    elif pc.type in (
        pyvizier.ParameterType.DISCRETE,
        pyvizier.ParameterType.CATEGORICAL,
        pyvizier.ParameterType.INTEGER,
    ):
      return NumpyArraySpecType.ONEHOT_EMBEDDING
    raise ValueError(f'Unknown type {pc.type}')


@attr.define(frozen=True, auto_attribs=True, eq=True, hash=True)
class NumpyArraySpec:
  """Encodes what a feature array represents.

  This class can be seen as a counterpart of Vizier ParameterConfig.
  Vizier ParameterConfigs describe Trial parameters. NumpyArraySpec describes
  numpy arrays returned from trial-to-numpy converters.

  This class is similar to `BoundedTensorSpec` in tf agents, except it carries
  extra information specific to vizier.

  If `type` is `DOUBLE`, then `dtype` is a floating type, and bounds are
  floating numbers. num_dimensions is always 1, and num_oovs is zero.

  If 'type' is `DISCRETE`, then `dtype` is an integer type, and bounds are
  integers. num_dimensions is always 1. Suppose `bounds=(x,y)`. Then integers
  x to-and-including (y-num_oovs) correspond to valid parameter values. The rest
  represent out-of-vocabulary values. For example, an integer parameter in
  the range 1 <= x <= 3 can be represented by a DISCRETE NumpyArraySpec with
  bounds=(1,4) and oov=1.

  If 'type' is `ONEHOT_EMBEDDING`, then `dtype` is a floating type, and bounds
  are floating numbers. In this case, the output array can have multiple
  columns.

  Attributes:
    type: Underlying type of the Vizier parameter corresponding to the array.
    dtype: The numpy array's type.
    bounds: Always inclusive in both directions.
    num_dimensions: Corresponds to shape[-1] of the numpy array. When `type` is
      `ONEHOT_EMBEDDING`, the first X dimensions correspond to valid parameter
      values. The other dimensions correspond to out-of-vocabulary values.
    name: Parameter name.
    num_oovs: Number of out-of-vocabulary items, for non-continuous type.
    scale: Scaling of the values.
  """

  type: NumpyArraySpecType = attr.field(
      validator=attr.validators.instance_of(NumpyArraySpecType)
  )
  dtype: np.dtype = attr.field(
      converter=np.dtype,
      validator=attr.validators.in_(
          [np.float32, np.int32, np.float64, np.int64, np.object_]
      ),
  )
  bounds: Union[Tuple[float, float], Tuple[int, int]]
  num_dimensions: int = attr.field(validator=attr.validators.instance_of(int))
  name: str = attr.field(validator=attr.validators.instance_of(str))
  num_oovs: int = attr.field(validator=attr.validators.instance_of(int))
  scale: Optional[pyvizier.ScaleType] = attr.field(
      default=None,
      validator=attr.validators.optional(
          attr.validators.instance_of(pyvizier.ScaleType)
      ),
  )

  def __attrs_post_init__(self):
    object.__setattr__(
        self, 'bounds', tuple(np.array(self.bounds, dtype=self.dtype))
    )

  @classmethod
  def from_parameter_config(
      cls,
      pc: pyvizier.ParameterConfig,
      type_factory: Callable[
          [pyvizier.ParameterConfig], NumpyArraySpecType
      ] = NumpyArraySpecType.default_factory,
      floating_dtype: Union[np.dtype, Type[np.floating]] = np.float32,
      int_dtype: Union[np.dtype, Type[np.integer]] = np.int32,
      *,
      pad_oovs: bool = True,
  ) -> 'NumpyArraySpec':
    """Factory function.

    Args:
      pc:
      type_factory: NumpyArraySpecType has `default_factory` and
        `embedding_factory`. The difference is in how they handle non-continuous
        parameters.
      floating_dtype: Dtype of the floating outputs.
      int_dtype: Dtype of the integer outputs.
      pad_oovs: If True, pad the out-of-vocabulary dimensions to onehot
        embedding.

    Returns:
      NumpyArraySpec.
    """
    the_type = type_factory(pc)
    if the_type == NumpyArraySpecType.CONTINUOUS:
      return NumpyArraySpec(
          the_type,
          np.dtype(floating_dtype),
          bounds=pc.bounds,
          num_dimensions=1,
          scale=pc.scale_type,
          name=pc.name,
          num_oovs=0,
      )
    elif the_type == NumpyArraySpecType.DISCRETE:
      return NumpyArraySpec(
          the_type,
          np.dtype(int_dtype),
          bounds=(0, len(pc.feasible_values)),
          num_dimensions=1,
          name=pc.name,
          num_oovs=1 if pad_oovs else 0,
      )
    elif the_type == NumpyArraySpecType.ONEHOT_EMBEDDING:
      return NumpyArraySpec(
          the_type,
          np.dtype(floating_dtype),
          bounds=(0.0, 1.0),
          num_dimensions=len(pc.feasible_values) + 1,
          name=pc.name,
          num_oovs=1 if pad_oovs else 0,
      )
    elif the_type == NumpyArraySpecType.OBJECT:
      return NumpyArraySpec(
          the_type,
          dtype=np.object_,
          bounds=(0, 0),
          num_dimensions=0,
          name=pc.name,
          num_oovs=0,
      )
    raise ValueError(f'Unknown type {type}')


def dict_to_array(array_dict: Mapping[Any, np.ndarray]) -> np.ndarray:
  r"""Converts a dict of (..., D_i) arrays to a (..., \sum_i D_i) array."""
  return np.concatenate(list(array_dict.values()), axis=-1)


class DictOf2DArrays(Mapping[str, np.ndarray]):
  """Dictionary of string to 2D arrays.

  All arrays share the first dimension, which is at a high level, the number of
  objects that this dictionary corresponds to.

  Attributes:
    size: Array's shape[0].
  """

  def __init__(self, d: Mapping[str, np.ndarray]):
    self._d = d
    shape = None
    for k, v in self.items():
      if shape is None:
        shape = v.shape
        if len(shape) != 2:
          raise ValueError(
              f'{k} has shape {v.shape} which is not length 2.'
              'DictOf2DArrays only supports 2D numpy arrays.'
          )
      if shape[0] != v.shape[0]:
        raise ValueError(
            f'{k} has shape {v.shape} which is not equal to {shape}.'
        )
    self._size = shape[0]

  def __getitem__(self, key: str) -> np.ndarray:
    return self._d[key]

  def __iter__(self) -> Iterator[str]:
    return iter(self._d)

  def __len__(self) -> int:
    return len(self._d)

  def __add__(self, other: 'DictOf2DArrays') -> 'DictOf2DArrays':
    if not isinstance(other, DictOf2DArrays):
      raise ValueError('You can add DictOf2DArrays only.')
    if len(self) != len(other):
      # We don't check the keys because it's too slow.
      raise ValueError('Two arrays have different length!')
    return DictOf2DArrays(
        {k: np.concatenate([self[k], other[k]], axis=0) for k in self}
    )

  @property
  def size(self) -> int:
    return self._size

  def asarray(self) -> np.ndarray:
    return dict_to_array(self._d)

  def dict_like(self, array: np.ndarray) -> 'DictOf2DArrays':
    """[Experimental] Converts an array into a dict with the same keys as this.

    This function acts like an inverse of `asarray()`, i.e. it satisfies
      `self.dict_like(self.asarray()) == self`.

    Example:
      d = DictOf2DArrays({'p1': [[1], [2], [3]], 'p2': [[4], [5], [6]]})
      d.dict_like([[1, 2], [3, 4]]) == {'p1': [[1], [3]], 'p2': [[2],[4]]}

    Args:
      array:

    Returns:
      DictOf2DArrays with the same shape spec as this.
    """
    begin = 0
    new_dict = dict()
    for k, v in self.items():
      end = begin + v.shape[1]
      new_dict[k] = array[:, begin:end].astype(v.dtype)
      begin = end

    return DictOf2DArrays(new_dict)


class TrialToNumpyDict(abc.ABC):
  """Parses a sequence of Trials to a dict keyed by parameter and metric names.

  Design note:
    A typical Keras pipeline consists of:
      1. Load data into arrays.
      2. Call Model.build() to initialize a model for the loaded data shape.
      3. Call Model.fit() to train the model.
      4. Call Model.__call__() to predict with the model.
    This abstraction allows us to switch implementations for the step number 1.

  This class consists of `to_xy()` function and properties that describe
  the returned values of `to_xy()`. Subclasses are free to have extra functions
  for convenience.
  """

  @abc.abstractmethod
  def to_xy(
      self, trials: Sequence[pyvizier.Trial]
  ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Extracts features-labels pairs that can be used for fitting a model."""
    pass

  @property
  @abc.abstractmethod
  def features_shape(self) -> Dict[str, Sequence[Union[int, None]]]:
    """Describes the shape of the first element returned from to_xy().

    Returns:
      Dict of feature names to shapes. `None` means that the shape depends
      on the input "trials".
    # TODO: Add compute_feature_shape(trials) that returns exact
    shapes.
    """
    pass

  @property
  @abc.abstractmethod
  def output_specs(self) -> Dict[str, NumpyArraySpec]:
    """Describes the semantics of the first element returned from to_xy().

    See NumpyArraySpec for more details.

    Returns:
      Dict of feature names to NumpyArraySpecs.
    """
    pass

  @property
  @abc.abstractmethod
  def labels_shape(self) -> Dict[str, Any]:
    """Describes the shape of the second element returned from to_xy().

    Returns:
      Dict of label names to shapes. `None` means that the shape depends
      on the input "trials".
    """
    pass

  @property
  @abc.abstractmethod
  def metric_information(self) -> Dict[str, pyvizier.MetricInformation]:
    """Describes the semantics of the second element returned from to_xy().

    See ModelOutputConverter for more details.
    Returns:
      Dict of label names to MetricInformation.
    """


class ModelInputConverter(metaclass=abc.ABCMeta):
  """Interface for extracting inputs to the model."""

  @abc.abstractmethod
  def convert(self, trials: Sequence[pyvizier.TrialSuggestion]) -> np.ndarray:
    """Returns an array of shape (number of trials, feature dimension).

    Args:
      trials:

    Returns:
      Returns an array of shape (number of trials, feature dimension).
      Subclasses must use a fixed feature dimension. In particular, it should
      be a constant function of the input trials.
    """

  @property
  @abc.abstractmethod
  def output_spec(self) -> NumpyArraySpec:
    """Provides specification of the output from this converter."""

  @property
  @abc.abstractmethod
  def parameter_config(self):
    """Original ParameterConfig that this converter acts on."""

  @abc.abstractmethod
  def to_parameter_values(
      self, array: np.ndarray
  ) -> List[Optional[pyvizier.ParameterValue]]:
    """Convert and clip to the nearest feasible parameter values.

    NOTE: value is automatically truncated to lie inside the search space.
      This method should only be called to convert suggestions.

    Args:
      array: has shape (number of trials, feature dimension)

    Returns:
      List of (ParameterValue or None).  A list entry is None when this
      converter's parameter doesn't exist in the Trial.  (This is common
      when parameters are conditional.)
    """


@dataclasses.dataclass
class ModelInputArrayBijector:
  """Transformations on the numpy arrays generated by ModelInputConverter."""

  forward_fn: Callable[[np.ndarray], np.ndarray]
  backward_fn: Callable[[np.ndarray], np.ndarray]
  output_spec: NumpyArraySpec  # Spec after forward_fn is applied.

  @classmethod
  def identity(cls, spec: NumpyArraySpec) -> 'ModelInputArrayBijector':
    return cls(lambda x: x, lambda x: x, spec)

  @classmethod
  def scaler_from_spec(cls, spec: NumpyArraySpec) -> 'ModelInputArrayBijector':
    """For continuous specs, linearize and scale it to (0, 1) range."""
    low, high = spec.bounds
    if spec.type != NumpyArraySpecType.CONTINUOUS:
      return cls.identity(attr.evolve(spec, scale=None))
    if low == high:

      def backward_fn(y):
        return np.where(np.isfinite(y), y + low - 0.5, y)

      def forward_fn(y):
        return np.where(np.isfinite(y), y - low + 0.5, y)

      return cls(
          forward_fn,
          backward_fn,
          attr.evolve(spec, bounds=(0.5, 0.5), scale=None),
      )

    if spec.scale == pyvizier.ScaleType.LOG:
      if low < 0 or high < 0:
        raise ValueError(
            'Log scale requires both parameter boundaries to be positive,'
            f' though low bound is {low} and high bound is {high}.'
        )
      low, high = np.log(low), np.log(high)
      denom = (high - low) or 1.0
      if denom < 1e-6:
        logging.warning('Unusually small range detected for %s', spec)
      scale_fn = lambda x, low=low, denom=denom: (np.log(x) - low) / denom
      unscale_fn = lambda x, low=low, denom=denom: np.exp(x * denom + low)
    elif spec.scale == pyvizier.ScaleType.REVERSE_LOG:
      raw_sum = low + high
      low, high = np.log(low), np.log(high)
      denom = (high - low) or 1.0
      if denom < 1e-6:
        logging.warning('Unusually small range detected for %s', spec)

      def scale_fn(x, low=low, raw_sum=raw_sum, denom=denom):
        return 1.0 - (np.log(raw_sum - x) - low) / denom

      def unscale_fn(x, high=high, raw_sum=raw_sum):
        return raw_sum - np.exp(high - denom * x)

    else:
      if not (spec.scale == pyvizier.ScaleType.LINEAR or spec.scale is None):
        logging.warning('Unknown scale type %s. Applying LINEAR', spec.scale)
      denom = high - low
      if denom < 1e-6:
        logging.warning('Unusually small range detected for %s', spec)
      if denom == 1.0 and low == 0:
        return cls.identity(attr.evolve(spec, scale=None))
      scale_fn = lambda x, high=high, low=low: (x - low) / (high - low)
      unscale_fn = lambda x, high=high, low=low: x * (high - low) + low

    return cls(
        scale_fn, unscale_fn, attr.evolve(spec, bounds=(0.0, 1.0), scale=None)
    )

  @classmethod
  def onehot_embedder_from_spec(
      cls, spec: NumpyArraySpec, *, dtype=np.float32, pad_oovs: bool = True
  ) -> 'ModelInputArrayBijector':
    """Given a discrete spec, one-hot embeds it."""
    if spec.type != NumpyArraySpecType.DISCRETE:
      return cls.identity(spec)

    num_oovs = 1 if pad_oovs else 0
    output_spec = NumpyArraySpec(
        NumpyArraySpecType.ONEHOT_EMBEDDING,
        dtype,
        bounds=(0.0, 1.0),
        num_dimensions=int(spec.bounds[1] - spec.bounds[0] + num_oovs),
        name=spec.name,
        num_oovs=num_oovs,
        scale=None,
    )

    def embed_fn(x: np.ndarray, output_spec=output_spec):
      """x is integer array of [N, 1]."""
      return np.eye(output_spec.num_dimensions, dtype=output_spec.dtype)[
          x.flatten()
      ]

    def unembed_fn(x: np.ndarray, spec=spec, output_spec=output_spec):
      return np.argmax(
          x[:, : output_spec.num_dimensions - output_spec.num_oovs], axis=1
      ).astype(spec.dtype)

    return cls(embed_fn, unembed_fn, output_spec)


def _create_default_getter(
    pconfig: pyvizier.ParameterConfig,
) -> Callable[[pyvizier.TrialSuggestion], Any]:
  """Create a default getter for the given parameter config."""

  def getter(trial, pconfig=pconfig):
    if pconfig.name not in trial.parameters:
      return None

    pvalue = trial.parameters[pconfig.name]
    if pconfig.type == pyvizier.ParameterType.DOUBLE:
      return pvalue.as_float
    elif pconfig.type == pyvizier.ParameterType.DISCRETE:
      return pvalue.as_float
    elif pconfig.type == pyvizier.ParameterType.INTEGER:
      return pvalue.as_int
    else:
      return pvalue.as_str

  return getter


class DefaultModelInputConverter(ModelInputConverter):
  """Converts trials into a (None, 1) array corresponding to a parameter.

  If the parameter_config is continuous, values obtained from `getter()` are
  directly returned as floating numbers. Otherwise, this converter returns
  the index of the value obtained from `getter()` within
  `parameter_config.feasible_points` as int32.
  """

  def __init__(
      self,
      parameter_config: pyvizier.ParameterConfig,
      getter: Optional[Callable[[pyvizier.TrialSuggestion], Any]] = None,
      *,
      float_dtype: Union[np.dtype, Type[np.floating]] = np.float32,
      max_discrete_indices: int = 10,
      scale: bool = False,
      onehot_embed: bool = False,
      converts_to_parameter: bool = True,
      pad_oovs: bool = True,
      should_clip: bool = True,
  ):
    """Init.

    Given B trials, convert() always converts to (B, 1) array. The returned
    array may contain NaNs.

    Args:
      parameter_config:
      getter: See class pydoc. If the getter is not specified, the default
        getter looks up `parameter_config.name` inside `Trial.parameters`.
      float_dtype: floating precision to be used.
      max_discrete_indices: If the parameter config has more than this many
        number of DISCRETE/INTEGER feasible points, then the parameter config is
        continuified first.
      scale:
      onehot_embed:
      converts_to_parameter: If False, this converter does not correspond to an
        actual parameter in Vizier search space, and `to_parameter_value` always
        returns None
      pad_oovs: If True, pad the out-of-vocabulary dimensions to onehot
        embedding.
      should_clip: If True, clipping should be applied to parameter values.
    """
    self._converts_to_parameter = converts_to_parameter
    self._parameter_config = copy.deepcopy(parameter_config)
    if (
        parameter_config.type
        in (pyvizier.ParameterType.INTEGER, pyvizier.ParameterType.DISCRETE)
        and parameter_config.num_feasible_values > max_discrete_indices
    ):
      parameter_config = parameter_config.continuify()

    # TODO: Make the default getter raise an Error if they encounter an
    # out-of-vocabulary value but pad_oovs is False.
    self._getter = getter or _create_default_getter(parameter_config)
    # Getter spec can only have DISCRETE or CONTINUOUS types.
    self._getter_spec = NumpyArraySpec.from_parameter_config(
        parameter_config,
        NumpyArraySpecType.default_factory,
        floating_dtype=float_dtype,
    )

    # Optionally scale and onehot embed.
    spec = self._getter_spec
    self.scaler = (
        ModelInputArrayBijector.scaler_from_spec(spec)
        if scale
        else ModelInputArrayBijector.identity(spec)
    )
    spec = self.scaler.output_spec
    if onehot_embed:
      self.onehot_encoder = ModelInputArrayBijector.onehot_embedder_from_spec(
          spec, dtype=float_dtype, pad_oovs=pad_oovs
      )
    else:
      self.onehot_encoder = ModelInputArrayBijector.identity(spec)

    spec = self.onehot_encoder.output_spec
    self._output_spec = spec
    self._should_clip = should_clip

  def convert(self, trials: Sequence[pyvizier.TrialSuggestion]) -> np.ndarray:
    """Returns an array of shape [len(trials), output_spec.num_dimensions].

    Args:
      trials:

    Returns:
      For each `trial`, if `self.getter(trial)` returns `None`, we _impute_ the
      value; otherwise, we _extract_ the value.
      If `self.parameter_config.type` is `DOUBLE`, then
        * EXTRACT: Directly use the getter's return value as float.
        * IMPUTE: Return `nan`.
      Otherwise,
        * EXTRACT: Returns the integer index of the getter's return value within
          feasible values.
        * IMPUTE: Returns `len(feasible_values)`.
    """
    if not trials:
      return np.zeros(
          [0, self.output_spec.num_dimensions], dtype=self.output_spec.dtype
      )

    value_converter = (
        self._convert_index
        if self._getter_spec.type == NumpyArraySpecType.DISCRETE
        else self._convert_continuous
    )
    values = [value_converter(t) for t in trials]
    array = np.asarray(values, dtype=self._getter_spec.dtype).reshape([-1, 1])
    return self.onehot_encoder.forward_fn(self.scaler.forward_fn(array))

  def _to_parameter_value(
      self, value: Union['np.float', float, int]
  ) -> Optional[pyvizier.ParameterValue]:
    """Converts to a single parameter value; see to_parameter_values().

    Be aware that the value is automatically truncated.

    Args:
      value:

    Returns:
      ParameterValue.
    """
    if not self._converts_to_parameter:
      return None
    elif not np.isfinite(value):
      return None
    elif self.parameter_config.type == pyvizier.ParameterType.DOUBLE:
      # Input parameter was DOUBLE. Output is also DOUBLE.
      if self._should_clip:
        value = np.clip(
            value,
            self._parameter_config.bounds[0],
            self._parameter_config.bounds[1],
        )
      return pyvizier.ParameterValue(float(value))
    elif self.output_spec.type == NumpyArraySpecType.CONTINUOUS:
      # The parameter config is originally discrete, but continuified.
      # Round to the closest number.
      diffs = np.abs(
          np.asarray(self.parameter_config.feasible_values, dtype=self.dtype)
          - value
      )

      idx = np.argmin(diffs)
      closest_number = pyvizier.ParameterValue(
          self.parameter_config.feasible_values[idx]
      )
      return pyvizier.ParameterValue(
          closest_number.cast_as_internal(self.parameter_config.type)
      )

    elif value >= len(self.parameter_config.feasible_values):
      return None
    else:
      return pyvizier.ParameterValue(
          self.parameter_config.feasible_values[value]
      )

  def to_parameter_values(
      self, array: np.ndarray
  ) -> List[Optional[pyvizier.ParameterValue]]:
    """Convert and clip to the nearest feasible parameter values."""
    array = self.scaler.backward_fn(self.onehot_encoder.backward_fn(array))
    return [self._to_parameter_value(v) for v in list(array.flatten())]

  def _convert_index(self, trial: pyvizier.TrialSuggestion):
    """Called by `convert()` if configured for a non-continuous parameter."""
    raw_value = self._getter(trial)
    if raw_value in self.parameter_config.feasible_values:
      return self.parameter_config.feasible_values.index(raw_value)
    else:
      # Return the catch-all missing index.
      return len(self.parameter_config.feasible_values)

  def _convert_continuous(self, trial: pyvizier.TrialSuggestion):
    """Called by `convert()` if configured for a continuous parameter."""
    raw_value = self._getter(trial)
    if raw_value is None:
      return np.nan
    else:
      return raw_value

  @property
  def dtype(self):
    """dtype of the array returned by convert()."""
    return self.output_spec.dtype

  @property
  def output_spec(self) -> NumpyArraySpec:
    return self._output_spec

  @property
  def parameter_config(self) -> pyvizier.ParameterConfig:
    """Original parameter config that this converter is defined on."""
    return self._parameter_config


class ModelOutputConverter(metaclass=abc.ABCMeta):
  """Metric converter interface.

  Unlike ModelInputConverter, this class is currently designed to always
  output a single-dimensional metric. This is because in standard Keras
  workflows, output shapes must be known when defining a model. (In contrast,
  input shapes are required at the time of _building_ a model.)
  """

  @abc.abstractmethod
  def convert(self, measurements: Sequence[pyvizier.Measurement]) -> np.ndarray:
    """Returns N x 1 array."""
    pass

  @abc.abstractmethod
  def to_metrics(
      self, labels: np.ndarray
  ) -> Sequence[Optional[pyvizier.Metric]]:
    """Returns a list of pyvizier metrics.

    The metrics are converted from an array of labels with shape (len(metrics),)
    or (len(metrics), 1) and nan values in the labels are translated to None.

    Args:
      labels: (len(metrics),) or (len(metrics), 1) shaped array of labels.

    Returns:
      A list of pyvizier metrics created with `labels`.
    """

  @property
  @abc.abstractmethod
  def metric_information(self) -> pyvizier.MetricInformation:
    """Describes the semantics of the return value from convert() method.

    Returns:
      The MetricInformation that reflects how the labels should be
      interpreted. It may not be identical to the MetricInformation that the
      converter was created from. For example, if the converter flips the signs
      or changes the semantics of safety configs, then the returned
      MetricInformation should reflect such changes.
    """

  @property
  def output_shape(self) -> Tuple[None, int]:
    return (None, 1)


class DefaultModelOutputConverter(ModelOutputConverter):
  """Converts measurements into numpy arrays."""

  def __init__(
      self,
      metric_information: pyvizier.MetricInformation,
      *,
      flip_sign_for_minimization_metrics: bool = False,
      shift_safe_metrics: bool = True,
      dtype: Union[
          Type[float], Type[int], Type[np.generic], np.dtype
      ] = np.float32,
      raise_errors_for_missing_metrics: bool = False,
  ):
    """Init.

    Args:
      metric_information:
      flip_sign_for_minimization_metrics: Flips the sign if the metric is to
        minimize.
      shift_safe_metrics: If True, subtract the safety threshold from the metric
        value. i.e. center the metric so that the safety threshold is 0. For a
        minimization safety metric if flip_sign_for_minimization_metrics is
        true, subtraction happens begore flipping the sign.
      dtype:
      raise_errors_for_missing_metrics: If True, raise errors. Otherwise, fill
        with NaN.
    """

    self._original_metric_information = metric_information
    self.flip_sign_for_minimization_metrics = flip_sign_for_minimization_metrics
    self.raise_errors_for_missing_metrics = raise_errors_for_missing_metrics
    self.dtype = dtype
    self.shift_safe_metrics = shift_safe_metrics

  @property
  def _should_flip_sign(self) -> bool:
    return (
        self._original_metric_information.goal
        == pyvizier.ObjectiveMetricGoal.MINIMIZE
        and self.flip_sign_for_minimization_metrics
    )

  def convert(
      self, measurements: Sequence[Optional[pyvizier.Measurement]]
  ) -> np.ndarray:
    """Returns a (len(measurements), 1) array."""
    if not measurements:
      return np.zeros([0, 1], dtype=self.dtype)

    all_metrics = [m.metrics if m is not None else dict() for m in measurements]
    if not self.raise_errors_for_missing_metrics:
      metricvalues = [
          metrics.get(self._original_metric_information.name, None)
          for metrics in all_metrics
      ]
      labels = [mv.value if mv else np.nan for mv in metricvalues]
    else:
      labels = [
          metrics[self._original_metric_information.name].value
          for metrics in all_metrics
      ]
    labels = np.asarray(labels, dtype=self.dtype)[:, np.newaxis]
    if (
        self.shift_safe_metrics
        and self._original_metric_information.type.is_safety
    ):
      labels -= self._original_metric_information.safety_threshold
    return labels * (-1 if self._should_flip_sign else 1)

  def to_metrics(
      self, labels: np.ndarray
  ) -> Sequence[Optional[pyvizier.Metric]]:
    """Converts an array of labels to pyvizier Metrics.

    Args:
      labels: (len(metrics),) or (len(metrics), 1) shaped array of labels.

    Returns:
      metrics: a list of pyvizier metrics.
    """
    if labels.ndim == 1:
      labels = labels[:, None]
    if labels.shape[1] > 1 or len(labels.shape) > 2:
      raise ValueError('The input array must be of shape (num,) or (num, 1).')

    labels = labels.flatten()
    if (
        self.shift_safe_metrics
        and self._original_metric_information.type.is_safety
    ):
      labels -= self._original_metric_information.safety_threshold

    labels = labels * (-1 if self._should_flip_sign else 1)
    metrics = [
        pyvizier.Metric(value=l) if np.isfinite(l) else None for l in labels
    ]
    return metrics

  @property
  def metric_information(self) -> pyvizier.MetricInformation:
    """Returns a copy that reflects how the converter treats the metric."""
    metric_information = copy.deepcopy(self._original_metric_information)
    if self.shift_safe_metrics and metric_information.type.is_safety:
      metric_information.safety_threshold = 0.0
    if self._should_flip_sign:
      metric_information = metric_information.flip_goal()
    return metric_information


class DefaultTrialConverter(TrialToNumpyDict):
  """Combines parameter and metric converters.

  Always takes the final measurement and convert to metrics.

  WARNING: This class is not responsible for filtering out non-completed or
  infeasible trials. Labels may contain NaN in such cases.

  For standard use cases, use one of the factory functions. For more
  flexibility, inject parameter_converters and metric_converters on your own.
  """

  def __init__(
      self,
      parameter_converters: Collection[ModelInputConverter],
      metric_converters: Collection[ModelOutputConverter] = tuple(),
  ):
    self.parameter_converters = list(parameter_converters)
    self.parameter_converters_dict = {
        pc.parameter_config.name: pc for pc in self.parameter_converters
    }
    self.metric_converters = list(metric_converters)
    self._metric_converters_dict = {
        mc.metric_information.name: mc for mc in self.metric_converters
    }

  def to_features(
      self, trials: Sequence[pyvizier.TrialSuggestion]
  ) -> Dict[str, np.ndarray]:
    """Shorthand for to_xy(trials))[0]."""
    result_dict = dict()
    for converter in self.parameter_converters:
      result_dict[converter.parameter_config.name] = converter.convert(trials)
    return result_dict

  def to_trials(
      self,
      features: Mapping[str, np.ndarray],
      labels: Optional[Mapping[str, np.ndarray]] = None,
  ) -> List[pyvizier.Trial]:
    """Inverse of `to_features` and optionally the inverse of `to_labels`.

    We assume that label values are already normalized and their signs are
    flipped if required.

    Args:
      features: A dictionary where keys correspond to parameter names in the
        returned Trial and values correspond to parameter values and have shape
        (num_obs, 1).
      labels: A dictionary of labels where each key corresponds to a metric name
        and its value is an array of shape (num_obs, 1) of observed metric
        values.

    Returns:
      A list of pyvizier trials created with parameters corresponding to
      `features` and final measurements corresponding to `labels`.
      NOTE: All final measurements have steps=1. If 'labels' is not passed, the
      output trials include `final_measurement=None` and `measurements=[]`.
    """
    invalid_features = [
        (k, v.shape)
        for k, v in features.items()
        if len(v.shape) != 2 or (len(v.shape) == 2 and v.shape[1] != 1)
    ]
    if invalid_features:
      raise ValueError(
          'Features need to contain 2d arrays with shape (num_obs, 1). Invalid'
          f' feature shapes: {invalid_features}'
      )

    if labels:
      invalid_labels = [
          (k, v.shape)
          for k, v in labels.items()
          if len(v.shape) != 2 or (len(v.shape) == 2 and v.shape[1] != 1)
      ]
      if invalid_labels:
        raise ValueError(
            'Labels need to contain 2d arrays with shape (num_obs, 1). Invalid'
            f' label shapes: {invalid_labels}'
        )

    if labels is None:
      return [
          pyvizier.Trial(parameters=p) for p in self.to_parameters(features)
      ]

    try:
      labels = DictOf2DArrays(labels)
    except ValueError as e:
      raise ValueError('Please check the shape of "labels"') from e

    try:
      features = DictOf2DArrays(features)
    except ValueError as e:
      raise ValueError('Please check the shape of "features"') from e

    if labels.size != features.size:
      raise ValueError(
          'The number of features and labels observations do not match.'
      )
    parameters = self.to_parameters(features)
    measurements = self._to_measurements(labels)
    for m in measurements:
      m.steps = 1

    trials = []
    for p, m in zip(parameters, measurements):
      trial = pyvizier.Trial(parameters=p, final_measurement=m)
      # _to_measurements returns an empty dict for NaN and non-finite metric
      # values.
      if not m.metrics:
        trial.complete(m, infeasibility_reason='Metrics are empty')
      trials.append(trial)
    return trials

  def _to_measurements(
      self, labels: Mapping[str, np.ndarray]
  ) -> List[pyvizier.Measurement]:
    """Converts a dictionary of labels into a list of pyvizier measurements.

    Each key in the dictionary corresponds to a metric and the length of the
    list of pyvizier measurements equals to the number of keys. Note that this
    method generates measurements without setting steps, hence steps are
    defaulted to zero. The caller should assign them later if desired.

    Args:
      labels: A dictionary of labels where each key corresponds to a metric name
        with dictionary values as metric values casted as an array of shape
        (num_obs,) or (num_obs, 1).

    Returns:
      A list of pyvizier measurements created with final measurements
        corresponding to `labels`.
    """
    try:
      DictOf2DArrays(labels)
    except ValueError as e:
      raise ValueError('Please check the shape of "labels"') from e
    measurements = [
        pyvizier.Measurement() for _ in range(DictOf2DArrays(labels).size)
    ]
    # Iterate through labels names and convert them.
    for metric_name, metric_values in labels.items():
      metric_converter = self._metric_converters_dict[metric_name]
      metrics_values = metric_converter.to_metrics(metric_values)
      for measurement_dict, value in zip(measurements, metrics_values):
        if value is not None:
          measurement_dict.metrics[metric_name] = value
    return measurements

  def to_parameters(
      self, features: Mapping[str, np.ndarray]
  ) -> List[pyvizier.ParameterDict]:
    """Convert to nearest feasible parameter value. NaNs trigger errors."""
    # TODO: NaNs should be ignored instead of triggering errors.

    # Validate features's shape, and create empty ParameterDicts.
    parameters = [
        pyvizier.ParameterDict() for _ in range(DictOf2DArrays(features).size)
    ]

    # Iterate through parameter names and convert them.
    for key, values in features.items():
      parameter_converter = self.parameter_converters_dict[key]
      parameter_values = parameter_converter.to_parameter_values(values)
      for param_dict, value in zip(parameters, parameter_values):
        if value is not None:
          param_dict[key] = value
    return parameters

  def to_labels(
      self, trials: Sequence[pyvizier.Trial]
  ) -> Dict[str, np.ndarray]:
    """Shorthand for to_xy(trials))[1]."""
    result_dict = dict()
    for converter in self.metric_converters:
      result_dict[converter.metric_information.name] = converter.convert(
          [t.final_measurement for t in trials]
      )
    return result_dict

  def to_labels_array(self, trials: Sequence[pyvizier.Trial]) -> np.ndarray:
    """Shorthand for dict_to_array(self.to_labels(trials))."""
    return dict_to_array(self.to_labels(trials))

  def to_xy(
      self, trials: Sequence[pyvizier.Trial]
  ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """See base class."""
    return self.to_features(trials), self.to_labels(trials)

  @property
  def features_shape(self) -> Dict[str, Tuple[Union[int, None], int]]:
    """See base class."""
    return {
        pc.output_spec.name: (None, pc.output_spec.num_dimensions)
        for pc in self.parameter_converters
    }

  @property
  def output_specs(self) -> Dict[str, NumpyArraySpec]:
    """See base class."""
    return {
        pc.output_spec.name: pc.output_spec for pc in self.parameter_converters
    }

  @property
  def labels_shape(self) -> Dict[str, Sequence[Union[int, None]]]:
    """See base class."""
    return {
        mc.metric_information.name: mc.output_shape
        for mc in self.metric_converters
    }

  @property
  def metric_information(self) -> Dict[str, pyvizier.MetricInformation]:
    """See base class."""
    return {
        mc.metric_information.name: mc.metric_information
        for mc in self.metric_converters
    }

  # TODO: Deprecate or update so that it returns SearchSpace.
  @property
  def parameter_configs(self) -> Dict[str, pyvizier.ParameterConfig]:
    """Returns a dict of the original Parameter configs."""
    return {
        converter.parameter_config.name: converter.parameter_config
        for converter in self.parameter_converters
    }

  @classmethod
  def from_study_configs(
      cls,
      study_configs: Sequence[pyvizier.ProblemStatement],
      metric_information: Collection[pyvizier.MetricInformation],
      *,
      use_study_id_feature: bool = True,
  ) -> 'DefaultTrialConverter':
    """Creates a converter from a list of study configs.

    Args:
      study_configs: StudyConfigs to be merged.
      metric_information: MetricInformation of metrics to be used as y-values.
      use_study_id_feature: If True, an extra parameter is added that
        corresponds to the STUDY_ID_FIELD inside metadata.

    Returns:
      `DefaultTrialConverter`.
    """
    # Merge parameter configs by name.
    merged_configs = list(
        pyvizier.SearchSpaceSelector([sc.search_space for sc in study_configs])
        .select_all()
        .merge()
    )

    merged_configs = {pc.name: pc for pc in merged_configs}
    parameter_converters = [
        DefaultModelInputConverter(pc) for pc in merged_configs.values()
    ]

    # Append study id feature if configured to do so.
    if use_study_id_feature:
      # Collect study ids.
      study_ids = set()
      for sc in study_configs:
        metalearn_study_id = sc.metadata.get(STUDY_ID_FIELD, None)
        if metalearn_study_id is None:
          continue
        study_ids.add(metalearn_study_id)

      # Validate.
      if not study_ids:
        logging.error(
            'use_study_id_feature was True, but none of the studies '
            'had study id configured.'
        )
        use_study_id_feature = False
      elif STUDY_ID_FIELD in merged_configs:
        raise ValueError(
            'Dataset name conflicts with a ParameterConfig '
            'that already exists: {}'.format(merged_configs[STUDY_ID_FIELD])
        )

      # Create new parameter config.
      parameter_config = pyvizier.ParameterConfig.factory(
          STUDY_ID_FIELD, feasible_values=list(study_ids)
      )
      merged_configs[STUDY_ID_FIELD] = parameter_config
      logging.info('Created a new ParameterConfig %s', parameter_config)

      # Create converter.
      parameter_converters.append(
          DefaultModelInputConverter(
              parameter_config,
              lambda t: t.metadata.get(STUDY_ID_FIELD, None),
              converts_to_parameter=False,
          )
      )

    return cls(
        parameter_converters,
        [DefaultModelOutputConverter(m) for m in metric_information],
    )

  @classmethod
  def from_study_config(
      cls, study_config: pyvizier.ProblemStatement
  ) -> 'DefaultTrialConverter':
    return cls.from_study_configs(
        [study_config],
        study_config.metric_information,
        use_study_id_feature=False,
    )


@attr.define
class TrialToArrayConverter:
  """EXPERIMENTAL: A quick-and-easy converter that returns a single array.

  Unlike TrialtoNumpyDict converters, `to_features` and `to_labels`
  return a single array of floating numbers. CATEGORICAL and DISCRETE parameters
  are one-hot embedded.

  IMPORTANT: Use a factory method (currently, there is one: `from_study_config`)
  instead of `__init__`.

  WARNING: This class is not exchangable with `TrialToNumpyDict`, and does not
  have functions that return shape or metric informations. Use it at your own
  risk.
  """

  _impl: DefaultTrialConverter

  def to_features(self, trials) -> np.ndarray:
    """Returns the features array with dimension: (n_trials, n_features)."""
    return dict_to_array(self._impl.to_features(trials))

  def to_labels(self, trials) -> np.ndarray:
    """Returns the labels array with dimension: (n_trials, n_metrics)."""
    # Pad up the labels.
    return dict_to_array(self._impl.to_labels(trials))

  def to_xy(self, trials) -> Tuple[np.ndarray, np.ndarray]:
    return self.to_features(trials), self.to_labels(trials)

  def to_parameters(self, arr: np.ndarray) -> Sequence[pyvizier.ParameterDict]:
    """Convert to nearest feasible parameter value. NaNs are preserved."""
    arrformat = DictOf2DArrays(self._impl.to_features([]))
    return self._impl.to_parameters(arrformat.dict_like(arr))

  @classmethod
  def from_study_config(
      cls,
      study_config: pyvizier.ProblemStatement,
      *,
      scale: bool = True,
      pad_oovs: bool = True,
      max_discrete_indices: int = 0,
      flip_sign_for_minimization_metrics: bool = True,
      should_clip=True,
      dtype=np.float64,
  ) -> 'TrialToArrayConverter':
    """From study config.

    Args:
      study_config:
      scale: If True, scales the parameters to [0, 1] range.
      pad_oovs: If True, add an extra dimension for out-of-vocabulary values for
        non-CONTINUOUS parameters.
      max_discrete_indices: For DISCRETE and INTEGER parameters that have more
        than this many feasible values will be continuified. When generating
        suggestions, values are rounded to the nearest feasible value. Note this
        default is different from the default in DefaultModelInputConverter.
      flip_sign_for_minimization_metrics: If True, flips the metric signs so
        that every metric maximizes.
      should_clip: Whether or not clipping should be done.
      dtype: dtype

    Returns:
      TrialToArrayConverter
    """

    def create_input_converter(parameter):
      return DefaultModelInputConverter(
          parameter,
          scale=scale,
          max_discrete_indices=max_discrete_indices,
          onehot_embed=True,
          float_dtype=dtype,
          pad_oovs=pad_oovs,
          should_clip=should_clip,
      )

    def create_output_converter(metric):
      return DefaultModelOutputConverter(
          metric,
          flip_sign_for_minimization_metrics=flip_sign_for_minimization_metrics,
          dtype=dtype,
      )

    sc = study_config  # alias, to keep pylint quiet in the next line.
    converter = DefaultTrialConverter(
        [
            create_input_converter(p)
            for p in sc.search_space.root.select_all().merge()
        ],
        [create_output_converter(m) for m in sc.metric_information],
    )
    return cls(converter)

  @property
  def output_specs(self) -> Sequence[NumpyArraySpec]:
    return [
        converter.output_spec for converter in self._impl.parameter_converters
    ]

  @property
  def metric_specs(self) -> Sequence[pyvizier.MetricInformation]:
    return [mc.metric_information for mc in self._impl.metric_converters]

  @property
  def dtype(self) -> np.dtype:
    return self.to_features([]).dtype
