"""Abstractions and default converters."""

import abc
import copy
import dataclasses
import enum
import itertools
from typing import Any, Callable, Collection, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

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
  """
  CONTINUOUS = 'CONTINUOUS'
  DISCRETE = 'DISCRETE'
  ONEHOT_EMBEDDING = 'ONEHOT_EMBEDDING'

  @classmethod
  def default_factory(cls,
                      pc: pyvizier.ParameterConfig) -> 'NumpyArraySpecType':
    """SpecType when encoding discretes as integer indices."""
    if pc.type == pyvizier.ParameterType.DOUBLE:
      return NumpyArraySpecType.CONTINUOUS
    elif pc.type in (pyvizier.ParameterType.DISCRETE,
                     pyvizier.ParameterType.CATEGORICAL,
                     pyvizier.ParameterType.INTEGER):
      return NumpyArraySpecType.DISCRETE
    raise ValueError(f'Unknown type {pc.type}')

  @classmethod
  def embedding_factory(cls,
                        pc: pyvizier.ParameterConfig) -> 'NumpyArraySpecType':
    """SpecType when encoding discretes as onehot embedding."""
    if pc.type == pyvizier.ParameterType.DOUBLE:
      return NumpyArraySpecType.CONTINUOUS
    elif pc.type in (pyvizier.ParameterType.DISCRETE,
                     pyvizier.ParameterType.CATEGORICAL,
                     pyvizier.ParameterType.INTEGER):
      return NumpyArraySpecType.ONEHOT_EMBEDDING
    raise ValueError(f'Unknown type {pc.type}')


@attr.define(frozen=True, auto_attribs=True)
class NumpyArraySpec:
  """Encodes what an array represents.

  This class is similar to `BoundedTensorSpec` in tf agents, except it carries
  extra information specific to vizier.

  If `type` is `DOUBLE`, then `dtype` is a floating type, and bounds are
  floating numbers. num_dimensions is always 1, and num_oovs is zero.

  If 'type' is `DISCRETE`, then `dtype` is an integer type, and bounds are
  integers. num_dimensions is always 1. Suppose `bounds=(x,y)`. Then integers
  x to (y-num_oovs) correspond to valid parameter values. The rest represent
  out-of-vocabulary values. For example, an integer parameter in range (1,3)
  can be represented by a DISCRETE NumpyArraySpec with bounds=(1,4) and oov=1.

  If 'type' is `ONEHOT_EMBEDDING`, then `dtype` is a floating type, and bounds
  are floating numbers. Suppose num_dimensions is X.

  Attributes:
    type: Underlying type of the Vizier parameter corresponding to the array.
    dtype: Numpy array's type.
    bounds: Always inclusive in both directions.
    num_dimensions: Corresponds to shape[-1] of the numpy array. When `type` is
      `ONEHOT_EMBEDDING`, the first X dimensions correspond to valid parameter
      values. The other dimensions correspond to out-of-vocabulary values.
      Otherwise, it is simply 1.
    name: Parameter name.
    num_oovs: Number of out-of-vocabulary items, for non-continuous type.
    scale: Scaling of the values.
  """
  type: NumpyArraySpecType
  dtype: np.dtype
  bounds: Union[Tuple[float, float], Tuple[int, int]]
  num_dimensions: int
  name: str
  num_oovs: int
  scale: Optional[pyvizier.ScaleType] = None

  def __attrs_post_init__(self):
    object.__setattr__(self, 'bounds',
                       tuple(np.array(self.bounds, dtype=self.dtype)))

  @classmethod
  def from_parameter_config(
      cls,
      pc: pyvizier.ParameterConfig,
      type_factory: Callable[
          [pyvizier.ParameterConfig],
          NumpyArraySpecType] = NumpyArraySpecType.default_factory,
      floating_dtype: np.dtype = np.float32,
      int_dtype: np.dtype = np.int32,
      *,
      pad_oovs: bool = True) -> 'NumpyArraySpec':
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
          floating_dtype,
          bounds=pc.bounds,
          num_dimensions=1,
          scale=pc.scale_type,
          name=pc.name,
          num_oovs=0)
    elif the_type == NumpyArraySpecType.DISCRETE:
      return NumpyArraySpec(
          the_type,
          int_dtype,
          bounds=(0, len(pc.feasible_values)),
          num_dimensions=1,
          name=pc.name,
          num_oovs=1 if pad_oovs else 0)
    elif the_type == NumpyArraySpecType.ONEHOT_EMBEDDING:
      return NumpyArraySpec(
          the_type,
          floating_dtype,
          bounds=(0., 1.),
          num_dimensions=len(pc.feasible_values) + 1,
          name=pc.name,
          num_oovs=1 if pad_oovs else 0)
    raise ValueError(f'Unknown type {type}')


def dict_to_array(array_dict: Dict[str, np.ndarray]) -> np.ndarray:
  r"""Converts a dict of (..., D_i) arrays to a (..., \sum_i D_i) array."""
  return np.concatenate(list(array_dict.values()), axis=-1)


class DictOf2DArrays(Mapping[str, np.ndarray]):
  """Dictionary of string to 2D arrays.

  All arrays share the first dimension, which is at a high level, the number of
  objects that this dictionary corresponds to.

  Attributes:
    size: Array's shape[0].
  """

  def __init__(self, d: Dict[str, np.ndarray]):
    self._d = d
    shape = None
    for k, v in self.items():
      if shape is None:
        shape = v.shape
        if len(shape) != 2:
          raise ValueError(f'{k} has shape {v.shape} which is not length 2.'
                           'DictOf2DArrays only supports 2D numpy arrays.')
      if shape[0] != v.shape[0]:
        raise ValueError(
            f'{k} has shape {v.shape} which is not equal to {shape}.')
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
        {k: np.concatenate([self[k], other[k]], axis=0) for k in self})

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

  A typical Keras/JAX pipeline consists of:
    1. Load data into arrays.
    2. Call Model.build() to initialize a model for the loaded data shape.
    3. Call Model.fit() to train the model.
    4. Call Model.__call__() to predict with the model.

  This abstraction allows a shared implementation of steps 1,2 and 3.
  """

  @abc.abstractmethod
  def to_xy(
      self, trials: Sequence[pyvizier.Trial]
  ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Returns (x,y) pair that can be used as input for keras.Model.fit()."""
    pass

  @property
  @abc.abstractmethod
  def features_shape(self) -> Dict[str, Sequence[Union[int, None]]]:
    """Returned value can be used as `input_shape` for keras.Model.build()."""
    pass

  @property
  @abc.abstractmethod
  def output_specs(self) -> Dict[str, NumpyArraySpec]:
    """Same keys as features_shape, with more details."""
    pass

  @property
  @abc.abstractmethod
  def labels_shape(self) -> Dict[str, Any]:
    pass

  @property
  @abc.abstractmethod
  def metric_information(self) -> Dict[str, pyvizier.MetricInformation]:
    pass


class ModelInputConverter(metaclass=abc.ABCMeta):
  """Interface for extracting inputs to the model."""

  @abc.abstractmethod
  def convert(self, trials: Sequence[pyvizier.Trial]) -> np.ndarray:
    """Returns an array of shape (number of trials, feature dimension).

    Args:
      trials:

    Returns:
      Returns an array of shape (number of trials, feature dimension).
      Subclasses must use a fixed feature dimension. In particular, it should
      be a constant function of the input trials.
    """
    pass

  @property
  @abc.abstractmethod
  def output_spec(self) -> NumpyArraySpec:
    """Provides specification of the output from this converter."""
    pass

  @property
  @abc.abstractmethod
  def parameter_config(self):
    """Original ParameterConfig that this converter acts on."""
    pass

  @abc.abstractmethod
  def to_parameter_values(
      self, array: np.ndarray) -> List[Optional[pyvizier.ParameterValue]]:
    """Convert to parameter values."""
    pass


@dataclasses.dataclass
class _ModelInputArrayBijector:
  """Transformations on the numpy arrays generated by ModelInputConverter."""
  forward_fn: Callable[[np.ndarray], np.ndarray]
  backward_fn: Callable[[np.ndarray], np.ndarray]
  output_spec: NumpyArraySpec  # Spec after forward_fn is applied.

  @classmethod
  def identity(cls, spec) -> '_ModelInputArrayBijector':
    return cls(lambda x: x, lambda x: x, spec)

  @classmethod
  def scaler_from_spec(cls, spec: NumpyArraySpec) -> '_ModelInputArrayBijector':
    """For continuous specs, linearize and scale it to (0, 1) range."""
    low, high = spec.bounds
    if spec.type != NumpyArraySpecType.CONTINUOUS:
      return cls.identity(attr.evolve(spec, scale=None))
    if low == high:

      def backward_fn(y):
        return np.where(np.isfinite(y), np.zeros_like(y) + low, y)

      return cls(lambda x: np.where(np.isfinite(x), np.zeros_like(x), x),
                 backward_fn, attr.evolve(spec, bounds=(.0, 1.), scale=None))

    if spec.scale == pyvizier.ScaleType.LOG:
      low, high = np.log(low), np.log(high)
      denom = (high - low) or 1.0
      if denom < 1e-6:
        logging.warning('Unusually small range detected for %s', spec)
      scale_fn = lambda x, high=high, low=low: (np.log(x) - low) / (high - low)
      unscale_fn = lambda x, high=high, low=low: np.exp(x * (high - low) + low)
    else:
      if not (spec.scale == pyvizier.ScaleType.LINEAR or spec.scale is None):
        logging.warning('Unknown scale type %s. Applying LINEAR', spec.scale)
      denom = (high - low)
      if denom < 1e-6:
        logging.warning('Unusually small range detected for %s', spec)
      if denom == 1.0 and low == 0:
        return cls.identity(attr.evolve(spec, scale=None))
      scale_fn = lambda x, high=high, low=low: (x - low) / (high - low)
      unscale_fn = lambda x, high=high, low=low: x * (high - low) + low

    return cls(scale_fn, unscale_fn,
               attr.evolve(spec, bounds=(.0, .1), scale=None))

  @classmethod
  def onehot_embedder_from_spec(cls,
                                spec: NumpyArraySpec,
                                *,
                                dtype=np.float32,
                                pad_oovs: bool = True
                               ) -> '_ModelInputArrayBijector':
    """Given a discrete spec, one-hot embeds it."""
    if spec.type != NumpyArraySpecType.DISCRETE:
      return cls.identity(spec)

    num_oovs = 1 if pad_oovs else 0
    output_spec = NumpyArraySpec(
        NumpyArraySpecType.ONEHOT_EMBEDDING,
        dtype,
        bounds=(0., 1.),
        num_dimensions=int(spec.bounds[1] - spec.bounds[0] + num_oovs),
        name=spec.name,
        num_oovs=num_oovs,
        scale=None)

    def embed_fn(x: np.ndarray, output_spec=output_spec):
      """x is integer array of [N, 1]."""
      return np.eye(
          output_spec.num_dimensions, dtype=output_spec.dtype)[x.flatten()]

    def unembed_fn(x: np.ndarray, spec=spec, output_spec=output_spec):
      return np.argmax(
          x[:, :output_spec.num_dimensions - output_spec.num_oovs],
          axis=1).astype(spec.dtype)

    return cls(embed_fn, unembed_fn, output_spec)


def _create_default_getter(
    pconfig: pyvizier.ParameterConfig) -> Callable[[pyvizier.Trial], Any]:
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

  def __init__(self,
               parameter_config: pyvizier.ParameterConfig,
               getter: Optional[Callable[[pyvizier.Trial], Any]] = None,
               *,
               float_dtype: np.dtype = np.float32,
               max_discrete_indices: int = 10,
               scale: bool = False,
               onehot_embed: bool = False,
               converts_to_parameter: bool = True,
               pad_oovs: bool = True):
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
      converts_to_parameter: If False, this converter does not correspodn to an
        actual parameter in Vizier search space, and `to_parameter_value` always
        returns None
      pad_oovs: If True, pad the out-of-vocabulary dimensions to onehot
        embedding.
    """
    self._converts_to_parameter = converts_to_parameter
    self._parameter_config = copy.deepcopy(parameter_config)
    if parameter_config.type in (
        pyvizier.ParameterType.INTEGER, pyvizier.ParameterType.DISCRETE
    ) and parameter_config.num_feasible_values > max_discrete_indices:
      parameter_config = parameter_config.continuify()

    # TODO: Make the default getter raise an Error if they encounter an
    # out-of-vocabulary value but pad_oovs is False.
    self._getter = getter or _create_default_getter(parameter_config)
    # Getter spec can only have DISCRETE or CONTINUOUS types.
    self._getter_spec = NumpyArraySpec.from_parameter_config(
        parameter_config,
        NumpyArraySpecType.default_factory,
        floating_dtype=float_dtype)

    # Optionally scale and onehot embed.
    spec = self._getter_spec
    self.scaler = (
        _ModelInputArrayBijector.scaler_from_spec(spec)
        if scale else _ModelInputArrayBijector.identity(spec))
    spec = self.scaler.output_spec
    self.onehot_encoder = (
        _ModelInputArrayBijector.onehot_embedder_from_spec(
            spec, dtype=float_dtype, pad_oovs=pad_oovs)
        if onehot_embed else _ModelInputArrayBijector.identity(spec))
    spec = self.onehot_encoder.output_spec

    self._output_spec = spec

  def convert(self, trials: Sequence[pyvizier.Trial]) -> np.ndarray:
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
      return np.zeros([0, self.output_spec.num_dimensions],
                      dtype=self.output_spec.dtype)

    value_converter = (
        self._convert_index if self._getter_spec.type
        == NumpyArraySpecType.DISCRETE else self._convert_continuous)
    values = [value_converter(t) for t in trials]
    array = np.asarray(values, dtype=self._getter_spec.dtype).reshape([-1, 1])
    return self.onehot_encoder.forward_fn(self.scaler.forward_fn(array))

  def _to_parameter_value(
      self, value: Union['np.float', float,
                         int]) -> Optional[pyvizier.ParameterValue]:
    """Converts to a single parameter value.

    Be aware that the value is automatically truncated.

    Args:
      value:

    Returns:
      ParameterValue.
    """
    if not self._converts_to_parameter:
      return None
    if self.parameter_config.type == pyvizier.ParameterType.DOUBLE:
      # Input parameter was DOUBLE. Output is also DOUBLE.
      return pyvizier.ParameterValue(
          float(
              np.clip(value, self._parameter_config.bounds[0],
                      self._parameter_config.bounds[1])))
    elif self.output_spec.type == NumpyArraySpecType.CONTINUOUS:
      # The parameter config is originally discrete, but continuified.
      # Round to the closest number.
      return pyvizier.ParameterValue(
          min(self.parameter_config.feasible_values,
              key=lambda feasible_value: abs(feasible_value - value)))
    elif value >= len(self.parameter_config.feasible_values):
      return None
    else:
      return pyvizier.ParameterValue(
          self.parameter_config.feasible_values[value])

  def to_parameter_values(
      self, array: np.ndarray) -> List[Optional[pyvizier.ParameterValue]]:
    """Convert and clip to the nearest feasible parameter values."""
    array = self.scaler.backward_fn(self.onehot_encoder.backward_fn(array))
    return [self._to_parameter_value(v) for v in list(array.flatten())]

  def _convert_index(self, trial: pyvizier.Trial):
    """Used for non-continuous types."""
    raw_value = self._getter(trial)
    if raw_value in self.parameter_config.feasible_values:
      return self.parameter_config.feasible_values.index(raw_value)
    else:
      # Return the catch-all missing index.
      return len(self.parameter_config.feasible_values)

  def _convert_continuous(self, trial: pyvizier.Trial):
    """Used for continuous types."""
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

  @property
  @abc.abstractmethod
  def metric_information(self) -> pyvizier.MetricInformation:
    """Reflects how the converter treates the metric.

    If this metric converter flips the signs or changes the semantics of
    safety configs, then the returned metric_information should reflect such
    changes.
    """
    pass

  @property
  def output_shape(self) -> Tuple[None, int]:
    return (None, 1)


class DefaultModelOutputConverter(ModelOutputConverter):
  """Converts measurements into numpy arrays."""

  def __init__(self,
               metric_information: pyvizier.MetricInformation,
               *,
               flip_sign_for_minimization_metrics: bool = False,
               shift_safe_metrics: bool = True,
               dtype: np.dtype = np.float32,
               raise_errors_for_missing_metrics: bool = False):
    """Init.

    Args:
      metric_information:
      flip_sign_for_minimization_metrics: Flips the sign if the metric is to
        minimize.
      shift_safe_metrics: If True, add (minimize) or subtract (maximize) safety
        threshold from the metric value.
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
    return (self._original_metric_information.goal
            == pyvizier.ObjectiveMetricGoal.MINIMIZE and
            self.flip_sign_for_minimization_metrics)

  def convert(
      self,
      measurements: Sequence[Optional[pyvizier.Measurement]]) -> np.ndarray:
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
    if (self.shift_safe_metrics and
        self._original_metric_information.type.is_safety):
      labels -= self._original_metric_information.safety_threshold
    return labels * (-1 if self._should_flip_sign else 1)

  @property
  def metric_information(self) -> pyvizier.MetricInformation:
    """Returns a copy that reflects how the converter treates the metric."""
    metric_information = copy.deepcopy(self._original_metric_information)
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

  def __init__(self,
               parameter_converters: Collection[ModelInputConverter],
               metric_converters: Collection[ModelOutputConverter] = tuple()):
    self.parameter_converters = list(parameter_converters)
    self._parameter_converters_dict = {
        pc.parameter_config.name: pc for pc in self.parameter_converters
    }
    self.metric_converters = list(metric_converters)

  def to_features(self,
                  trials: Sequence[pyvizier.Trial]) -> Dict[str, np.ndarray]:
    """Returned value can be used as `x` for keras.Model.fit()."""
    result_dict = dict()
    for converter in self.parameter_converters:
      result_dict[converter.parameter_config.name] = converter.convert(trials)
    return result_dict

  def to_trials(self, dictionary: Mapping[str,
                                          np.ndarray]) -> List[pyvizier.Trial]:
    """Inverse of `to_features`."""
    return [
        pyvizier.Trial(parameters=p) for p in self.to_parameters(dictionary)
    ]

  def to_parameters(
      self, dictionary: Mapping[str,
                                np.ndarray]) -> List[pyvizier.ParameterDict]:
    """Convert to nearest feasible parameter value. NaNs are preserved."""
    # TODO: Add a boolean flag to disable automatic clipping.
    param_dicts = [
        pyvizier.ParameterDict()
        for _ in range(len(list(dictionary.values())[0]))
    ]
    for key, values in dictionary.items():
      parameter_converter = self._parameter_converters_dict[key]
      parameter_values = parameter_converter.to_parameter_values(values)
      for param_dict, value in zip(param_dicts, parameter_values):
        if value is not None:
          param_dict[key] = value
    return param_dicts

  def to_labels(self,
                trials: Sequence[pyvizier.Trial]) -> Dict[str, np.ndarray]:
    """Returned value can be used as `y` for keras.Model.fit()."""
    result_dict = dict()
    for converter in self.metric_converters:
      result_dict[converter.metric_information.name] = converter.convert(
          [t.final_measurement for t in trials])
    return result_dict

  def to_labels_array(self, trials: Sequence[pyvizier.Trial]) -> np.ndarray:
    """Shorthand for dict_to_array(self.to_labels(trials))."""
    return dict_to_array(self.to_labels(trials))

  def to_xy(
      self, trials: Sequence[pyvizier.Trial]
  ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Returned value can be used as `x`, `y` for keras.Model.fit()."""
    return self.to_features(trials), self.to_labels(trials)

  @property
  def features_shape(self) -> Dict[str, Tuple[Union[int, None], int]]:
    """Returned value can be used as `input_shape` for keras.Model.build()."""
    return {
        pc.output_spec.name: (None, pc.output_spec.num_dimensions)
        for pc in self.parameter_converters
    }

  def compute_features_shape(
      self, trials: Sequence[pyvizier.Trial]) -> Dict[str, Tuple[int, int]]:
    return {k: (len(trials), v[1]) for k, v in self.features_shape.items()}

  @property
  def output_specs(self) -> Dict[str, NumpyArraySpec]:
    return {
        pc.output_spec.name: pc.output_spec for pc in self.parameter_converters
    }

  @property
  def labels_shape(self) -> Dict[str, Sequence[Union[int, None]]]:
    return {
        mc.metric_information.name: mc.output_shape
        for mc in self.metric_converters
    }

  @property
  def metric_information(self) -> Dict[str, pyvizier.MetricInformation]:
    return {
        mc.metric_information.name: mc.metric_information
        for mc in self.metric_converters
    }

  @property
  def parameter_configs(self) -> Dict[str, pyvizier.ParameterConfig]:
    return {
        converter.parameter_config.name: converter.parameter_config
        for converter in self.parameter_converters
    }

  @classmethod
  def from_study_configs(
      cls,
      study_configs: Sequence[pyvizier.StudyConfig],
      metric_information: Collection[pyvizier.MetricInformation],
      *,
      use_study_id_feature: bool = True) -> 'DefaultTrialConverter':
    """Creates a converter from a list of study configs.

    Args:
      study_configs: StudyConfigs to be merged.
      metric_information: MetricInformation of metrics to be used as y-values.
      use_study_id_feature: If True, an extra parameter is added that
        corresponds to the STUDY_ID_FIELD inside metadata.

    Returns:
      `DefaultTrialConverter`.
    """
    # Cache ParameterConfigs.
    # Traverse through all parameter configs and merge the same-named ones.
    parameter_configs: Dict[str, pyvizier.ParameterConfig] = dict()
    for study_config in study_configs:
      all_parameter_configs = itertools.chain.from_iterable([
          top_level_config.traverse()
          for top_level_config in study_config.search_space.parameters
      ])
      for parameter_config in all_parameter_configs:
        name = parameter_config.name  # Alias
        existing_config = parameter_configs.get(name, None)
        if existing_config is None:
          parameter_configs[name] = parameter_config
        else:
          parameter_configs[name] = pyvizier.ParameterConfig.merge(
              existing_config, parameter_config)

    parameter_converters = []
    for pc in parameter_configs.values():
      parameter_converters.append(DefaultModelInputConverter(pc))

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
        logging.error('use_study_id_feature was True, but none of the studies '
                      'had study id configured.')
        use_study_id_feature = False
      elif STUDY_ID_FIELD in parameter_configs:
        raise ValueError('Dataset name conflicts with a ParameterConfig '
                         'that already exists: {}'.format(
                             parameter_configs[STUDY_ID_FIELD]))

      # Create new parameter config.
      parameter_config = pyvizier.ParameterConfig.factory(
          STUDY_ID_FIELD, feasible_values=list(study_ids))
      parameter_configs[STUDY_ID_FIELD] = parameter_config
      logging.info('Created a new ParameterConfig %s', parameter_config)

      # Create converter.
      parameter_converters.append(
          DefaultModelInputConverter(
              parameter_config,
              lambda t: t.metadata.get(STUDY_ID_FIELD, None),
              converts_to_parameter=False))

    return cls(parameter_converters,
               [DefaultModelOutputConverter(m) for m in metric_information])

  @classmethod
  def from_study_config(cls, study_config: pyvizier.StudyConfig):
    return cls.from_study_configs([study_config],
                                  study_config.metric_information,
                                  use_study_id_feature=False)


class TrialToArrayConverter:
  """TrialToArrayConverter.

  Use a factory method (currently, there is one: `from_study_config`) instead
  of `__init__`.

  Unlike TrialtoNumpyDict converters, `to_features` and `to_labels`
  return a single array of floating numbers. CATEGORICAL and DISCRETE parameters
  are one-hot embedded.
  """
  _waiver = 'I am aware that this code may break at any point.'

  def __init__(self, impl: DefaultTrialConverter, _waiver: str = ''):
    """SHOULD NOT BE USED! Use factory classmethods e.g. from_study_config."""

    if _waiver != self._waiver:
      raise ValueError('Sign the waiver if you want to use init directly.')
    self._impl = impl

  def to_features(self, trials) -> np.ndarray:
    return dict_to_array(self._impl.to_features(trials))

  def to_labels(self, trials) -> np.ndarray:
    return dict_to_array(self._impl.to_labels(trials))

  def to_xy(self, trials) -> Tuple[np.ndarray, np.ndarray]:
    return self.to_features(trials), self.to_labels(trials)

  def to_parameters(self, arr: np.ndarray) -> Sequence[pyvizier.ParameterDict]:
    """Convert to nearest feasible parameter value. NaNs are preserved."""
    # TODO: Add a boolean flag to disable automatic clipping.
    arrformat = DictOf2DArrays(self._impl.to_features([]))
    return self._impl.to_parameters(arrformat.dict_like(arr))

  @classmethod
  def from_study_config(cls,
                        study_config: pyvizier.StudyConfig,
                        *,
                        scale: bool = True,
                        pad_oovs: bool = True,
                        max_discrete_indices: int = 0,
                        flip_sign_for_minimization_metrics: bool = True,
                        dtype=np.float64) -> 'TrialToArrayConverter':
    """From study config.

    Args:
      study_config:
      scale: If True, scales the parameters to [0, 1] range.
      pad_oovs: If True, add an extra dimension for out-of-vocabulary values for
        non-CONTINUOUS parameters.
      max_discrete_indices: For DISCRETE and INTEGER types that have more than
        this many feasible values will be continuified. When generating
        suggestions, values are rounded to the nearest feasible value.
      flip_sign_for_minimization_metrics: If True, flips the metric signs so
        that every metric maximizes.
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
          pad_oovs=pad_oovs)

    def create_output_converter(metric):
      return DefaultModelOutputConverter(
          metric,
          flip_sign_for_minimization_metrics=flip_sign_for_minimization_metrics,
          dtype=dtype)

    sc = study_config  # alias, to keep pylint quiet in the next line.
    converter = DefaultTrialConverter(
        [create_input_converter(p) for p in sc.search_space.parameters],
        [create_output_converter(m) for m in sc.metric_information])

    return cls(converter, cls._waiver)

  @property
  def output_specs(self) -> Sequence[NumpyArraySpec]:
    return [
        converter.output_spec for converter in self._impl.parameter_converters
    ]
