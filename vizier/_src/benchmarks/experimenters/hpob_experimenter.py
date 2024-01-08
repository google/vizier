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

"""HPO-B dataset.

Note that we denote (X,Y) as a batched set of trials (i.e. suggestions X and
objectives Y) and (x,y) as a single trial. This is slightly different from (X,y)
notation used in the handler to denote batched trials.
"""
# TODO: Replace internal HPOB experimenter with this.
# pylint:disable=invalid-name
import copy
import enum
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import attr
import attrs
import numpy as np

from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter
from vizier._src.benchmarks.experimenters.hpob import handler as handler_lib

import xgboost as xgb

Open = open

METRIC_NAME = 'objective_value'
# Offset applied to parameter values before the log transformation.
_TF_OFFSET = 1e-4


@attrs.define(auto_attribs=True)
class _Dataset:
  """Raw data from HPO-B.

  X and Y are guaranteed to have compatible shapes. A dataset can be sliced like
  regular numpy arrays but but cannot be indexed at a single point, i.e.
    `dataset[0]` is not allowed
    `dataset[0:1]` is allowed
    `dataset[dataset.Y > 0]` is allowed.

  If the log-transformation is applied to a feature, it's offset by a constant
    x_log = np.log(x+0.0001)

  Attributes:
    X: 2-D array of shape (number of observations) * (number of input features).
      The features may be scaled and log-transformed. _SearchspaceDescriptor
      holds the necessary information to recover the original values.
    Y: 2-D array of objective values, of shape (number of observations, 1). The
      values are not pre-processed.
  """

  X: np.ndarray = attrs.field(converter=np.asarray)
  Y: np.ndarray = attrs.field(converter=np.asarray)

  def __attrs_post_init__(self) -> None:
    """Performs validation."""
    if len(self.X.shape) != 2:
      raise ValueError(f'X must be 2-D. Given: {self.X.shape}')
    if len(self.Y.shape) != 2:
      raise ValueError(f'Y must be 2-D. Given: {self.Y.shape}')

    if self.X.shape[0] != self.Y.shape[0]:
      raise ValueError(f'X and y must have same number of rows. '
                       f'X.shape={self.X.shape}, y.shape={self.Y.shape}')

  def __getitem__(self, idx: slice) -> '_Dataset':
    return _Dataset(self.X[idx], self.Y[idx])

  def __len__(self) -> int:
    return self.Y.shape[0]


@attr.define(init=True, kw_only=False)
class _VariableDescriptor:
  """Variable descriptor."""
  name: str = attrs.field()
  min_value: Optional[float] = attrs.field(default=None)
  max_value: Optional[float] = attrs.field(default=None)
  min_value_before_tf: Optional[float] = attrs.field(default=None)
  max_value_before_tf: Optional[float] = attrs.field(default=None)
  apply_log: bool = attrs.field(default=True)
  optional: bool = attrs.field(default=False)
  categories: List[str] = attrs.field(factory=list)

  def scale(self, x: np.ndarray) -> np.ndarray:
    # Raw numbers to HPOB scale. Input and output are 1-D
    if self.is_categorical:
      return x
    if self.apply_log:
      x = np.log(x)
    x = (x - self.min_value) / (self.max_value - self.min_value)
    return x

  def unscale(self, x: np.ndarray) -> np.ndarray:
    # HPOB scale to raw numbers. Input and output are 1-D.
    if self.is_categorical:
      return x
    x = x * (self.max_value - self.min_value) + self.min_value
    if self.apply_log:
      x = np.exp(x)
    bounds = self.bounds
    return np.clip(x, bounds[0], bounds[1])

  @classmethod
  def from_info(cls, name: str, info: Dict[str, Any], apply_log: bool,
                optional: bool) -> '_VariableDescriptor':
    if 'categories' in info:
      return cls(name, categories=info['categories'], optional=optional)
    else:
      return cls(
          name,
          min_value=info['min'],
          max_value=info['max'],
          min_value_before_tf=info['min_before_tf'],
          max_value_before_tf=info['max_before_tf'],
          apply_log=apply_log,
          optional=optional)

  @property
  def is_categorical(self) -> bool:
    return bool(self.categories)

  @property
  def bounds(self) -> Optional[Tuple[float, float]]:
    if self.is_categorical:
      bounds = None
    elif self.apply_log:
      bounds = (self.min_value_before_tf + _TF_OFFSET,
                self.max_value_before_tf + _TF_OFFSET)
    else:
      bounds = (self.min_value, self.max_value)
    return bounds  # pytype: disable=bad-return-type

  def as_parameter_config(self) -> vz.ParameterConfig:
    """Return the Vizier parameter config."""
    if self.is_categorical:
      return vz.ParameterConfig.factory(
          self.name, feasible_values=self.categories)
    else:
      if self.apply_log:
        scale_type = vz.ScaleType.LOG
      else:
        scale_type = vz.ScaleType.LINEAR
      return vz.ParameterConfig.factory(
          self.name, bounds=self.bounds, scale_type=scale_type)

  def as_discrete_parameter_config(self, feasible_values) -> vz.ParameterConfig:
    if self.is_categorical:
      raise ValueError('Only continuous parameters can be discretized.')
    else:
      pc = self.as_parameter_config()
      return vz.ParameterConfig.factory(
          pc.name, feasible_values=feasible_values, scale_type=pc.scale_type)


@attrs.define
class _SearchspaceDescriptor:
  variables: Dict[str, _VariableDescriptor]
  order: List[str]

  def column_index(self, name: str) -> int:
    return self.order.index(name)


class NaPolicy(enum.Enum):
  """Decides what to do given an optional parameter that may be missing.

    'drop': *_na does not exist in search space, but trials may be missing the
      associated parameter.
    'discrete': *_na is a separate parameter that takes value 0 or 1.
    'continuous': *_na is a separate parameter that takes value in [0, 1].
  """
  DROP = 'DROP'
  DISCRETE = 'DISCRETE'
  CONTINUOUS = 'CONTINUOUS'


class CategoricalPolicy(enum.Enum):
  """Decides what to do given a C-way categorical parameter.

    'as_categorical': Treat it as a categorical parameter.
    'as_continuous': Treat it as C continuous parameters in [0, 1], i.e.
      [0, 1]^C.
  """
  AS_CATEGORICAL = 'AS_CATEGORICAL'
  AS_CONTINUOUS = 'AS_CONTINUOUS'


class _HPOBVizierConverter:
  """Converts between HPOB and Vizier representations."""

  def __init__(
      self,
      descriptor: _SearchspaceDescriptor,
      data: Optional[_Dataset] = None,
      *,
      na_policy: NaPolicy = NaPolicy.DROP,
      categorical_policy: CategoricalPolicy = CategoricalPolicy.AS_CATEGORICAL):
    """Init.

    Args:
      descriptor: Search space descriptor.
      data: If not None, automatically identify discrete features.
      na_policy: See NaPolicy.
      categorical_policy: See CategoricalPolicy.

    Raises:
      ValueError:
    """
    self._descriptor = descriptor

    self.problem = vz.ProblemStatement()
    self.problem.metric_information.append(
        vz.MetricInformation(
            name=METRIC_NAME, goal=vz.ObjectiveMetricGoal.MAXIMIZE))
    self._na_policy = na_policy
    self._categorical_policy = categorical_policy

    if data is not None and data.X.shape[1] != len(descriptor.order):
      raise ValueError(
          f'data shape {data.X.shape} is not compatible with descriptor, which'
          f'defines {len(descriptor.order)} columns!')

    space = self.problem.search_space
    for variable in descriptor.variables.values():
      if variable.is_categorical:
        if self._categorical_policy == CategoricalPolicy.AS_CATEGORICAL:
          space.add(variable.as_parameter_config())
        elif self._categorical_policy == CategoricalPolicy.AS_CONTINUOUS:
          for category in variable.categories:
            space.add(
                vz.ParameterConfig.factory(
                    name=f'{variable.name}.ohe._{category}', bounds=(0., 1.)))
        else:
          raise ValueError(f'Unknown policy: {self._categorical_policy}')
      elif variable.optional:
        space.add(variable.as_parameter_config())
        # For optional parameters, handle the na dimension.
        if self._na_policy == NaPolicy.DROP:
          pass
        elif self._na_policy == NaPolicy.DISCRETE:
          # Optional variable.
          space.add(
              vz.ParameterConfig.factory(
                  f'{variable.name}.na', feasible_values=(0., 1.)))
        elif self._na_policy == NaPolicy.CONTINUOUS:
          # Optional variable.
          space.add(
              vz.ParameterConfig.factory(
                  f'{variable.name}.na', bounds=(0., 1.)))
        else:
          raise ValueError(f'Unknown policy: {self._na_policy}')
      elif data is not None:
        # TODO: Support integer conversions
        # TODO: Clean up with PEP 572 once python 3.8 arrives.
        uniq = np.unique(data.X[:, descriptor.column_index(variable.name)])
        if uniq.size < 10:
          space.add(
              variable.as_discrete_parameter_config(
                  list(variable.unscale(uniq))))
        else:
          space.add(variable.as_parameter_config())
      else:
        space.add(variable.as_parameter_config())

  def to_trials(self, dataset: _Dataset) -> List[vz.Trial]:
    """Convert HPOB data to Vizier trials."""
    xarray, yarray = dataset.X, dataset.Y
    trials = []
    for xrow, yrow in zip(xarray, yarray):
      trial = vz.Trial()
      params = trial.parameters
      for val, column_name in zip(xrow, self._descriptor.order):
        if '.ohe._' in column_name:
          # Categorical parameter
          if self._categorical_policy == CategoricalPolicy.AS_CATEGORICAL:
            if val:
              variable_name, category = column_name.split('.ohe._')
              if variable_name in params:
                raise ValueError(
                    f'The categorical parmateter {variable_name} has '
                    'more than one-hot encoding dimensions set to non-zero: '
                    f'{params[variable_name]} and {category}.')
              params[variable_name] = category
          elif self._categorical_policy == CategoricalPolicy.AS_CONTINUOUS:
            params[column_name] = float(val)
          else:
            raise ValueError(f'Unknown policy: {self._categorical_policy}')
        elif column_name.endswith('.na'):
          if self._na_policy == NaPolicy.DROP:
            # Delete the associated parameter.
            variable_name = column_name[:column_name.find('.na')]
            if variable_name not in params:
              raise ValueError(
                  f'This code assumes that VARIABLE column precedes VARIABLE.na,'
                  f'which is the case for HPOB-v3. '
                  f'However, {variable_name} did not precede {column_name})')
            if val:
              del params[variable_name]
          elif self._na_policy in (NaPolicy.DISCRETE, NaPolicy.CONTINUOUS):
            # Treat .na column as a regular parameter.
            params[column_name] = float(val)
          else:
            raise ValueError(f'Unknown policy: {self._na_policy}')
        else:
          params[column_name] = self._descriptor.variables[column_name].unscale(  # pytype: disable=unsupported-operands
              val)
      trial.complete(vz.Measurement({METRIC_NAME: yrow}))
      trials.append(trial)

    return trials

  def array_dim(self) -> int:
    """Returns the second dimension of the to_array()."""
    return len(self._descriptor.order)

  def to_array(self, trials: Sequence[vz.Trial]) -> np.ndarray:
    """Convert trial parameters to HPOB scaled array.

    Args:
      trials: Length N sequence.

    Returns:
      Array of shape [N, D], where D = self.array_dim()
    """
    all_values = []
    for trial in trials:
      params = trial.parameters
      values = []
      for column_name in self._descriptor.order:
        if '.ohe._' in column_name:
          if self._categorical_policy == CategoricalPolicy.AS_CATEGORICAL:
            # one hot embedding
            variable_name, category = column_name.split('.ohe._')
            value = float(params.get_value(variable_name) == category)
          elif self._categorical_policy == CategoricalPolicy.AS_CONTINUOUS:
            # continuous
            value = float(params.get_value(column_name))
          else:
            raise ValueError(f'Unknown policy: {self._categorical_policy}')
        elif column_name.endswith('.na'):
          if self._na_policy == NaPolicy.DROP:
            variable_name = column_name[:column_name.find('.na')]
            value = float(params.get_value(variable_name, None) is None)
          elif self._na_policy in (NaPolicy.CONTINUOUS, NaPolicy.DISCRETE):
            value = float(params.get_value(column_name))
          else:
            raise ValueError(f'Unknown policy: {self._na_policy}')
        else:
          if column_name in params:
            value = params.get_value(column_name)
            value = self._descriptor.variables[column_name].scale(value)  # pytype: disable=wrong-arg-types
          else:
            value = .0
        values.append(value)
      all_values.append(values)

    return np.array(all_values, dtype=np.float64)


def _surrogate_bounds(handler: handler_lib.HPOBHandler, search_space_id: str,
                      dataset_id: str) -> Tuple[float, float]:
  """Surrogate function bounds."""
  surrogate_name = 'surrogate-' + search_space_id + '-' + dataset_id
  y_min = handler.surrogates_stats[surrogate_name]['y_min']
  y_max = handler.surrogates_stats[surrogate_name]['y_max']
  return y_min, y_max


@attr.define
class HPOBExperimenter(experimenter.Experimenter):
  """HPOB Experimenter. Use HPOBContainer to create a valid instance."""
  _converter: _HPOBVizierConverter = attr.field()
  _handler: handler_lib.HPOBHandler = attr.field()

  _search_space_id: str = attr.field()
  _dataset_id: str = attr.field()
  _seed_trials: Sequence[vz.Trial] = attr.field()
  _normalize_y: bool = attr.field(kw_only=True)

  # Surrogate maps scaled and log-transformed features to raw objective values.
  _surrogate: xgb.Booster = attr.field(init=False)
  # Minimum value of the EvaluateContinuous output.
  _y_min: float = attr.field(init=False)
  # Maximum value of the EvaluateContinuous output.
  _y_max: float = attr.field(init=False)

  def __attrs_post_init__(self):
    self._surrogate = self._load_surrogate(self._search_space_id,
                                           self._dataset_id)
    self._y_min, self._y_max = _surrogate_bounds(self._handler,
                                                 self._search_space_id,
                                                 self._dataset_id)

  def _load_surrogate(self, search_space_id: str,
                      dataset_id: str) -> xgb.Booster:
    bst_surrogate = xgb.Booster()
    surrogate_name = 'surrogate-' + search_space_id + '-' + dataset_id
    bst_surrogate.load_model(self._handler.surrogates_dir + surrogate_name +
                             '.json')
    return bst_surrogate

  def array_dim(self) -> int:
    return self._converter.array_dim()

  def normalize_ys(self, y_array: np.ndarray) -> np.ndarray:
    new_y = (y_array - self._y_min) / (self._y_max - self._y_min)
    # Clip to eliminate potential numerical errors.
    new_y = np.clip(new_y, 0., 1.)
    return new_y

  def unnormalize_ys(self, y_array: np.ndarray) -> np.ndarray:
    new_y = self._y_min + y_array * (self._y_max - self._y_min)
    # Clip to eliminate potential numerical errors.
    new_y = np.clip(new_y, self._y_min, self._y_max)
    return new_y

  def _EvaluateArray(self, x_array: np.ndarray, normalize: bool) -> np.ndarray:
    x_q = xgb.DMatrix(x_array)
    new_y = self._surrogate.predict(x_q)
    if normalize:
      return self.normalize_ys(new_y)
    else:
      return np.clip(new_y, self._y_min, self._y_max)

  def EvaluateArray(self, x_array: np.ndarray) -> np.ndarray:
    """Evaluates the surrogate model on numpy array.

    Args:
      x_array: Size [N, D] array where D = self.array_dim(). The entries are in
        [0, 1] range.

    Returns:
      Size [N] array.
    """
    return self._EvaluateArray(x_array, self._normalize_y)

  def _EvaluateContinuous(self, trial: vz.Trial, normalize: bool) -> float:
    x_array = self._converter.to_array([trial])
    y_array = self._EvaluateArray(x_array[0].reshape(1, -1), normalize)
    return float(y_array.item())

  def EvaluateContinuous(self, trial: vz.Trial) -> float:
    """Returns a float."""
    # Convert Trial to X matrix.
    return self._EvaluateContinuous(trial, normalize=self._normalize_y)

  def evaluate(self, suggestions: Sequence[vz.Trial]):
    """Populates three metrics."""
    for suggestion in suggestions:
      unnormalized = self._EvaluateContinuous(suggestion, normalize=False)
      normalized = self._EvaluateContinuous(suggestion, normalize=True)
      y = normalized if self._normalize_y else unnormalized

      final_measurement = vz.Measurement(
          metrics={
              METRIC_NAME: y,
              f'{METRIC_NAME}_unnormalized': unnormalized,
              f'{METRIC_NAME}_normalized': normalized
          })
      suggestion.complete(final_measurement)

  def problem_statement(self) -> vz.ProblemStatement:
    problem = copy.deepcopy(self._converter.problem)
    problem.metadata.ns('hpob').update({
        'search_space_id': self._search_space_id,
        'dataset_id': self._dataset_id
    })
    return problem

  def __str__(self) -> str:
    return (f'HPOB experimenter on search_space_id:{self._search_space_id}, '
            f'dataset_id:{self._dataset_id}')

  def DebugString(self) -> str:
    return str(self)

  def get_initial_trials(
      self,
      count: int = 5,
      *,
      hpob_seed: Optional[str] = None,
      rng: Optional[np.random.RandomState] = None) -> List[vz.Trial]:
    """Get an initial batch of trials.

    Args:
      count:
      hpob_seed: If set, must be one of ( 'test0', 'test1', 'test2', 'test3',
        'test4'). Uses the fixed seed trials from HPO-B.
      rng: Must be set if hpob_seed is unset. Chooses initial trials randomly.

    Returns:
      List of seed trials.
    """
    if hpob_seed:
      init_ids = self._handler.bo_initializations[self._search_space_id][
          self._dataset_id][hpob_seed][:count]
    elif rng is None:
      raise ValueError('rng must be provided if hpob_seed is not set.')
    else:
      init_ids = rng.choice(
          np.arange(len(self._seed_trials)), count, replace=False)

    initial_trials = [self._seed_trials[init_id] for init_id in init_ids]
    # Assign trial id and final_measurement.objective_value.
    for i, trial in enumerate(initial_trials):
      trial.id = i + 1
      for metric in trial.final_measurement.metrics:
        if metric.name == METRIC_NAME:
          break
      else:
        raise ValueError(f'No metric named {METRIC_NAME} is found.')
    return initial_trials


# Download files from https://github.com/releaunifreiburg/HPO-B/.
ROOT_DIR = 'hpob-data/'
SURROGATES_DIR = 'saved-surrogates/'

DEFAULT_TEST_MODE = 'v3-test'
DEFAULT_MODE = 'v3-train-augmented'

TRAIN = 'train'
VALID = 'validation'
TEST = 'test'

SPLITS = [TRAIN, VALID, TEST]


class HPOBContainer:
  """HPOB container."""

  def __init__(
      self,
      handler: Optional[handler_lib.HPOBHandler] = None,
      *,
      auto_discretize: bool = False,
      na_policy: NaPolicy = NaPolicy.DROP,
      categorical_policy: CategoricalPolicy = CategoricalPolicy.AS_CATEGORICAL,
      normalize_y: bool = False,
      clip_y: bool = True,
      metadata_dir: str = ROOT_DIR,
      use_surrogate_values: bool = True):
    """Init.

    Args:
      handler: HPOBHandler. If not provided, a new one is automatically created
        from the default path.
      auto_discretize: If True, inspect the data columns and identify discrete
        parameters.
      na_policy: See NaPolicy.
      categorical_policy: See CategoricalPolicy.
      normalize_y: Normalize y values to [0, 1] from handler's surrogates_stats.
      clip_y: clip y value by [y_min, y_max] from handler's surrogates_stats.
        This is required to prepare initial seeds in the continuous evaluation
        setting.
      metadata_dir: Path to `meta_dataset_descriptors_v2.json` file.
      use_surrogate_values: If True, use surrogate values in get_study() and
        get_xy() methods.
    """
    if not handler:
      handler = handler_lib.HPOBHandler(
          root_dir=ROOT_DIR, mode=DEFAULT_MODE, surrogates_dir=SURROGATES_DIR)
    metadata_path = metadata_dir + 'meta_dataset_descriptors_v2.json'
    with Open(metadata_path, 'rt') as f:
      self._meta_dataset_descriptors_v2 = json.load(f)
    self._handler = handler

    self._auto_discretize = auto_discretize
    self._na_policy = na_policy
    self._categorical_policy = categorical_policy
    self._clip_y = clip_y
    self._normalize_y = normalize_y
    self._use_surrogate_values = use_surrogate_values

    self._all_data: Dict[Tuple[str, str], _Dataset] = dict()
    for split in (self._handler.meta_train_data,
                  self._handler.meta_validation_data,
                  self._handler.meta_test_data):
      if split is not None:
        for search_space_id, datasets in split.items():
          for dataset_id, datadict in datasets.items():
            y = datadict['y']
            if self._clip_y:
              y_min, y_max = _surrogate_bounds(self._handler, search_space_id,
                                               dataset_id)
              y = np.clip(y, y_min, y_max)
            self._all_data[search_space_id,
                           dataset_id] = _Dataset(datadict['X'], y)

  def dataset_keys(self, split: str) -> Iterable[Tuple[str, str]]:
    """Iterate through all dataset keys for the given split.

    Args:
      split: HPOBContainer.TRAIN, VALID, or TEST.

    Yields:
      Tuple of search_space_id and dataset_id.
    """
    if split == TRAIN:
      root = self._handler.meta_train_data
    elif split == VALID:
      root = self._handler.meta_validation_data
    elif split == TEST:
      root = self._handler.meta_test_data

    if root is not None:
      for search_space_id, dataset_ids in root.items():
        for dataset_id in dataset_ids:
          yield search_space_id, dataset_id

  def _get_descriptor(self, search_space_id: str) -> _SearchspaceDescriptor:
    """Get the search space descriptor."""
    d = self._meta_dataset_descriptors_v2[search_space_id]
    variables = d['variables']
    descriptors = dict()
    for name in variables:
      if name.endswith('.na'):
        # These variables indicate whether an "optional" parameter is specified
        # or not. We ignore them for now.
        continue
      descriptors[name] = _VariableDescriptor.from_info(
          name,
          variables[name],
          apply_log=name in d['variables_to_apply_log'],
          optional=f'{name}.na' in variables)
    return _SearchspaceDescriptor(descriptors, d['variables_order'])

  def get_xy(self, search_space_id: str,
             dataset_id: str) -> Tuple[np.ndarray, np.ndarray]:
    data, _ = self._get_data_and_converter(search_space_id, dataset_id)
    return data.x, data.y  # pytype: disable=attribute-error

  def _get_data_and_converter(
      self, search_space_id: str,
      dataset_id: str) -> Tuple[_Dataset, _HPOBVizierConverter]:
    """Get data and converter."""
    data: _Dataset = copy.deepcopy(self._all_data[search_space_id, dataset_id])

    if self._normalize_y:
      y_min, y_max = _surrogate_bounds(self._handler, search_space_id,
                                       dataset_id)
      data.Y = (data.Y - y_min) / (y_max - y_min)

    converter = _HPOBVizierConverter(
        self._get_descriptor(search_space_id),
        (data if self._auto_discretize else None),
        na_policy=self._na_policy,
        categorical_policy=self._categorical_policy)
    if self._use_surrogate_values:
      surrogate = self._get_experimenter_from_data_and_converter(
          data, converter, search_space_id, dataset_id)
      data.Y = surrogate.EvaluateArray(data.X)
    return data, converter

  def _get_experimenter_from_data_and_converter(
      self, data: _Dataset, converter: _HPOBVizierConverter,
      search_space_id: str, dataset_id: str) -> HPOBExperimenter:
    return HPOBExperimenter(
        converter,
        self._handler,
        search_space_id,
        dataset_id,
        converter.to_trials(data),
        normalize_y=self._normalize_y)

  def get_experimenter(self, search_space_id: str,
                       dataset_id: str) -> HPOBExperimenter:
    data, converter = self._get_data_and_converter(search_space_id, dataset_id)
    return self._get_experimenter_from_data_and_converter(
        data, converter, search_space_id, dataset_id)

  def get_problem_and_trials(self, search_space_id: str,
                             dataset_id: str) -> vz.ProblemAndTrials:
    data, converter = self._get_data_and_converter(search_space_id, dataset_id)
    return vz.ProblemAndTrials(
        problem=converter.problem, trials=converter.to_trials(data))
