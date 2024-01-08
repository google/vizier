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

"""Core population utilities."""
import collections
import json
from typing import Any, Callable, Collection, List, Optional, Sequence, Tuple, Type
import attr
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.evolution import templates
from vizier.interfaces import serializable
from vizier.pyvizier import converters
from vizier.utils import json_utils

# TODO: Use a byte encoding instead of JSON.


def _filter_and_split(
    metrics: Collection[vz.MetricInformation],
) -> Tuple[List[vz.MetricInformation], List[vz.MetricInformation]]:
  """Choose objective and safety metrics and split.

  Args:
    metrics:

  Returns:
    Tuple of objective and safety metrics.
  """
  metrics_by_type = collections.defaultdict(list)
  for metric in metrics:
    metrics_by_type[metric.type].append(metric)

  return (
      metrics_by_type[vz.MetricType.OBJECTIVE],
      metrics_by_type[vz.MetricType.SAFETY],
  )


def _concat(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
  return np.concatenate([a1, a2], axis=0)


def _shape_equals(
    instance_to_shape: Callable[[Any], Collection[Optional[int]]]
):
  """Creates a shape validator for attr.

  For example, _shape_equals(lambda s : [3, None]) validates that the shape has
  length 2 and its first element is 3.

  Args:
    instance_to_shape: Takes instance as input and returns the desired shape for
      the instance. `None` is treated as "any number".

  Returns:
    A validator that can be passed into attr.ib or attr.field.
  """

  def validator(instance, attribute, value) -> None:
    shape = instance_to_shape(instance)

    def _validator_boolean():
      if len(value.shape) != len(shape):
        return False
      for s1, s2 in zip(value.shape, shape):
        if (s2 is not None) and (s1 != s2):
          return False
      return True

    if not _validator_boolean():
      raise ValueError(
          f'{attribute.name} has shape {value.shape} '
          f'which does not match the expected shape {shape}'
      )

  return validator


@attr.define(frozen=True, init=False)
class Offspring(serializable.Serializable):
  """Offspring which corresponds to trial suggestions.

  Attributes:
    xs: Encoding of trial parameters.
    generations: The number of changes this gene went through.
    ids: Identifier for the gene. Can be used to track the "family tree" created
      from the execution of an evolutionary algorithm. For example, when a new
      offspring is sampled, it is assigned a new id and its successors all carry
      the same id.
  """

  xs: np.ndarray = attr.field(
      validator=[
          attr.validators.instance_of(np.ndarray),
          _shape_equals(lambda s: [len(s), None]),
      ]
  )
  ids: np.ndarray = attr.field(
      validator=[
          attr.validators.instance_of(np.ndarray),
          _shape_equals(lambda s: [len(s)]),
      ]
  )

  generations: np.ndarray = attr.field(
      validator=[
          attr.validators.instance_of(np.ndarray),
          _shape_equals(lambda s: [len(s)]),
      ]
  )

  def __init__(
      self,
      xs: np.ndarray,
      ids: Optional[np.ndarray] = None,
      generations: Optional[np.ndarray] = None,
  ):
    if generations is None:
      generations = np.zeros([xs.shape[0]])
    if ids is None:
      ids = np.zeros([xs.shape[0]])

    self.__attrs_init__(xs, ids, generations)

  def __len__(self) -> int:
    return self.generations.shape[0]

  def __getitem__(self, index: Any) -> 'Offspring':
    return Offspring(self.xs[index], self.ids[index], self.generations[index])

  def __add__(self, other: 'Offspring') -> 'Offspring':
    return Offspring(
        _concat(self.xs, other.xs),
        _concat(self.ids, other.ids),
        _concat(self.generations, other.generations),
    )

  @classmethod
  def load(cls: Type['Offspring'], metadata: vz.Metadata) -> 'Offspring':
    encoded = metadata.get('values', cls=str)
    try:
      decoded = json.loads(encoded, object_hook=json_utils.numpy_hook)
    except Exception as e:
      raise serializable.DecodeError('Failed to decode') from e
    return cls(**decoded)

  def dump(self) -> vz.Metadata:
    encoded = json.dumps(attr.asdict(self), cls=json_utils.NumpyEncoder)
    return vz.Metadata({'values': encoded})


@attr.define(frozen=True)
class Population(templates.Population):
  """Population.

  No validations are done.

  Attributes:
    xs: [len(trials), D] array where D is the number of features. The values are
      in [0, 1] range.
    ys: [len(trials), M_1] array where M_1 is the number of objective metrics.
      The values should be pre-processed as maximization metrics.
    cs: [len(trials), M_2] array where M_2 is the number of soft constraint
      metrics. The values should be pre-processed such that the constraint is to
      get them greater than zero.
    ages: The number of selection phases that each entity survived for.
    generations: The number of ancestors of each entity.
    ids: Identifier for the gene aka "family name".
    trial_ids: Trial ids. For debugging and testing only.
  """

  xs: np.ndarray = attr.field(
      validator=[
          attr.validators.instance_of(np.ndarray),
          _shape_equals(lambda s: [len(s), None]),
      ]
  )
  ys: np.ndarray = attr.field(
      validator=[
          attr.validators.instance_of(np.ndarray),
          _shape_equals(lambda s: [len(s), None]),
      ]
  )
  cs: np.ndarray = attr.field(
      validator=[
          attr.validators.instance_of(np.ndarray),
          _shape_equals(lambda s: [len(s), None]),
      ]
  )
  ages: np.ndarray = attr.field(
      validator=[
          attr.validators.instance_of(np.ndarray),
          _shape_equals(lambda s: [len(s)]),
      ]
  )
  generations: np.ndarray = attr.field(
      validator=[
          attr.validators.instance_of(np.ndarray),
          _shape_equals(lambda s: [len(s)]),
      ]
  )
  ids: np.ndarray = attr.field(
      validator=[
          attr.validators.instance_of(np.ndarray),
          _shape_equals(lambda s: [len(s)]),
      ]
  )
  trial_ids: np.ndarray = attr.field(
      validator=[
          attr.validators.instance_of(np.ndarray),
          _shape_equals(lambda s: [len(s)]),
      ]
  )

  def __len__(self) -> int:
    return self.ys.shape[0]

  def __getitem__(
      self,
      index: Any,
     
  ) -> 'Population':
    return Population(**{k: v[index] for k, v in attr.asdict(self).items()})

  def __add__(self, other: 'Population') -> 'Population':
    return Population(
        _concat(self.xs, other.xs),
        _concat(self.ys, other.ys),
        _concat(self.cs, other.cs),
        _concat(self.ages, other.ages),
        _concat(self.generations, other.generations),
        _concat(self.ids, other.ids),
        _concat(self.trial_ids, other.trial_ids),
    )

  @classmethod
  def recover(cls, metadata: vz.Metadata) -> 'Population':
    encoded = metadata.get('values', default='', cls=str)
    try:
      decoded = json.loads(encoded, object_hook=json_utils.numpy_hook)
    except json.JSONDecodeError as e:
      raise serializable.DecodeError('Failed to recover state.') from e
    return cls(**decoded)

  def dump(self) -> vz.Metadata:
    encoded = json.dumps(attr.asdict(self), cls=json_utils.NumpyEncoder)
    return vz.Metadata({'values': encoded})

  def empty_like(self) -> 'Population':
    """Creates an empty population that has the same shape as this."""

    return Population(
        **{
            k: np.zeros([0] + list(v.shape[1:]))
            for k, v in attr.asdict(self).items()
        }
    )


def _create_parameter_converters(
    search_space: vz.SearchSpace,
) -> Collection[converters.DefaultModelInputConverter]:
  """Returns parameter converters."""
  if search_space.is_conditional:
    raise ValueError('Cannot handle conditional search space!')

  def create_input_converter(
      pc: vz.ParameterConfig,
  ) -> converters.DefaultModelInputConverter:
    return converters.DefaultModelInputConverter(
        pc, scale=True, max_discrete_indices=0, onehot_embed=True
    )

  return [create_input_converter(pc) for pc in search_space.parameters]


def _create_metric_converter(
    mc: vz.MetricInformation,
) -> converters.DefaultModelOutputConverter:
  # TODO: Do something other than raising an error
  return converters.DefaultModelOutputConverter(
      mc,
      flip_sign_for_minimization_metrics=True,
      shift_safe_metrics=True,
      raise_errors_for_missing_metrics=True,
  )


class PopulationConverter(templates.PopulationConverter):
  """Population converter."""

  def __init__(
      self,
      search_space: vz.SearchSpace,
      metrics: Collection[vz.MetricInformation],
      *,
      metadata_ns: str = 'population',
      trial_converter: Optional[converters.DefaultTrialConverter] = None,
  ):
    self._objective_metrics, self._safe_metrics = _filter_and_split(metrics)
    self._num_objective_metrics = len(self._objective_metrics)
    self._num_safe_metrics = len(self._safe_metrics)
    self._metrics = self._objective_metrics + self._safe_metrics
    self._metadata_ns = metadata_ns

    self._trial_converter = trial_converter or converters.DefaultTrialConverter(
        _create_parameter_converters(search_space),
        [_create_metric_converter(mc) for mc in self._metrics],
    )
    self._empty_feature_dict = converters.DictOf2DArrays(
        self._trial_converter.to_features([])
    )

  def to_suggestions(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, offsprings: Offspring
  ) -> Collection[vz.TrialSuggestion]:
    parameters_list = self._trial_converter.to_parameters(
        self._empty_feature_dict.dict_like(offsprings.xs)
    )
    suggestions = [vz.TrialSuggestion(p) for p in parameters_list]
    for idx, t in enumerate(suggestions):
      t.metadata.ns(self._metadata_ns).update(offsprings[idx : idx + 1].dump())
    return suggestions

  def _empty_offsprings(self) -> Offspring:
    return Offspring(
        self._empty_feature_dict.asarray(), np.zeros([0]), np.zeros([0])
    )

  def to_population(self, completed: Sequence[vz.CompletedTrial]) -> Population:
    """Converts trials into population. Accepts an empty list."""
    offsprings = self._empty_offsprings()  # create empty
    # Each Trial should contain its genes as metadata. (Note that
    # genes-to-trial mapping is many-to-one). We try to load the genes.
    for t in completed:
      metadata = t.metadata.ns(self._metadata_ns)
      try:
        offsprings += Offspring.load(metadata)
      except serializable.DecodeError:
        # Upon failure, arbitrarily choose one set of genes that map to the
        # current trial.
        offsprings += Offspring(
            converters.DictOf2DArrays(
                self._trial_converter.to_features([t])
            ).asarray(),
            np.zeros([1]),
            np.zeros([1]),
        )

    ys = self._trial_converter.to_labels_array(completed)
    return Population(
        offsprings.xs,
        ys[:, : self._num_objective_metrics],
        ys[:, self._num_objective_metrics :],
        np.zeros([ys.shape[0]]),
        offsprings.generations,
        offsprings.ids,
        np.asarray([t.id for t in completed], dtype=np.int32),
    )


class UniformRandomSampler(templates.Sampler[Offspring]):
  """Generates uniformly random samples."""

  def __init__(self, search_space: vz.SearchSpace, seed: Optional[int] = None):
    if search_space.is_conditional:
      raise ValueError(f'{type(self)} does not support conditional spaces.')
    self._trial_converter = converters.DefaultTrialConverter(
        _create_parameter_converters(search_space)
    )
    self._dimension = sum(
        v[-1] for v in self._trial_converter.features_shape.values()
    )
    self._num_samples = 0
    self._rng = np.random.RandomState(seed)

  def sample(self, count: int) -> Offspring:
    self._num_samples += count
    return Offspring(
        self._rng.random([count, self._dimension]),
        ids=np.arange(self._num_samples - count, self._num_samples),
    )


class LinfMutation(templates.Mutation):
  """L-inf Mutations. Values are assumed to be in [0,1] range."""

  def __init__(self, norm: float = 0.1, seed: Optional[int] = None):
    self._norm = norm
    self._rng = np.random.RandomState(seed)

  def mutate(self, population: Population, count: int) -> Offspring:
    """Perturb by a uniform sample from l-inf ball.

    If the perturbation pushes a coordinate out of [0, 1] range, it "wraps
    around". This is likely suboptimal if optimum is at or near the boundary.

    Args:
      population:
      count:

    Returns:
      Offsprings.
    """
    # Mutate and truncate to [-5, 1.5] range. Perturbations should generally
    # NOT generate values outside this range.
    arr = population.xs.copy()
    arr += self._rng.uniform(-self._norm, self._norm, arr.shape)
    arr = np.maximum(np.minimum(arr, 1.5), -0.5)

    # When the value is outside [0, 1] range, wrap around the limits.
    # -0.2 becomes 0.8, 1.2 becomes 0.2, etc.
    arr = np.where(arr < 0, 1 + arr, np.where(arr > 1, arr - 1, arr))
    return Offspring(arr, population.ids, population.generations + 1)
