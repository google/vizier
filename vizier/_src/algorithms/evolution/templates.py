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

"""Generic interfaces and template for evolutionary algorithms.

The evolutionary algorithm and Vizier interaction workflow is:
  * Algorithm generates `Offsprings`, which contains information about the
  "genes". `PopulationConverter` converts them to `TrialSuggestions`.
  * User evaluates `TrialSuggestions` and obtains completed `Trial` objects.
  `PopulationConverter` converts them to `Population`, which is what the
  algorithm receives.
  * This template does not dictate the nature of these conversions. They can be
  stochastic, deterministic, bijective, injective, inverse of each other or not.

Why are Offsprings and Population serializable?
  * `Offsprings` is serializable so that the original genes can be preserved
  through the conversions.
  * `Population` is serializable so that the algorithm can serialize its state.

A canonical evolutionary algorithm consists of the following operations:
  * `Sampler` generates fresh `Offsprings` independent of `Population`.
    It is generally used for sampling an initial batch of `Offsprings`.
  * `Survival` selects a subset of `Population` to keep.
  * `Mutation` mutates `Population` to generate new `Offsprings`.

To use the provided template, one should implement a subclass of `Offsprings`,
`Population`, `PopulationConverter`, `Sampler`, `Survival`, and `Mutation`.
"""

import abc
from typing import Callable, Generic, Optional, Sequence, TypeVar, Union

from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier.interfaces import serializable

_OffspringsType = TypeVar('_OffspringsType', bound=serializable.Serializable)


class Sampler(Generic[_OffspringsType], abc.ABC):
  """Creates new offsprings, typically at the start of evolutionary strategy."""

  @abc.abstractmethod
  def sample(self, count: int) -> _OffspringsType:
    """Generate offsprings."""


# Temporary typevar, used for defininig Population interface.
_P = TypeVar('_P', bound=serializable.Serializable)


class Population(serializable.Serializable, abc.ABC):
  """Typically contains genes (x-values) and scores (y-values).

  The supported operations are similar to python sequence, except that slicing
  always results in a `Population` of length 1.
  """

  @abc.abstractmethod
  def __len__(self) -> int:
    pass

  @abc.abstractmethod
  def __getitem__(
      self: _P,
      index: Union[int, slice],
     
  ) -> _P:
    pass

  @abc.abstractmethod
  def __add__(self: _P, other: _P) -> _P:
    pass


# Actual Typevar. Covariant with `Population`.
_PopulationType = TypeVar('_PopulationType')


class PopulationConverter(abc.ABC, Generic[_PopulationType, _OffspringsType]):

  @abc.abstractmethod
  def to_population(self,
                    completed: Sequence[vz.CompletedTrial]) -> _PopulationType:
    """Adds trials to the population."""

  @abc.abstractmethod
  def to_suggestions(
      self, offsprings: _OffspringsType) -> Sequence[vz.TrialSuggestion]:
    """Converts offsprings to suggestions."""


class Survival(abc.ABC, Generic[_PopulationType]):

  @abc.abstractmethod
  def select(self, population: _PopulationType) -> _PopulationType:
    """Survival mechanism."""


class Mutation(abc.ABC, Generic[_PopulationType, _OffspringsType]):

  @abc.abstractmethod
  def mutate(self, population: _PopulationType, count: int) -> _OffspringsType:
    """Generate offsprings, whose contain count many offsprings."""


class CanonicalEvolutionDesigner(vza.PartiallySerializableDesigner,
                                 Generic[_PopulationType, _OffspringsType]):
  """Evolution algorithm template."""

  def __init__(
      self,
      converter: PopulationConverter[_PopulationType, _OffspringsType],
      sampler: Sampler[_OffspringsType],
      survival: Survival[_PopulationType],
      *,
      adaptation: Mutation[_PopulationType, _OffspringsType],
      adaptation_callable: Optional[
          Callable[[int], Mutation[_PopulationType, _OffspringsType]]
      ] = None,
      initial_population: Optional[_PopulationType] = None,
      first_survival_after: Optional[int] = None,
      population_size: int = 50,
  ):
    """Init.

    Args:
      converter:
      sampler:
      survival:
      adaptation: Default adaptation. Will be overwrote if adaptation_callable
        is specified.
      adaptation_callable: Adapation as a function of number of Trials seen.
      initial_population: The initial population to seed the evolution.
      first_survival_after: Apply the survival step after observing this many
        trials. If unset, it defaults to twice the `population_size`.
      population_size: Survival steps reduce the population to this size.
    """
    self._survival = survival
    self._adaptation = adaptation
    self._converter = converter
    self._sampler = sampler
    self._population_size = population_size
    self._adaptation_callable = adaptation_callable
    self._first_survival_after = (
        first_survival_after or self._population_size * 2
    )
    self._num_trials_seen = 0
    self._population = initial_population or converter.to_population([])

  @property
  def converter(self) -> PopulationConverter[_PopulationType, _OffspringsType]:
    return self._converter

  @property
  def population(self) -> _PopulationType:
    return self._population

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    count = count or self._population_size
    if self._num_trials_seen < self._first_survival_after:
      return self._converter.to_suggestions(self._sampler.sample(count))

    if self._adaptation_callable is not None:
      adaptation = self._adaptation_callable(self._num_trials_seen)
    else:
      adaptation = self._adaptation

    suggestions = self._converter.to_suggestions(
        adaptation.mutate(self._population, count)
    )
    return suggestions

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    completed = completed.trials
    self._num_trials_seen += len(completed)
    candidates = self._population + self._converter.to_population(completed)
    self._population = self._survival.select(candidates)

  def load(self, metadata: vz.Metadata):
    self._population = type(self._population).recover(metadata)

  def dump(self) -> vz.Metadata:
    return self._population.dump()
