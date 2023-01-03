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

"""Quasi-random designer."""

import collections
import math
import random
import sys
from typing import List, Optional, Sequence, Iterable

import attr
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier.interfaces import serializable
from vizier.pyvizier import converters

################################################################################
# PyTypes
NumpyArraySpec = converters.NumpyArraySpec
NumpyArraySpecType = converters.NumpyArraySpecType
ScaleType = vz.ScaleType
################################################################################


def _is_prime(n: int) -> bool:
  """Check if `n` is a prime number."""
  return all(n % i != 0 for i in range(2, int(n**0.5) + 1)) and n != 2


def _generate_primes(n: int) -> List[int]:
  """Generate all primes less than `n` (except 2) using the Sieve of Sundaram."""
  if n <= 2:
    raise ValueError('n has to be larger than 2: %s' % n)
  half_m1 = int((n - 2) / 2)
  sieve = [0] * (half_m1 + 1)
  for outer in range(1, half_m1 + 1):
    inner = outer
    while outer + inner + 2 * outer * inner <= half_m1:
      sieve[outer + inner + (2 * outer * inner)] = 1
      inner += 1
  return [2 * i + 1 for i in range(1, half_m1 + 1) if sieve[i] == 0]


def _init_primes(num_dimensions: int) -> List[int]:
  """Get list of primes for all parameters in the search space."""
  primes = []
  prime_attempts = 1
  while len(primes) < num_dimensions + 1:
    primes = _generate_primes(1000 * prime_attempts)
    prime_attempts += 1
  primes = primes[-num_dimensions - 1:-1]
  return primes


@attr.define(init=False, kw_only=True)
class _HaltonSequence(serializable.PartiallySerializable):
  """Encapsulates the generation of a scrambled halton sequence.

  Specifically, this class is a fork inspired by the implementation of
  scrambled Halton sequence of quasi-random numbers (by Google):

  https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/halton.py

  """

  _num_dimensions: int = attr.field(validator=attr.validators.instance_of(int))

  _skip_points: int = attr.field(validator=attr.validators.instance_of(int))

  _num_points_generated: int = attr.field(
      validator=attr.validators.instance_of(int))

  _primes: List[int] = attr.field(
      init=True,
      converter=list,
      validator=attr.validators.deep_iterable(
          member_validator=attr.validators.instance_of(int),
          iterable_validator=attr.validators.instance_of(Iterable)))

  _scramble: bool = attr.field(
      default=False, validator=attr.validators.instance_of(bool), kw_only=True)

  def __init__(self,
               num_dimensions: int,
               *,
               skip_points: int,
               num_points_generated: int = 0,
               primes_override: Optional[List[int]] = None,
               scramble: bool = False):
    """Create a Halton sequence generator.

    Args:
      num_dimensions: Number of dimensions for each point in the seqeunce. This
        corresponds to the number of parameters in the Vizier search space.
      skip_points: The number of initial points that should be skipped before
        the first point is returned.
      num_points_generated: Number of points that have already been generated.
      primes_override: If supplied, use these primes to seed each dimension of
        the Halton sequence. This is useful for testing. NOTE: These values are
        not validated, so it is the responsibility of the user to supply
        legitimate primes.
      scramble: If True, will scramble the resulting Halton sequence. This is
        intended to be used for testing.

    Returns:
      A HaltonSequence object.
    """
    if skip_points < 0:
      raise ValueError('skip_points must be non-negative: %s' % skip_points)

    if primes_override:
      if len(primes_override) != num_dimensions:
        raise ValueError(
            'Expected len(primes_overrides) and num_dimensions to '
            f'be the same size. len(primes_overrides): {len(primes_override)},'
            f'num_dimensions: {num_dimensions}')
      primes = primes_override
    else:
      primes = _init_primes(num_dimensions)

    self.__attrs_init__(
        num_dimensions=num_dimensions,
        skip_points=skip_points,
        num_points_generated=num_points_generated,
        primes=primes,
        scramble=scramble)

  def load(self, metadata: vz.Metadata) -> None:
    self._num_points_generated = int(
        metadata.ns('halton')['num_points_generated'])

  def dump(self) -> vz.Metadata:
    metadata = vz.Metadata()
    metadata.ns('halton')['num_points_generated'] = str(
        self._num_points_generated)
    return metadata

  def _get_scrambled_halton_value(self, index: int, base: int) -> float:
    """Get a scrambled Halton value for a given `index`, seeded by `base`."""
    if not _is_prime(base):
      raise ValueError('base is not prime: %s' % base)

    result = 0.0
    base_rec = 1.0 / base
    f = base_rec
    i = index + 1  # For index 0 we want 1/base returned, not 0.

    # Use a fixed seed to generate the permutation in a deterministic way.
    if self._scramble:
      local_random = random.Random(base)
      permutation = list(range(1, base))
      local_random.shuffle(permutation)
      permutation = [0] + permutation
    while i > 0:
      i, mod = divmod(i, base)
      if self._scramble:
        result += f * permutation[mod]
      else:
        result += f * mod
      f *= base_rec

    if (0.0 > result) or (result > 1.0):
      raise ValueError(
          'Something wrong has happened; halton_value should be within [0, 1]: %f'
          % result)
    return result

  def get_next_list(self) -> List[float]:
    """Get the next list in a sequence seeded by `primes`.

    This implementation and its associated unit tests are inspired by another
    implementation from Google.

    https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/halton.py

    Returns:
      An sublist of the Halton sequence. Every value in the list should be
      within [0,1].
    """
    index = self._num_points_generated + self._skip_points
    halton_list = [
        self._get_scrambled_halton_value(index, prime) for prime in self._primes
    ]
    self._num_points_generated += 1
    return halton_list


class QuasiRandomDesigner(vza.PartiallySerializableDesigner):
  """Sample points using quasi-random search from the scaled search space.

  This implementation uses a scrambled Halton sequence.
  """

  def __init__(self, search_space: vz.SearchSpace, *, skip_points: int = 100):
    """Init.

    Args:
      search_space: Must be a flat search space.
      skip_points: If positive, then these first points in the sequence are
        discarded in order to avoid unwanted correlations.
    """
    if search_space.is_conditional:
      raise ValueError(
          f'This designer {self} does not support conditional search.')

    def create_input_converter(pc):
      return converters.DefaultModelInputConverter(
          pc,
          scale=True,
          max_discrete_indices=sys.maxsize,
          float_dtype=np.float64)

    self._converter = converters.DefaultTrialConverter(
        [create_input_converter(pc) for pc in search_space.parameters])

    for spec in self._converter.output_specs.values():
      if spec.type not in [
          NumpyArraySpecType.CONTINUOUS, NumpyArraySpecType.DISCRETE
      ]:
        raise ValueError(f'Unsupported type: {spec.type} in {spec}')
      if spec.num_dimensions != 1:
        raise ValueError('Multi-dimensional discrete types are unsuppored. '
                         'Received spec: %s' % spec)

    self._halton_generator = _HaltonSequence(
        len(search_space.parameters),
        skip_points=skip_points,
        num_points_generated=0,
        scramble=False)

    self._output_specs = tuple(self._converter.output_specs.values())

  @classmethod
  def from_problem(cls, problem: vz.ProblemStatement):
    """For wrapping via `PartiallySerializableDesignerPolicy`."""
    return QuasiRandomDesigner(problem.search_space)

  def load(self, metadata: vz.Metadata) -> None:
    self._halton_generator.load(metadata)

  def dump(self) -> vz.Metadata:
    return self._halton_generator.dump()

  def _generate_discrete_point(self, spec: NumpyArraySpec,
                               halton_value: float) -> int:
    """Generate a discrete parameter value from a Halton value."""

    # +1 because the bounds are inclusive on both ends.
    num_discrete_options = spec.bounds[1] - spec.bounds[0] + 1 - spec.num_oovs
    # Get a number in [0,  num_discrete_options].
    halton_value *= num_discrete_options
    # Get an integer between 0 and num_discrete_options-1 (inclusive).
    halton_value = int(math.floor(halton_value))
    return halton_value + int(spec.bounds[0])

  def update(self, _) -> None:
    pass

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    """Suggest new suggestions, taking into account `count`."""
    count = count or 1

    sample = collections.defaultdict(list)
    for _ in range(count):
      # Dimension of halton_list is [P], where P is number of primes used.
      halton_list = self._halton_generator.get_next_list()
      for dimension_index, spec in enumerate(self._output_specs):
        # Only CONTINUOUS and DISCRETE are supported.
        halton_value = halton_list[dimension_index]
        if spec.type == NumpyArraySpecType.CONTINUOUS:
          # Trial-Numpy converter was configured to scale values to [0, 1].
          # We sample from that range and rely on it scaled correctly when the
          # Trials are created.
          # halton_value is also within [0, 1].
          sample[spec.name].append(np.float64(halton_value))
        elif spec.type == NumpyArraySpecType.DISCRETE:
          # Trial-Numpy converter expects an integer for discrete/categorical
          # parameters.
          sample[spec.name].append(
              np.int64(self._generate_discrete_point(spec, halton_value)))
        else:
          raise ValueError(
              f'Unsupported spec type: {spec.type}. self._converter should be configured to return CONTINUOUS or DISCRETE specs only.'
          )
    sample = {
        name: np.expand_dims(np.asarray(elements), axis=-1)
        for (name, elements) in sample.items()
    }
    return [
        vz.TrialSuggestion(p) for p in self._converter.to_parameters(sample)
    ]
