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

"""Quasi-random designer."""

import collections
import math
import sys
import time
from typing import Optional, Sequence

import numpy as np
from scipy.stats import qmc
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier.pyvizier import converters


class QuasiRandomDesigner(vza.PartiallySerializableDesigner):
  """Sample points using quasi-random Halton algorithm."""

  def __init__(
      self,
      search_space: vz.SearchSpace,
      *,
      skip_points: int = 1000,
      seed: Optional[int] = None,
  ):
    """Init.

    Args:
      search_space: Must be a flat search space.
      skip_points: If positive, then these first points in the sequence are
        discarded in order to avoid unwanted correlations.
      seed: Random seed.
    """
    if search_space.is_conditional:
      raise ValueError(
          f'This designer {self} does not support conditional search.'
      )

    def _create_input_converter(pc):
      return converters.DefaultModelInputConverter(
          pc,
          scale=True,
          max_discrete_indices=sys.maxsize,
          float_dtype=np.float64,
      )

    self._converter = converters.DefaultTrialConverter(
        [_create_input_converter(pc) for pc in search_space.parameters]
    )

    for spec in self._converter.output_specs.values():
      if spec.type not in [
          converters.NumpyArraySpecType.CONTINUOUS,
          converters.NumpyArraySpecType.DISCRETE,
      ]:
        raise ValueError(f'Unsupported type: {spec.type} in {spec}')
      if spec.num_dimensions != 1:
        raise ValueError(
            'Multi-dimensional discrete types are unsuppored. Received spec: %s'
            % spec
        )
    self._seed = seed if seed is not None else np.int32(time.time())
    self._halton = qmc.Halton(
        d=len(self._converter.output_specs),
        seed=self._seed,
    )
    self._halton.fast_forward(skip_points)
    self._skip_points = skip_points
    self._output_specs = tuple(self._converter.output_specs.values())

  @classmethod
  def from_problem(
      cls, problem: vz.ProblemStatement, seed: Optional[int] = None
  ):
    """For wrapping via `PartiallySerializableDesignerPolicy`."""
    return QuasiRandomDesigner(problem.search_space, seed=seed)

  def load(self, metadata: vz.Metadata) -> None:
    """Loads designer's state from metadata."""
    self._seed = int(metadata.ns('quasi_random')['seed'])
    self._halton = qmc.Halton(
        d=len(self._converter.output_specs),
        seed=int(metadata.ns('quasi_random')['seed']),
    )
    self._skip_points = int(metadata.ns('quasi_random')['skip_points'])
    # Skip forward to where the previous Halton sequence stopped.
    self._halton.fast_forward(self._skip_points)

  def dump(self) -> vz.Metadata:
    """Dumps the designer's state."""
    metadata = vz.Metadata()
    metadata.ns('quasi_random')['skip_points'] = str(self._skip_points)
    metadata.ns('quasi_random')['seed'] = str(self._seed)
    return metadata

  def _generate_discrete_point(
      self, spec: converters.NumpyArraySpec, halton_value: float
  ) -> int:
    """Generate a discrete parameter value from a Halton value."""
    # +1 because the bounds are inclusive on both ends.
    num_discrete_options = spec.bounds[1] - spec.bounds[0] + 1 - spec.num_oovs
    # Get a number in [0,  num_discrete_options].
    halton_value *= num_discrete_options
    # Get an integer between 0 and num_discrete_options-1 (inclusive).
    halton_value = int(math.floor(halton_value))
    return halton_value + int(spec.bounds[0])

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    pass

  def suggest(
      self, count: Optional[int] = None
  ) -> Sequence[vz.TrialSuggestion]:
    """Suggest new suggestions, taking into account `count`."""
    count = count or 1
    sample = collections.defaultdict(list)
    for _ in range(count):
      # Generate random values in range of [0.0, 1.0].
      halton_values = self._halton.random(len(self._output_specs))[0]
      # Increment 'skip_points' to reflect designer's state.
      self._skip_points += len(self._output_specs)
      for dimension_index, spec in enumerate(self._output_specs):
        halton_value = halton_values[dimension_index]
        if spec.type == converters.NumpyArraySpecType.CONTINUOUS:
          # Samples are in [0,1]; Relies on the converter to scale back to
          # original scale once parameters are created.
          sample[spec.name].append(np.float64(halton_value))
        elif spec.type == converters.NumpyArraySpecType.DISCRETE:
          # Converter expects an integer for discrete/categorical parameters.
          sample[spec.name].append(
              np.int64(self._generate_discrete_point(spec, halton_value))
          )
        else:
          # Only CONTINUOUS and DISCRETE are supported (doesn't support ONEHOT).
          raise ValueError(
              f'Unsupported spec type: {spec.type}. {self._converter} should be'
              ' configured to return CONTINUOUS or DISCRETE specs only.'
          )
    sample = {
        name: np.expand_dims(np.asarray(elements), axis=-1)
        for (name, elements) in sample.items()
    }
    # Convert the samples back to parameters in the original search space.
    return [
        vz.TrialSuggestion(p) for p in self._converter.to_parameters(sample)
    ]
