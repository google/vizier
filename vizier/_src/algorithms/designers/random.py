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

"""Random designer."""

from typing import Any, Optional, Sequence

import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier.pyvizier import converters


class RandomDesigner(vza.Designer):
  """Random designer.

  This designer uniformly samples from the scaled search space.
  """

  def __init__(self,
               search_space: vz.SearchSpace,
               *,
               dtype: Any = np.float64,
               seed: Optional[int] = None):
    """Init.

    Args:
      search_space: Must be a flat search space.
      dtype:
      seed: Any valid seed for np.random.RandomState.
    """
    if search_space.is_conditional:
      # TODO: Add conditional sampling case.
      raise ValueError(
          f'This designer {self} does not support conditional search.')

    def create_input_converter(pc):
      # Setting 'max_discrete_indices' to not continuify DISCRETE parameters.
      return converters.DefaultModelInputConverter(
          pc, scale=True, max_discrete_indices=np.inf, float_dtype=dtype
      )

    self._converter = converters.DefaultTrialConverter(
        [create_input_converter(pc) for pc in search_space.parameters])

    self._rng = np.random.RandomState(seed)

  @classmethod
  def from_problem(
      cls, problem: vz.ProblemStatement, seed: Optional[int] = None
  ):
    """For wrapping via `PartiallySerializableDesignerPolicy`."""
    return RandomDesigner(problem.search_space, seed=seed)

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    pass

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    """Make new suggestions.

    Args:
      count: Makes best effort to generate this many suggestions. If None,
        suggests as many as the algorithm wants.

    Returns:
      New suggestions.
    """
    count = count or 1
    sample = dict()
    for name, spec in self._converter.output_specs.items():
      if spec.type == converters.NumpyArraySpecType.DISCRETE:
        sample[name] = self._rng.random_integers(
            spec.bounds[0], spec.bounds[1] - spec.num_oovs, size=[count, 1])
      elif spec.type == converters.NumpyArraySpecType.CONTINUOUS:
        # Trial-Numpy converter was configured to scale values to [0, 1].
        # Sample uniformly in that range and convert back to Trials.
        sample[name] = self._rng.random([count, 1])
      else:
        raise ValueError(f'Unsupported type: {spec.type} in {spec}')
    return [
        vz.TrialSuggestion(p) for p in self._converter.to_parameters(sample)
    ]
