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

"""Grid Search Designer which searches over a discretized grid of Trial parameter values."""
import random
from typing import List, Optional, Sequence
import numpy as np

from vizier import algorithms
from vizier import pyvizier
from vizier.pyvizier import converters


class GridSearchDesigner(algorithms.PartiallySerializableDesigner):
  """Grid Search designer.

  This designer searches over a grid of hyper-parameter values.

  NOTE: The grid search index (i.e. which grid point to output) is based the
  number of suggestions created so far (regardless of completion or not). This
  means the class must be wrapped via `PartiallySerializableDesignerPolicy` for
  use in Pythia, thus requiring load/dump implementations.
  """

  def __init__(
      self,
      search_space: pyvizier.SearchSpace,
      shuffle_seed: Optional[int] = None,
      *,
      double_grid_resolution: int = 10,
  ):
    """Init.

    Args:
      search_space: Must be a flat search space.
      shuffle_seed: Whether to shuffle the grid ordering. If None, uses the
        given ordering from original search space.
      double_grid_resolution: Number of grid points for DOUBLE parameters.
    """
    if search_space.is_conditional:
      raise ValueError(
          f'This designer {self} does not support conditional search.'
      )
    self._search_space = search_space
    self._double_grid_resolution = double_grid_resolution
    self._current_index = 0

    # Makes the grid values for every parameter.
    self._grid_values = {}
    for parameter_config in self._search_space.parameters:
      self._grid_values[parameter_config.name] = (
          self._grid_points_from_parameter_config(parameter_config)
      )

    # Shuffle the grid if specified.
    if shuffle_seed is not None:
      rng = random.Random(shuffle_seed)
      # Shuffle dict keys.
      shuffled_items = list(self._grid_values.items())
      rng.shuffle(shuffled_items)
      self._grid_values = dict(shuffled_items)
      # Shuffle dict value lists.
      for param_name in self._grid_values:
        rng.shuffle(self._grid_values[param_name])

  @classmethod
  def from_problem(
      cls,
      problem: pyvizier.ProblemStatement,
      shuffle_seed: Optional[int] = None,
  ):
    """For wrapping via `PartiallySerializableDesignerPolicy`."""
    return GridSearchDesigner(problem.search_space, shuffle_seed)

  def update(self, _) -> None:
    pass

  def suggest(
      self, count: Optional[int] = None
  ) -> Sequence[pyvizier.TrialSuggestion]:
    """Make new suggestions.

    Args:
      count: Makes best effort to generate this many suggestions. If None,
        suggests as many as the algorithm wants.

    Returns:
      New suggestions.
    """
    count = count or 1

    parameter_dicts = []
    for index in range(self._current_index, self._current_index + count):
      # Use index to select parameters via Cartesian Product ordering.
      # Effectively equivalent to itertools.product(list_of_lists)[index]`,
      # without the memory blowup.
      parameter_dict = pyvizier.ParameterDict()
      temp_index = index
      for p_name in self._grid_values:
        p_length = len(self._grid_values[p_name])
        p_index = temp_index % p_length
        parameter_dict[p_name] = self._grid_values[p_name][p_index]
        temp_index = temp_index // p_length
      parameter_dicts.append(parameter_dict)

    self._current_index += len(parameter_dicts)
    return [pyvizier.TrialSuggestion(parameters=p) for p in parameter_dicts]

  def load(self, metadata: pyvizier.Metadata) -> None:
    """Load the current index."""
    self._current_index = int(metadata.ns('grid')['current_index'])

  def dump(self) -> pyvizier.Metadata:
    """Dump the current index."""
    metadata = pyvizier.Metadata()
    metadata.ns('grid')['current_index'] = str(self._current_index)
    return metadata

  def _grid_points_from_parameter_config(
      self,
      parameter_config: pyvizier.ParameterConfig,
  ) -> List[pyvizier.ParameterValue]:
    """Produces grid points from a parameter_config."""
    if parameter_config.type == pyvizier.ParameterType.DOUBLE:
      min_value, max_value = parameter_config.bounds
      if min_value == max_value:
        return [pyvizier.ParameterValue(value=min_value)]

      converter = converters.DefaultModelInputConverter(
          parameter_config, scale=True
      )
      grid_scalars = np.linspace(0.0, 1.0, num=self._double_grid_resolution)
      return converter.to_parameter_values(grid_scalars)  # pytype:disable=bad-return-type

    elif parameter_config.type == pyvizier.ParameterType.INTEGER:
      min_value, max_value = parameter_config.bounds
      return [
          pyvizier.ParameterValue(value=value)
          for value in range(min_value, max_value + 1)
      ]

    elif parameter_config.type in [
        pyvizier.ParameterType.DISCRETE,
        pyvizier.ParameterType.CATEGORICAL,
    ]:
      return [
          pyvizier.ParameterValue(value=value)
          for value in parameter_config.feasible_values
      ]

    else:
      raise ValueError(
          'ParameterConfig type is not one of the supported primitives for'
          f' ParameterConfig: {parameter_config}'
      )
