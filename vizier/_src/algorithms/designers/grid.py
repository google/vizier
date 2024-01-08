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

"""Grid Search Designer which searches over a discretized grid of Trial parameter values."""

import copy
import random
from typing import Dict, List, Optional, Sequence

from absl import logging
import attrs
import numpy as np
from vizier import algorithms
from vizier import pyvizier
from vizier.interfaces import serializable
from vizier.pyvizier import converters


GridValues = Dict[str, List[pyvizier.ParameterValue]]


@attrs.define(auto_attribs=False, init=False)
class GridSearchDesigner(algorithms.PartiallySerializableDesigner):
  """Grid Search designer.

  This designer searches over a grid of hyper-parameter values.

  NOTE: The grid search index (i.e. which grid point to output) is based the
  number of suggestions created so far (regardless of completion or not). This
  means the class must be wrapped via `PartiallySerializableDesignerPolicy` for
  use in Pythia, thus requiring load/dump implementations.
  """

  _unshuffled_grid_values: GridValues = attrs.field()
  _grid_values: GridValues = attrs.field()
  _current_index: int = attrs.field()
  _shuffle_seed: Optional[int] = attrs.field()
  _double_grid_resolution: int = attrs.field()

  _metadata_ns: str = 'grid'  # class-level constant.

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
    self._shuffle_seed = shuffle_seed
    self._double_grid_resolution = double_grid_resolution
    self._current_index = 0

    # Creates unshuffled grid values for every parameter. This is just a
    # template for creating the potentially shuffled self._grid_values to be
    # used in suggest().
    self._unshuffled_grid_values = {}
    for parameter_config in self._search_space.parameters:
      self._unshuffled_grid_values[parameter_config.name] = (
          self._grid_points_from_parameter_config(parameter_config)
      )

    # Set true grid values to be used during suggest calls.
    self._grid_values = self._maybe_shuffled_grid_values(self._shuffle_seed)

  @classmethod
  def from_problem(
      cls,
      problem: pyvizier.ProblemStatement,
      seed: Optional[int] = None,
  ):
    """For wrapping via `PartiallySerializableDesignerPolicy`."""
    return GridSearchDesigner(problem.search_space, seed)

  def update(
      self,
      completed: algorithms.CompletedTrials,
      all_active: algorithms.ActiveTrials,
  ) -> None:
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
    metadata = metadata.ns(self._metadata_ns)
    try:
      current_index = int(metadata['current_index'])
      none_or_int = metadata['shuffle_seed']
      shuffle_seed = None if (none_or_int == 'None') else int(none_or_int)
      logging.info('Restored shuffle seed: %s', shuffle_seed)
    except (KeyError, ValueError) as e:
      raise serializable.HarmlessDecodeError() from e

    self._current_index = current_index
    self._shuffle_seed = shuffle_seed
    self._grid_values = self._maybe_shuffled_grid_values(self._shuffle_seed)

  def dump(self) -> pyvizier.Metadata:
    """Dump the current index."""
    metadata = pyvizier.Metadata()
    metadata.ns(self._metadata_ns)['current_index'] = str(self._current_index)
    metadata.ns(self._metadata_ns)['shuffle_seed'] = str(self._shuffle_seed)
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

  def _maybe_shuffled_grid_values(
      self, shuffle_seed: Optional[int]
  ) -> GridValues:
    grid_values = copy.deepcopy(self._unshuffled_grid_values)

    # Shuffle the grid if specified.
    if shuffle_seed is not None:
      rng = random.Random(shuffle_seed)
      # Shuffle dict keys.
      shuffled_items = list(grid_values.items())
      rng.shuffle(shuffled_items)
      grid_values = dict(shuffled_items)
      # Shuffle dict value lists.
      for param_name in grid_values:
        rng.shuffle(grid_values[param_name])

    return grid_values
