# Copyright 2022 Google LLC.
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

"""Grid Search Designer which searches over a discretized grid of Trial parameter values.
"""
from typing import List, Mapping, Optional, Sequence
import numpy as np
from vizier import algorithms
from vizier import pyvizier

# TODO: Make this a user settable parameter.
GRID_RESOLUTION = 10  # For double parameters.


def _grid_points_from_parameter_config(
    parameter_config: pyvizier.ParameterConfig
) -> List[pyvizier.ParameterValue]:
  """Produces grid points from a parameter_config."""
  if parameter_config.type == pyvizier.ParameterType.DOUBLE:
    min_value, max_value = parameter_config.bounds

    if min_value == max_value:
      return [pyvizier.ParameterValue(value=min_value)]

    grid_scalars = np.linspace(min_value, max_value, num=GRID_RESOLUTION)
    return [pyvizier.ParameterValue(value=value) for value in grid_scalars]

  elif parameter_config.type == pyvizier.ParameterType.INTEGER:
    min_value, max_value = parameter_config.bounds
    return [
        pyvizier.ParameterValue(value=value)
        for value in range(min_value, max_value + 1)
    ]

  elif parameter_config.type == pyvizier.ParameterType.CATEGORICAL:
    return [
        pyvizier.ParameterValue(value=value)
        for value in parameter_config.feasible_values
    ]

  elif parameter_config.type == pyvizier.ParameterType.DISCRETE:
    return [
        pyvizier.ParameterValue(value=value)
        for value in parameter_config.feasible_values
    ]
  else:
    raise ValueError(
        f'ParameterConfig type is not one of the supported primitives for ParameterConfig: {parameter_config}'
    )


def _make_grid_values(
    search_space: pyvizier.SearchSpace
) -> Mapping[str, List[pyvizier.ParameterValue]]:
  """Makes the grid values for every parameter."""
  grid_values = {}
  for parameter_config in search_space.parameters:
    grid_values[parameter_config.name] = _grid_points_from_parameter_config(
        parameter_config)
  return grid_values


def _make_grid_search_parameters(
    indices: Sequence[int],
    search_space: pyvizier.SearchSpace) -> List[pyvizier.ParameterDict]:
  """Selects the specific parameters from an index and study_spec based on the natural ordering over a Cartesian Product.

  This is looped over a sequence of indices. For a given `index`, this is
  effectively equivalent to itertools.product(list_of_lists)[index].

  Args:
    indices: Index over Cartesian Product.
    search_space: SearchSpace to produce the Cartesian Product. Ordering decided
      alphabetically over the parameter names.

  Returns:
    ParameterDict for a trial suggestion.
  """
  # TODO: Add conditional sampling case.
  grid_values = _make_grid_values(search_space)
  parameter_dicts = []
  for index in indices:
    parameter_dict = pyvizier.ParameterDict()
    temp_index = index
    for p_name in grid_values:
      p_length = len(grid_values[p_name])
      p_index = temp_index % p_length
      parameter_dict[p_name] = grid_values[p_name][p_index]
      temp_index = temp_index // p_length
    parameter_dicts.append(parameter_dict)
  return parameter_dicts


class GridSearchDesigner(algorithms.PartiallySerializableDesigner):
  """Grid Search designer.

  This designer searches over a grid of hyper-parameter values.

  NOTE: The grid search index (i.e. which grid point to output) is calculated
  according to the number of suggestions created so far (regardless of
  completion or not). This means the class must be wrapped via
  `PartiallySerializableDesignerPolicy` for use in Pythia, thus requiring
  load/dump implementations.
  """

  def __init__(self, search_space: pyvizier.SearchSpace):
    """Init.

    Args:
      search_space: Must be a flat search space.
    """
    if search_space.is_conditional:
      raise ValueError(
          f'This designer {self} does not support conditional search.')
    self._search_space = search_space
    self._current_index = 0

  @classmethod
  def from_problem(cls, problem: pyvizier.ProblemStatement):
    """For wrapping via `PartiallySerializableDesignerPolicy`."""
    return GridSearchDesigner(problem.search_space)

  def update(self, _) -> None:
    pass

  def suggest(
      self, count: Optional[int] = None) -> Sequence[pyvizier.TrialSuggestion]:
    """Make new suggestions.

    Args:
      count: Makes best effort to generate this many suggestions. If None,
        suggests as many as the algorithm wants.

    Returns:
      New suggestions.
    """
    count = count or 1
    parameter_dicts = _make_grid_search_parameters(
        range(self._current_index, self._current_index + count),
        self._search_space)
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
