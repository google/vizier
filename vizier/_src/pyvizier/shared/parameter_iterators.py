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

"""Tools for iterating through a collection of Parameter(config)s."""

from typing import Iterator, Generator
import copy


from vizier._src.pyvizier.shared.parameter_config import ParameterConfig
from vizier._src.pyvizier.shared.parameter_config import SearchSpace
from vizier._src.pyvizier.shared.trial import ParameterDict
from vizier._src.pyvizier.shared.trial import ParameterValueTypes


class SequentialParameterBuilder(Iterator[ParameterConfig]):
  """Builds a ParameterDict by choosing one parameter value at a time.

  Example usage:
    from vizier import pyvizier as vz

    def decide_value(pc: vz.ParameterConfig) -> vz.ParameterValueTypes:
      ...

    search_space : vz.SearchSpace
    builder = SequentialParameterBuilder(search_space)

    for pc in builder:
      builder.choose_value(decide_value(pc))

    assert isinstance(builder.parameters, vz.ParameterDict)
  """

  def __init__(self,
               search_space: SearchSpace,
              
               *,
               traverse_order: str = 'dfs'):
    """Init.

    See the class pydoc for more details.

    Args:
      search_space: Search space to iterate over.
      traverse_order: 'dfs' or 'bfs'.
    """
    self._parameters = ParameterDict()
    self._traverse_order = traverse_order
    self._gen = self._coroutine(search_space)
    self._next = next(self._gen)
    self._stop_iteration = None

  def _coroutine(
      self, search_space: SearchSpace
  ) -> Generator[ParameterConfig, ParameterValueTypes, None]:
    search_space = copy.deepcopy(search_space)
    while search_space.parameters:
      parameter_config = search_space.parameters[0]
      value = yield parameter_config
      subspace = search_space.get(
          parameter_config.name).get_subspace_deepcopy(value)
      search_space.pop(parameter_config.name)

      if self._traverse_order == 'bfs':
        for child_parameter in subspace.parameters:
          search_space.add(child_parameter)
      else:
        for parameter in search_space.parameters:
          subspace.add(parameter)
        search_space = subspace
      self._parameters[parameter_config.name] = value

  def __next__(self) -> ParameterConfig:
    if self._stop_iteration is not None:
      raise self._stop_iteration
    return self._next

  def choose_value(self, value: ParameterValueTypes) -> None:
    """Choose the value for the last ParameterConfig."""
    try:
      self._next = self._gen.send(value)
    except StopIteration as e:
      self._stop_iteration = e

  @property
  def parameters(self) -> ParameterDict:
    """Parameters chosen so far.

    WARNING: Do not mutate the dict until this Iterator is exhausted.
    """
    return self._parameters
