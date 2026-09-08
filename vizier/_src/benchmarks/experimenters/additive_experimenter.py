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

"""Additive experimenter to sum up experimenters and concatenate their search spaces."""

import copy
from typing import Sequence
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter


class AdditiveExperimenter(experimenter.Experimenter):
  """AdditiveExperimenter take a list of experimenters and concatenates their search space.

  Parameters from the input experimenters are given a suffix corresponding to
  which experiment they came from. The suffix for each experiment is unique.
  This ensures that parameter names do not collide.

  Right now, this code drops 'metric information' and 'meta-data' from the
  problem statements. I (Tim Chu) don't yet understand what those are, and
  will add this functionality in later.
  """

  def __init__(self, experimenters: list[experimenter.Experimenter]):
    # Why aren't there property accessor methods for search_space in class
    # ProblemStatement?
    search_spaces = [
        experimenter.problem_statement().search_space
        for experimenter in experimenters
    ]
    search_space_suffixes = ["_" + str(i) for i, _ in enumerate(experimenters)]
    combined_search_space = pyvizier.SearchSpace()
    for i, search_space in enumerate(search_spaces):
      search_space_suffix = search_space_suffixes[i]
      param_names = search_space.parameter_names
      for param_name in param_names:
        new_param_name = param_name + search_space_suffix
        # Is deep copy going to work on param_config?
        new_param_config = copy.deepcopy(search_space.get(param_name))
        # Setting name via private variable: is this okay?
        new_param_config._name = new_param_name
        combined_search_space.add(new_param_config)
    self._problem_statement = pyvizier.ProblemStatement()

  def problem_statement(self) -> pyvizier.ProblemStatement:
    return copy.deepcopy(self._problem_statement)

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    # How is this supposed to work?
    pass
