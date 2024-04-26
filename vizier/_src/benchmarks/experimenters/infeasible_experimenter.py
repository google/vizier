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

"""Experimenters which induce infeasibility behaviors."""

import copy
import json
import random
from typing import Sequence
import attrs
import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter
from vizier.pyvizier import converters


@attrs.define
class HashingInfeasibleExperimenter(experimenter.Experimenter):
  """Simulates randomly selected (deterministic) infeasibility function."""

  exptr: experimenter.Experimenter = attrs.field()
  infeasible_prob: float = attrs.field(default=0.2, kw_only=True)
  seed: int = attrs.field(default=0, kw_only=True)

  def __attrs_post_init__(self):
    self._problem = copy.deepcopy(self.exptr.problem_statement())

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    metric_configs = self._problem.metric_information
    for suggestion in suggestions:
      if self._is_infeasible(suggestion.parameters):
        suggestion.complete(
            vz.Measurement(metrics={mc.name: np.nan for mc in metric_configs}),
            infeasibility_reason='HashingInfeasibleExperimenter',
        )
      else:
        self.exptr.evaluate([suggestion])

  def _is_infeasible(self, parameters: vz.ParameterDict) -> bool:
    hash_str = json.dumps(parameters.as_dict(), sort_keys=True) + str(self.seed)
    if random.Random(hash_str).random() < self.infeasible_prob:
      return True
    return False

  def problem_statement(self) -> vz.ProblemStatement:
    return self._problem


@attrs.define
class ParamRegionInfeasibleExperimenter(experimenter.Experimenter):
  """Selects a parameter and splits its values into feasible/infeasible."""

  exptr: experimenter.Experimenter = attrs.field()
  parameter_name: str = attrs.field()
  infeasible_interval: tuple[float, float] = attrs.field(
      default=(0.0, 0.2), kw_only=True
  )

  def __attrs_post_init__(self):
    self._problem = copy.deepcopy(self.exptr.problem_statement())

    param_config = self._problem.search_space.get(self.parameter_name)
    if param_config.type == vz.ParameterType.CATEGORICAL:
      raise ValueError('Categorical param type unsupported currently.')
    self._converter = converters.DefaultModelInputConverter(
        param_config, max_discrete_indices=0, scale=True
    )

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    param_features = self._converter.convert(suggestions)

    metric_configs = self._problem.metric_information
    for sugg_index, param_feature in enumerate(param_features):
      p = param_feature.item()
      suggestion = suggestions[sugg_index]
      if self.infeasible_interval[0] <= p <= self.infeasible_interval[1]:
        suggestion.complete(
            vz.Measurement(metrics={mc.name: np.nan for mc in metric_configs}),
            infeasibility_reason='ParameterRegionInfeasibleExperimenter',
        )
      else:
        self.exptr.evaluate([suggestion])

  def problem_statement(self) -> vz.ProblemStatement:
    return self._problem
