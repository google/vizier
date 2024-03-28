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

"""Decorators for Policy.suggest."""

import attrs
import numpy as np
from vizier._src.pythia.policy import Policy
from vizier._src.pythia.policy import SuggestDecision
from vizier._src.pythia.policy import SuggestRequest
from vizier._src.pyvizier.shared import parameter_config
from vizier._src.pyvizier.shared import parameter_iterators as pi
from vizier._src.pyvizier.shared import trial
from vizier.pyvizier.converters import core


def seed_with_default(suggest_fn):
  """Decorator for Policy.suggest to always suggest the default or center."""

  def wapper_fn(self: Policy, request: SuggestRequest) -> SuggestDecision:
    """If study is empty, suggests a default trial before using the policy."""
    if request.max_trial_id > 0:
      return suggest_fn(self, request)

    search_space = request.study_config.search_space
    builder = pi.SequentialParameterBuilder(search_space)

    for pc in builder:
      if pc.default_value is not None:
        builder.choose_value(pc.default_value)
      elif pc.type in (
          parameter_config.ParameterType.CATEGORICAL,
          parameter_config.ParameterType.INTEGER,
          parameter_config.ParameterType.DISCRETE,
      ):
        builder.choose_value(pc.feasible_values[len(pc.feasible_values) // 2])
      elif pc.type == parameter_config.ParameterType.DOUBLE:
        if pc.num_feasible_values == 1:
          builder.choose_value(pc.bounds[0])
        else:
          scaler = core.ModelInputArrayBijector.scaler_from_spec(
              core.NumpyArraySpec.from_parameter_config(pc)
          )
          builder.choose_value(scaler.backward_fn(np.array(0.5)).item())
        scaler = core.ModelInputArrayBijector.scaler_from_spec(
            core.NumpyArraySpec.from_parameter_config(pc)
        )
        builder.choose_value(scaler.backward_fn(np.array(0.5)).item())
    decision = SuggestDecision([trial.TrialSuggestion([builder.parameters])])

    if request.count > 1:
      more_suggestions = suggest_fn(
          attrs.evolve(request, count=request.count - 1)
      )
      decision.suggestions.extend(more_suggestions.suggestions)
      decision.metadata = more_suggestions.metadata

    return decision

  return wapper_fn
