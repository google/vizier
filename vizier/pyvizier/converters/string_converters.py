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

"""Converter utils for parameters for free-form strings."""

from typing import Sequence
import copy
import json
import attrs
from vizier import pyvizier as vz

_METADATA_VERSION = '0.0.1a'
PROMPT_TUNING_NS = 'prompt_tuning'


@attrs.define
class PromptTuningConfig:
  """Variables and utils for configuring prompt tuning."""

  default_prompts: dict[str, str] = attrs.field(factory=dict)

  def augment_problem(
      self, problem: vz.ProblemStatement
  ) -> vz.ProblemStatement:
    """Augments problem statement to enable for prompt tuning."""
    for k, v in self.default_prompts.items():
      problem.search_space.root.add_categorical_param(k, [v], default_value=v)
    problem.metadata.ns(PROMPT_TUNING_NS)['version'] = _METADATA_VERSION
    return problem

  def to_prompt_trials(self, trials: Sequence[vz.Trial]) -> Sequence[vz.Trial]:
    """Convert to prompt Trial via metadata to string valued parameters."""
    prompt_trials = copy.deepcopy(trials)
    for trial in prompt_trials:
      prompt_values = json.loads(trial.metadata.ns(PROMPT_TUNING_NS)['values'])
      for k in self.default_prompts.keys():
        if k in prompt_values:
          trial.parameters[k] = prompt_values[k]
    return prompt_trials

  def to_valid_suggestions(
      self, suggestions: Sequence[vz.TrialSuggestion]
  ) -> Sequence[vz.TrialSuggestion]:
    """Returns TrialSuggestions that are valid in the augmented problem."""
    valid_suggestions = copy.deepcopy(suggestions)
    for suggestion in valid_suggestions:
      prompt_values = {}
      for k, default_value in self.default_prompts.items():
        prompt_values[k] = suggestion.parameters[k].value
        suggestion.parameters[k] = default_value
      suggestion.metadata.ns(PROMPT_TUNING_NS)['values'] = json.dumps(
          prompt_values
      )
      suggestion.metadata.ns(PROMPT_TUNING_NS)['version'] = _METADATA_VERSION
    return valid_suggestions
