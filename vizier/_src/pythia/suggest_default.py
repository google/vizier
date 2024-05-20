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

import functools
import types
from typing import TypeVar

import attrs
from vizier import pyvizier as vz
from vizier._src.pythia.policy import Policy
from vizier._src.pythia.policy import SuggestDecision
from vizier._src.pythia.policy import SuggestRequest


_T = TypeVar('_T')


def get_default_parameters(search_space: vz.SearchSpace) -> vz.ParameterDict:
  """Gets the default parameters for the given search space."""
  builder = vz.SequentialParameterBuilder(search_space)

  for pc in builder:
    if pc.default_value is not None:
      builder.choose_value(pc.default_value)
    elif pc.type in (
        vz.ParameterType.CATEGORICAL,
        vz.ParameterType.INTEGER,
        vz.ParameterType.DISCRETE,
    ):
      # Choose the middle value.
      builder.choose_value(pc.feasible_values[len(pc.feasible_values) // 2])
    elif pc.type == vz.ParameterType.DOUBLE:
      if pc.num_feasible_values == 1:
        builder.choose_value(pc.bounds[0])
      else:
        # TODO: Handle scaling properly.
        midpoint = (pc.bounds[0] + pc.bounds[1]) / 2
        builder.choose_value(midpoint)
  return builder.parameters


def seed_with_default(suggest_fn: _T) -> _T:
  """Decorator for Policy.suggest to always suggest the default or center.

  How to use as a decorator:

  ```
  class MyPolicy(Policy):
    @seed_with_default
    def suggest(self, ...):
      ...
  ```

  How to use as a function:
  class MyPolicy(Policy):

    def __init__(self, ..., *, use_seed_with_default: bool):
      if use_seed_with_default:
        self.suggest = seed_with_default(self.suggest)

    def suggest(self, ...):
      ...

  Args:
    suggest_fn:

  Returns:
    suggest_fn that suggest the default or center of the search space if
    the study is empty.
  """

  if hasattr(suggest_fn, '__self__'):
    unbound = seed_with_default(suggest_fn.__func__)
    return types.MethodType(unbound, suggest_fn.__self__)

  @functools.wraps(suggest_fn)
  def wrapper_fn(self: Policy, request: SuggestRequest) -> SuggestDecision:
    """If study is empty, suggests a default trial before using the policy."""
    if request.max_trial_id > 0:
      return suggest_fn(self, request)

    default_parameters = get_default_parameters(
        request.study_config.search_space
    )
    decision = SuggestDecision([vz.TrialSuggestion(default_parameters)])

    if request.count > 1:
      more_suggestions = suggest_fn(
          self, attrs.evolve(request, count=request.count - 1)
      )
      decision.suggestions.extend(more_suggestions.suggestions)
      decision.metadata = more_suggestions.metadata

    return decision

  return wrapper_fn
