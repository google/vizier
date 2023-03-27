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

"""Converters for PyVizier with RayTune."""

from typing import Any, Dict

from ray import tune
from vizier import pyvizier as vz


class SearchSpaceConverter:
  """Converts pyvizier.SearchSpace <-> RayTune Search Space."""

  @classmethod
  def to_dict(
      cls,
      search_space: vz.SearchSpace,
  ) -> Dict[str, Any]:
    """Converts PyVizier ProblemStatement to Proto version."""
    param_space = {}
    for param in search_space.parameters:
      lower, upper = param.bounds
      param_space[param.name] = tune.uniform(lower, upper)
    return param_space
