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

"""Shared classes for representing studies."""

from typing import List
import attr
from vizier._src.pyvizier.shared import base_study_config
from vizier._src.pyvizier.shared import trial


@attr.define(frozen=True, init=True, slots=True, kw_only=False)
class ProblemAndTrials:
  """Container for problem statement and trials."""
  problem: base_study_config.ProblemStatement = attr.ib(init=True)
  trials: List[trial.Trial] = attr.ib(
      init=True,
      # TODO: Remove the pylint.
      converter=lambda x: list(x),  # pylint: disable=unnecessary-lambda
      default=attr.Factory(list))
