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

"""StudyConfig used by pythia policies."""

import enum
import attr

from vizier._src.pyvizier.shared import base_study_config


class StudyState(enum.Enum):
  ACTIVE = 'ACTIVE'
  ABORTED = 'ABORTED'
  COMPLETED = 'COMPLETED'


@attr.define
class StudyStateInfo:
  state: StudyState = attr.field(
      converter=StudyState, validator=attr.validators.instance_of(StudyState))
  explanation: str = attr.field(default='')


@attr.define(frozen=True, init=True)
class StudyDescriptor:
  """Light-weight, cross-platform summary of Study."""

  config: base_study_config.ProblemStatement = attr.ib(
      init=True,
      validator=[
          attr.validators.optional(
              attr.validators.instance_of(base_study_config.ProblemStatement))
      ])

  guid: str = attr.ib(
      init=True,
      validator=[attr.validators.optional(attr.validators.instance_of(str))],
      kw_only=True)

  max_trial_id: int = attr.ib(
      init=True,
      validator=[attr.validators.optional(attr.validators.instance_of(int))],
      kw_only=True)
