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

"""Convenience classes for configuring Vizier Early-Stopping Configs."""
import copy

import attr
from vizier._src.service import study_pb2

# When new early stopping config protos are added, include them below
# with a Union[]
AutomatedStoppingConfigProto = study_pb2.StudySpec.DefaultEarlyStoppingSpec


@attr.s(frozen=True, init=True, slots=True, kw_only=True)
class AutomatedStoppingConfig:
  """A wrapper for study_pb2.automated_stopping_spec."""
  _proto: AutomatedStoppingConfigProto = attr.ib(init=True, kw_only=True)

  @classmethod
  def default_stopping_spec(cls) -> 'AutomatedStoppingConfig':
    """Use Vizier's default early stopping."""
    config = study_pb2.StudySpec.DefaultEarlyStoppingSpec()
    return cls(proto=config)

  @classmethod
  def from_proto(
      cls, proto: AutomatedStoppingConfigProto) -> 'AutomatedStoppingConfig':
    return cls(proto=proto)

  def to_proto(self) -> AutomatedStoppingConfigProto:
    """Returns this object as a proto."""
    return copy.deepcopy(self._proto)
