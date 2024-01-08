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

"""Wrapper classes for Context protos and other messages in them."""
from typing import Dict, Optional

import attr
from vizier._src.pyvizier.shared import common
from vizier._src.pyvizier.shared import trial

Metadata = common.Metadata
ParameterValue = trial.ParameterValue


@attr.s(auto_attribs=True, frozen=False, init=True, slots=True)
class Context:
  """Wrapper for Context proto."""
  description: Optional[str] = attr.ib(
      init=True,
      kw_only=True,
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(str)),
      on_setattr=attr.setters.validate)

  parameters: Dict[str, ParameterValue] = attr.ib(
      init=True,
      kw_only=True,
      factory=dict,
      validator=attr.validators.deep_mapping(
          key_validator=attr.validators.instance_of(str),
          value_validator=attr.validators.instance_of(ParameterValue),
          mapping_validator=attr.validators.instance_of(dict)),
      on_setattr=attr.setters.validate)  # pytype: disable=wrong-arg-types

  metadata: Metadata = attr.ib(
      init=True,
      kw_only=True,
      default=Metadata(),
      validator=attr.validators.instance_of(Metadata),
      on_setattr=attr.setters.validate)

  related_links: Dict[str, str] = attr.ib(
      init=True,
      kw_only=True,
      factory=dict,
      validator=attr.validators.deep_mapping(
          key_validator=attr.validators.instance_of(str),
          value_validator=attr.validators.instance_of(str),
          mapping_validator=attr.validators.instance_of(dict)),
      on_setattr=attr.setters.validate)  # pytype: disable=wrong-arg-types
