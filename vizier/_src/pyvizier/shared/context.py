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
