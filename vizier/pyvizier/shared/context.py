"""Wrapper classes for vizier_pb2.Trial and other messages in it.

Example usage:
  trial = Trial.from_proto(trial_proto)
  print('This trial's auc is: ', trial.final_measurement.metrics['auc'].value)
  print('This trial had parameter "n_hidden_layers": ',
        trial.parameters['n_hidden_layers'].value)
"""
from typing import Dict, Optional

import attr
from vizier.pyvizier.shared import common
from vizier.pyvizier.shared import trial

Metadata = common.Metadata
ParameterValue = trial.ParameterValue


@attr.s(auto_attribs=True, frozen=False, init=True, slots=True)
class Context:
  """Wrapper for learning_vizier.service.Context proto."""
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
