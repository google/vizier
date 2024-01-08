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

"""Contains resource utilities, such as name parsing.

Convention for variable naming for string identifiers:
  1. {}_id means single component (e.g. study_id = '1312931').
  2. {}_name means directory (e.g. study_name =
  'owners/my_username/studies/1312931').

Everything related to naming should be self-contained in this file; i.e.
dependents should only call official API functions from this file and never
explicitly write their own string processing.
"""
import re
import attr

from vizier.utils import attrs_utils

# Resource components cannot contain "/".
_resource_component_validator = [attrs_utils.assert_re_fullmatch(r'[^\/]+')]


@attr.define(init=True, frozen=True)
class OwnerResource:
  """Resource for Owners."""

  _owner_id: str = attr.ib(init=True, validator=_resource_component_validator)

  @classmethod
  def from_name(cls, resource_name: str):
    owner_match = re.match(r'^owners\/(?P<owner_id>[^\/]+)$', resource_name)

    if owner_match:
      return OwnerResource(owner_id=owner_match.group('owner_id'))
    else:
      raise ValueError(f'Incorrect resource name sent: {resource_name}')

  @property
  def owner_id(self):
    return self._owner_id

  @property
  def name(self) -> str:
    return f'owners/{self._owner_id}'


@attr.define(init=True, frozen=True)
class StudyResource:
  """Resource for Studies."""

  owner_id: str = attr.ib(init=True, validator=_resource_component_validator)
  study_id: str = attr.ib(init=True, validator=_resource_component_validator)

  @classmethod
  def from_name(cls, resource_name: str):
    """Creates StudyResource from a name."""
    study_match = re.match(
        r'^owners\/(?P<owner_id>[^\/]+)\/studies\/(?P<study_id>[^\/]+)$',
        resource_name,
    )

    if study_match:
      return StudyResource(
          study_match.group('owner_id'), study_match.group('study_id')
      )
    else:
      raise ValueError(
          f'{repr(resource_name)} is not a valid name for a Study resource.'
      )

  @property
  def owner_resource(self) -> OwnerResource:
    return OwnerResource(owner_id=self.owner_id)

  @property
  def name(self) -> str:
    return f'owners/{self.owner_id}/studies/{self.study_id}'

  def trial_resource(self, trial_id: str) -> 'TrialResource':
    """Creates a TrialResource when given a trial_id."""
    int_id = int(trial_id)
    if int_id <= 0:
      raise ValueError('Invalid trial_id: "{trial_id}"')
    return TrialResource(self.owner_id, self.study_id, int_id)


@attr.define(init=True, frozen=True)
class TrialResource:
  """Resource for Trials."""

  owner_id: str = attr.ib(init=True, validator=_resource_component_validator)
  study_id: str = attr.ib(init=True, validator=_resource_component_validator)
  trial_id: int = attr.ib(
      init=True,
      validator=[
          attr.validators.instance_of(int),
          attrs_utils.assert_not_negative,
      ],
  )

  @classmethod
  def from_name(cls, resource_name: str) -> 'TrialResource':
    """Creates TrialResource from a name."""
    trial_match = re.match(
        r'^owners\/(?P<owner_id>[^\/]+)\/studies\/(?P<study_id>[^\/]+)\/trials\/(?P<trial_id>[^\/]+)$',
        resource_name,
    )

    if trial_match:
      return TrialResource(
          trial_match.group('owner_id'),
          trial_match.group('study_id'),
          int(trial_match.group('trial_id')),
      )
    else:
      raise ValueError(
          f'{repr(resource_name)} is not a valid name for a Trial resource.'
      )

  @property
  def study_resource(self) -> StudyResource:
    return StudyResource(owner_id=self.owner_id, study_id=self.study_id)

  @property
  def early_stopping_operation_resource(
      self,
  ) -> 'EarlyStoppingOperationResource':
    return EarlyStoppingOperationResource(
        owner_id=self.owner_id, study_id=self.study_id, trial_id=self.trial_id
    )

  @property
  def name(self) -> str:
    return (
        f'owners/{self.owner_id}/studies/{self.study_id}/trials/{self.trial_id}'
    )


@attr.define(init=True, frozen=True)
class EarlyStoppingOperationResource:
  """Resource for Early Stopping Operations."""

  owner_id: str = attr.ib(init=True, validator=_resource_component_validator)
  study_id: str = attr.ib(init=True, validator=_resource_component_validator)
  trial_id: int = attr.ib(
      init=True,
      validator=[
          attr.validators.instance_of(int),
          attrs_utils.assert_not_negative,
      ],
  )

  @property
  def operation_id(self) -> str:
    return f'earlystopping/{self.study_id}/{self.trial_id}'

  @property
  def name(self) -> str:
    return f'owners/{self.owner_id}/operations/{self.operation_id}'

  @classmethod
  def from_name(cls, resource_name: str):
    """Creates EarlyStoppingOperationResource from a name."""
    operation_match = re.match(
        r'^owners\/(?P<owner_id>[^\/]+)/operations/earlystopping/(?P<study_id>[^\/]+)/(?P<trial_id>[^\/]+)$',
        resource_name,
    )
    if operation_match:
      return EarlyStoppingOperationResource(
          operation_match.group('owner_id'),
          operation_match.group('study_id'),
          int(operation_match.group('trial_id')),
      )
    else:
      raise ValueError(f'Incorrect resource name sent: {resource_name}')

  @property
  def trial_resource(self) -> TrialResource:
    return TrialResource(
        owner_id=self.owner_id, study_id=self.study_id, trial_id=self.trial_id
    )


@attr.define(init=True, frozen=True)
class SuggestionOperationResource:
  """Resource for Suggestion Operations."""

  owner_id: str = attr.ib(init=True, validator=_resource_component_validator)
  study_id: str = attr.ib(init=True, validator=_resource_component_validator)
  client_id: str = attr.ib(init=True, validator=_resource_component_validator)
  operation_number: int = attr.ib(
      init=True,
      validator=[
          attr.validators.instance_of(int),
          attrs_utils.assert_not_negative,
      ],
  )

  @property
  def operation_id(self) -> str:
    return (
        f'suggestion/{self.study_id}/{self.client_id}/{self.operation_number}'
    )

  @property
  def name(self) -> str:
    return f'owners/{self.owner_id}/operations/{self.operation_id}'

  @classmethod
  def from_name(cls, resource_name: str):
    """Creates SuggestionOperationResource from a name."""
    operation_match = re.match(
        r'^owners\/(?P<owner_id>[^\/]+)/operations/suggestion/(?P<study_id>[^\/]+)/(?P<client_id>[^\/]+)/(?P<operation_number>[^\/]+)$',
        resource_name,
    )
    if operation_match:
      return SuggestionOperationResource(
          operation_match.group('owner_id'),
          operation_match.group('study_id'),
          operation_match.group('client_id'),
          int(operation_match.group('operation_number')),
      )
    else:
      raise ValueError(f'Incorrect resource name sent: {resource_name}')
