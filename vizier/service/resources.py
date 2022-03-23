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
from typing import Optional, Any, Union
import attr


def _assert_not_empty(instance: Any, attribute: Any,
                      value: Optional[Union[str, int]]):
  del instance, attribute
  if value is None:
    raise ValueError('must not be empty.')


@attr.define(init=True, frozen=True)
class OwnerResource:
  """Resource for Owners."""
  _owner_id: str = attr.ib(init=True, validator=_assert_not_empty)

  @classmethod
  def from_name(cls, resource_name: str):
    owner_match = re.match(r'^owners/(?P<owner_id>[\w\s]+)$', resource_name)

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
  owner_id: str = attr.ib(init=True, validator=_assert_not_empty)
  study_id: str = attr.ib(init=True, validator=_assert_not_empty)

  @classmethod
  def from_name(cls, resource_name: str):
    """Creates StudyResource from a name."""
    study_match = re.match(
        r'^owners/(?P<owner_id>[\w\s]+)/studies/(?P<study_id>[\w\s]+)$',
        resource_name)

    if study_match:
      return StudyResource(
          study_match.group('owner_id'), study_match.group('study_id'))
    else:
      raise ValueError(f'Incorrect resource name sent: {resource_name}')

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
  owner_id: str = attr.ib(init=True, validator=_assert_not_empty)
  study_id: str = attr.ib(init=True, validator=_assert_not_empty)
  trial_id: int = attr.ib(init=True, validator=_assert_not_empty)

  @classmethod
  def from_name(cls, resource_name: str) -> 'TrialResource':
    """Creates TrialResource from a name."""
    trial_match = re.match(
        r'^owners/(?P<owner_id>[\w\s]+)/studies/(?P<study_id>[\w\s]+)/trials/(?P<trial_id>[\w\s]+)$',
        resource_name)

    if trial_match:
      return TrialResource(
          trial_match.group('owner_id'), trial_match.group('study_id'),
          int(trial_match.group('trial_id')))
    else:
      raise ValueError(f'Incorrect resource name sent: {resource_name}')

  def make_early_stopping_operation(self, operation_id: str = ''):
    pass

  @property
  def study_resource(self) -> StudyResource:
    return StudyResource(owner_id=self.owner_id, study_id=self.study_id)

  @property
  def name(self) -> str:
    return f'owners/{self.owner_id}/studies/{self.study_id}/trials/{self.trial_id}'


@attr.define(init=True, frozen=True)
class EarlyStoppingOperationResource:
  """Resource for Early Stopping Operations."""
  owner_id: str = attr.ib(init=True, validator=_assert_not_empty)
  study_id: str = attr.ib(init=True, validator=_assert_not_empty)
  trial_id: int = attr.ib(init=True, validator=_assert_not_empty)

  @property
  def operation_id(self) -> str:
    return f'earlystopping_{self.study_id}_{self.trial_id}'

  @property
  def name(self) -> str:
    return f'owners/{self.owner_id}/operations/{self.operation_id}'

  @classmethod
  def from_name(cls, resource_name: str):
    """Creates EarlyStoppingOperationResource from a name."""
    operation_match = re.match(
        r'^owners/(?P<owner_id>[\w\s]+)/operations/earlystopping_(?P<study_id>[\w\s]+)_(?P<trial_id>[\w\s]+)$',
        resource_name)
    if operation_match:
      return EarlyStoppingOperationResource(
          operation_match.group('owner_id'), operation_match.group('study_id'),
          int(operation_match.group('trial_id')))
    else:
      raise ValueError(f'Incorrect resource name sent: {resource_name}')

  @property
  def trial(self) -> TrialResource:
    return TrialResource(
        owner_id=self.owner_id, study_id=self.study_id, trial_id=self.trial_id)


@attr.define(init=True, frozen=True)
class SuggestionOperationResource:
  """Resource for Suggestion Operations."""

  owner_id: str = attr.ib(init=True, validator=_assert_not_empty)
  client_id: str = attr.ib(init=True, validator=_assert_not_empty)
  operation_number: int = attr.ib(init=True, validator=_assert_not_empty)

  @property
  def operation_id(self) -> str:
    return f'suggestion_{self.client_id}_{self.operation_number}'

  @property
  def name(self) -> str:
    return f'owners/{self.owner_id}/operations/{self.operation_id}'

  @classmethod
  def from_name(cls, resource_name: str):
    """Creates SuggestionOperationResource from a name."""
    operation_match = re.match(
        r'^owners/(?P<owner_id>[\w\s]+)/operations/suggestion_(?P<client_id>[\w\s]+)_(?P<operation_number>[\w\s]+)$',
        resource_name)
    if operation_match:
      return SuggestionOperationResource(
          operation_match.group('owner_id'), operation_match.group('client_id'),
          int(operation_match.group('operation_number')))
    else:
      raise ValueError(f'Incorrect resource name sent: {resource_name}')
