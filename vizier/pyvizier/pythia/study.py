"""StudyConfig used by pythia policies."""

from typing import List

import attr

from vizier._src.pyvizier.shared import base_study_config
from vizier._src.pyvizier.shared import common


@attr.define(frozen=False, init=True, slots=True)
class StudyConfig:
  """StudyConfig used by pythia policies."""

  search_space: base_study_config.SearchSpace = attr.field(
      init=True,
      factory=base_study_config.SearchSpace,
      validator=attr.validators.instance_of(base_study_config.SearchSpace))

  metric_information: List[base_study_config.MetricInformation] = attr.ib(
      init=True,
      factory=list,
      validator=attr.validators.deep_iterable(
          member_validator=attr.validators.instance_of(
              base_study_config.MetricInformation),
          iterable_validator=attr.validators.instance_of(list)),
      on_setattr=attr.setters.validate,
      kw_only=True)

  metadata: common.Metadata = attr.field(
      init=True,
      kw_only=True,
      factory=common.Metadata,
      validator=attr.validators.instance_of(common.Metadata))

  @property
  def debug_info(self) -> str:
    return ''


@attr.define(frozen=True, init=True)
class StudyDescriptor:
  """Light-weight summary for Study."""

  config: StudyConfig = attr.ib(
      init=True,
      validator=[
          attr.validators.optional(attr.validators.instance_of(StudyConfig))
      ])

  guid: str = attr.ib(
      init=True,
      default='',
      validator=[attr.validators.optional(attr.validators.instance_of(str))])

  max_trial_id: int = attr.ib(
      init=True,
      default=0,
      validator=[attr.validators.optional(attr.validators.instance_of(int))],
      kw_only=True)
