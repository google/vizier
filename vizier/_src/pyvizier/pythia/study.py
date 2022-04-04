"""StudyConfig used by pythia policies."""

import attr

from vizier._src.pyvizier.shared import base_study_config

StudyConfig = base_study_config.ProblemStatement


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
