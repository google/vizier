"""StudyConfig used by pythia policies."""

import attr

from vizier._src.pyvizier.shared import base_study_config


@attr.define(frozen=True, init=True)
class StudyDescriptor:
  """Light-weight summary for Study."""

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
