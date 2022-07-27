"""Wrapper class for OSS study_pb2.Study."""
# TODO: Make this a shared class.

from typing import List
import attr
from vizier._src.pyvizier.shared import base_study_config
from vizier._src.pyvizier.shared import trial


@attr.define
class StudyWithTrials:
  trials: List[trial.Trial] = attr.field()
  problem: base_study_config.ProblemStatement = attr.field()
