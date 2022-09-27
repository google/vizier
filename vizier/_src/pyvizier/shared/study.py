"""Shared classes for representing studies."""

from typing import List
import attr
from vizier._src.pyvizier.shared import base_study_config
from vizier._src.pyvizier.shared import trial


@attr.define(frozen=True, init=True, slots=True, kw_only=False)
class ProblemAndTrials:
  """Container for problem statement and trials."""
  problem: base_study_config.ProblemStatement = attr.ib(init=True)
  trials: List[trial.Trial] = attr.ib(
      init=True,
      # TODO: Remove the pylint.
      converter=lambda x: list(x),  # pylint: disable=unnecessary-lambda
      default=attr.Factory(list))
