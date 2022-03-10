"""Deltas represent a change in a study's trials."""
import dataclasses
from typing import Sequence

from vizier.pyvizier import pythia as vz


@dataclasses.dataclass(frozen=True)
class CompletedTrials:
  """A group of completed trials.

  Attributes:
    completed: Completed Trials.
  """
  completed: Sequence[vz.Trial]
