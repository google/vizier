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

"""The Policy can use these classes to communicate with Vizier."""

import abc
import datetime
from typing import Iterable, List, Optional

from vizier import pyvizier as vz


class PolicySupporter(abc.ABC):
  """Used by Policy instances to communicate with Vizier."""

  # TODO: Change to GetStudyDescriptor (Maybe: Note that
  #   StudyDescriptor includes $max_trial_id, so it causes many more database
  #   collisions than ProblemStatement, especially if there are lots of
  #   workers.)
  @abc.abstractmethod
  def GetStudyConfig(self, study_guid: str) -> vz.ProblemStatement:
    """Requests a StudyConfig from Vizier.

    This sends a PythiaToVizier.trial_selector packet and waits for the
    response(s).  You can call this multiple times, and it is thread-friendly,
    so you can even overlap calls.

    Args:
      study_guid: The GUID of the study whose StudyConfig you want. Note that
        access control applies.

    Returns:
      The requested StudyConfig proto.

    Raises:
      CancelComputeError: (Do not catch.)
      PythiaProtocolError: (Do not catch.)
      VizierDatabaseError: If the database operation raises an error, e.g. if
        $study_guid refers to a nonexistent or inaccessible study.
    """

  # TODO: Should take `TrialFilter` as input, instead of
  # its fields listed as keyword arguments.
  @abc.abstractmethod
  def GetTrials(
      self,
      *,
      study_guid: Optional[str] = None,
      trial_ids: Optional[Iterable[int]] = None,
      min_trial_id: Optional[int] = None,
      max_trial_id: Optional[int] = None,
      status_matches: Optional[vz.TrialStatus] = None,
      include_intermediate_measurements: bool = True
  ) -> List[vz.Trial]:
    """Requests Trials from Vizier.

    Args:
      study_guid: The GUID of the study to get Trials from. If None, uses the
        current Study.
      trial_ids: a list of Trial id numbers to acquire (if None, allows all
        Trials.)
      min_trial_id: Trials with Trial.id >= min_trial ID are selected, if not
        None.
      max_trial_id: Trials with Trial.id <= min_trial ID are selected, if not
        None.
      status_matches: If not None, filters for Trials where
        Trial.status==status_matches.  Selects all Trials by default.
      include_intermediate_measurements: If True (default), the returned Trials
        will include all intermediate measurements.  If False, PolicySupporter
        _may_ leave the `measurements` field empty in the returned Trials (e.g.
        to optimize speed).

    Note that the $final_measurement field will always be included when
      available, i.e. for COMPLETED Trials.

    Returns:
      Trials obtained from Vizier, in order of increasing Trial ID.
      Each argument selects a subset of the Study's Trials, and the result is
      the intersection of the subsets.

    Raises:
      CancelComputeError: (Do not catch.)
      PythiaProtocolError: (Do not catch.)
      VizierDatabaseError: If the database operation raises an error, e.g. if
        $study_guid refers to a nonexistent or inaccessible study.
    """

  @property
  @abc.abstractmethod
  def study_guid(self) -> str:
    """Default study GUID."""

  def CheckCancelled(self, note: Optional[str] = None) -> None:
    """Throws a CancelComputeError on timeout or if Vizier cancels.

    This should be called occasionally by any long-running computation.
    Raises an exception if the interaction has been cancelled by the Vizier
    side of the protocol; the exception shuts down the Pythia server.

    Args:
      note: for debugging.

    Raises:
      CancelComputeError: (Do not catch.)
    """
    pass

  def TimeRemaining(self) -> datetime.timedelta:
    """The time remaining to compute a result.

    Returns:
      The remaining time before the RPC is considered to have timed out; it
      returns datetime.timedelta.max if no deadline was specified in the RPC.

    This is an alternative to calling CheckCancelled(); both have the goal of
    terminating runaway computations.  If your computation times out,
    you should raise TemporaryPythiaError (if you want a retry) or
    InactivateStudyError (if not).
    """
    return datetime.timedelta(hours=1.0)
