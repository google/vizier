"""The Policy can use these classes to communicate with Vizier."""

import abc
import collections
import dataclasses
import datetime
from typing import Dict, Iterable, List, Optional

from vizier import pyvizier as vz


@dataclasses.dataclass(frozen=True)
class MetadataDelta:
  """Carries cumulative delta for a batch metadata update.

  Attributes:
    on_study: Updates to be made on study-level metadata.
    on_trials: Maps trial id to updates.
  """

  on_study: vz.Metadata = dataclasses.field(default_factory=vz.Metadata)
  on_trials: Dict[int, vz.Metadata] = dataclasses.field(
      default_factory=lambda: collections.defaultdict(vz.Metadata))


class _MetadataUpdateContext:
  """Metadata update context.

  Usage:
    # All metadata updates in the context are queued, not immediately applied.
    # Upon exit, supporter handles all metadata updates in a batch.
    with pythia2._MetadataUpdateContext(policy_supporter) as mu:
      # Study-level metadata.
      mu.assign('namespace', 'key', 'value')
      # Trial-level metadata.
      mu.assign('namespace', 'key', 'value', trial_id=1)
      # Same as above but with a side effect. After this line the following
      # line is True:
      #   trial.metadata.ns('namespace')['key'] == 'value'
      mu.assign('namespace', 'key', 'value', trial)
  """

  def __init__(self, supporter: 'PolicySupporter'):
    self._supporter = supporter
    self._delta = MetadataDelta()

  # pylint: disable=invalid-name
  def assign(self,
             namespace: str,
             key: str,
             value: vz.MetadataValue,
             trial: Optional[vz.Trial] = None,
             *,
             trial_id: Optional[int] = None):
    """Assigns metadata.

    Args:
      namespace: Namespace of the metadata. See vz.Metadata doc for more
        details.
      key:
      value:
      trial: If specified, `trial_id` must be None. It behaves the same as when
        `trial_id=trial.id`, except that `trial` is immediately modified.
      trial_id: If specified, `trial` must be None. If both `trial` and
        `trial_id` are None, then the key-value pair will be assigned to the
        study.

    Raises:
      ValueError:
    """
    if trial is None and trial_id is None:
      self._delta.on_study.ns(namespace)[key] = value
    elif trial is not None and trial_id is not None:
      raise ValueError(
          'At most one of `trial` and `trial_id` can be specified.')
    elif trial is not None:
      self._delta.on_trials[trial.id].ns(namespace)[key] = value
      trial.metadata.ns(namespace)[key] = value
    elif trial_id is not None:
      self._delta.on_trials[trial_id].ns(namespace)[key] = value

  def __enter__(self):
    return self

  def __exit__(self, *args):
    """upon exit, sends a batch update request."""
    self._supporter.SendMetadata(self._delta)


class PolicySupporter(abc.ABC):
  """Used by Policy instances to communicate with Vizier."""

  # TODO: Change to GetStudyDescriptor.
  @abc.abstractmethod
  def GetStudyConfig(self, study_guid: Optional[str] = None) -> vz.StudyConfig:
    """Requests a StudyConfig from Vizier.

    This sends a PythiaToVizier.trial_selector packet and waits for the
    response(s).  You can call this multiple times, and it is thread-friendly,
    so you can even overlap calls.

    Args:
      study_guid: The GUID of the study whose StudyConfig you want. Note that
        access control applies. By default, use the current study's GUID.

    Returns:
      The requested StudyConfig proto.

    Raises:
      CancelComputeError: (Do not catch.)
      PythiaProtocolError: (Do not catch.)
      VizierDatabaseError: If the database operation raises an error, e.g. if
        $study_guid refers to a nonexistent or inaccessible study.
    """

  @abc.abstractmethod
  def GetTrials(
      self,
      *,
      study_guid: Optional[str] = None,
      trial_ids: Optional[Iterable[int]] = None,
      min_trial_id: Optional[int] = None,
      max_trial_id: Optional[int] = None,
      status_matches: Optional[vz.TrialStatus] = None,
      include_intermediate_measurements: bool = True) -> List[vz.Trial]:
    """Requests Trials from Vizier.

    Args:
      study_guid: The GUID of the study to get Trials from.  Default is None,
        which means the current Study.
      trial_ids: a list of Trial id numbers to acquire.
      min_trial_id: Trials in [min_trial_id, max_trial_id] are selected, if at
        least one of the two is not None.
      max_trial_id: Trials in [min_trial_id, max_trial_id] are selected, if at
        least one of the two is not None.
      status_matches: If not None, filters for Trials where
        Trial.status==status_matches.  The default passes all types of Trial.
      include_intermediate_measurements: If True (default), the returned Trials
        must have all measurements. Note that the final Measurement is always
        included for COMPLETED Trials. If False, PolicySupporter _may_ leave
        `measurements` field empty in the returned Trials in order to optimize
        speed, but it is not required to do so.

    Returns:
      Trials obtained from Vizier.

    Raises:
      CancelComputeError: (Do not catch.)
      PythiaProtocolError: (Do not catch.)
      VizierDatabaseError: If the database operation raises an error, e.g. if
        $study_guid refers to a nonexistent or inaccessible study.

    NOTE: if $trial_ids is set, $min_trial_id, $max_trial_id, and
      $status_matches will be ignored.
    """

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

  def MetadataUpdate(self) -> _MetadataUpdateContext:
    """Queues metadata updates, then passes them to UpdateMetadata().

    Usage:
      ps = PolicySupporter()
      with ps.MetadataUpdate() as mu:
        # Study-level metadata.
        mu.assign('namespace', 'key', 'value')
        # Trial-level metadata.
        mu.assign('namespace', 'key', 'value', trial_id=1)

    Returns:
      A _MetadataUpdateContext instance to use as a context.
    Raises:
      CancelComputeError: (Do not catch.)
      PythiaProtocolError: (Do not catch.)
      VizierDatabaseError: If the database operation raises an error.
    """
    return _MetadataUpdateContext(self)

  @abc.abstractmethod
  def SendMetadata(self, delta: MetadataDelta) -> None:
    """Updates the Study's metadata in Vizier's database.

    The MetadataUpdate() method is preferred for normal use.

    Args:
      delta: Metadata to be uploaded to the Vizier database.

    Raises:
      CancelComputeError: (Do not catch.)
      PythiaProtocolError: (Do not catch.)
      VizierDatabaseError: If the database operation raises an error.
    """
