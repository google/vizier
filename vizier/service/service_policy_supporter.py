"""Policy Supporter used for the OSS Vizier service.

The Policy can use these methods to communicate with Vizier.
"""

import datetime
from typing import Iterable, List, Optional
from vizier import pythia
from vizier import pyvizier as vz
from vizier.service import pyvizier
from vizier.service import utils

from vizier.service import vizier_service_pb2
from vizier.service import vizier_service_pb2_grpc


# TODO: Implement UpdateMetadata after metadata additions
class ServicePolicySupporter(pythia.PolicySupporter):
  """Service version of the PolicySupporter."""

  # TODO: Replace vizier_service_instance with a vizier_client.
  def __init__(
      self, study_guid: str,
      vizier_service_instance: vizier_service_pb2_grpc.VizierServiceServicer):
    """Initalization stores a Vizier Service Instance to list Trials.

    Note that this is NOT sending an actual RPC to the server, because the
    policy is running in the same process as the server, but this can be
    trivially modified.

    Args:
      study_guid: A default study_name; the name of this study.
      vizier_service_instance: vizier_service.VizierService() to be used for
        retriving from datastore.
    """
    self._study_guid = study_guid
    self._vizier_service = vizier_service_instance

  def GetStudyConfig(self, study_guid: Optional[str] = None) -> vz.StudyConfig:
    if study_guid is None:
      study_guid = self._study_guid
    request = vizier_service_pb2.GetStudyRequest(name=study_guid)
    study = self._vizier_service.GetStudy(request, None)
    return pyvizier.StudyConfig.from_proto(study.study_spec)

  def GetTrials(
      self,
      *,
      study_guid: Optional[str] = None,  # Equivalent to Study.name in OSS.
      trial_ids: Optional[Iterable[int]] = None,
      min_trial_id: Optional[int] = None,
      max_trial_id: Optional[int] = None,
      status_matches: Optional[vz.TrialStatus] = None,
      include_intermediate_measurements: bool = True) -> List[vz.Trial]:
    """Requests Trials from Vizier."""

    if study_guid is None:
      study_guid = self._study_guid
    request = vizier_service_pb2.ListTrialsRequest(parent=study_guid)
    # Implicitly creates a copy of the data.
    trials = self._vizier_service.ListTrials(request, None).trials
    all_pytrials = pyvizier.TrialConverter.from_protos(trials)

    if trial_ids is not None:
      # We're going to do repeated comparisons in _trial_filter(), so this
      # needs to be more real than an Iterator.
      trial_ids = frozenset(trial_ids)

    def _trial_filter(trial) -> bool:
      if trial_ids is not None and trial.id not in trial_ids:
        return False
      else:
        if min_trial_id is not None:
          if trial.id < min_trial_id:
            return False
        if max_trial_id is not None:
          if trial.id > max_trial_id:
            return False
        if status_matches:
          if trial.status != status_matches:
            return False
      return True

    filtered_pytrials = [t for t in all_pytrials if _trial_filter(t)]

    # Doesn't affect datastore when measurements are deleted.
    if not include_intermediate_measurements:
      for filtered_pytrial in filtered_pytrials:
        filtered_pytrial.measurements = []

    return filtered_pytrials

  def CheckCancelled(self, note: Optional[str] = None) -> None:
    """Throws a CancelComputeError on timeout or if Vizier cancels."""
    pass  # Do nothing since it's one single process.

  def TimeRemaining(self) -> datetime.timedelta:
    """The time remaining to compute a result."""
    return datetime.timedelta.max  # RPCs don't have timeouts in OSS.

  def SendMetadata(self, delta: pythia.MetadataDelta) -> None:
    """Updates the metadata."""
    self.CheckCancelled('UpdateMetadata entry')
    request = vizier_service_pb2.UpdateMetadataRequest(name=self._study_guid)
    # Study Metadata
    for ns in delta.on_study.namespaces():
      for k, v in delta.on_study.abs_ns(ns).items():
        utils.AssignKeyValuePlus(request, trial_id=None, key=k, ns=ns, value=v)
    # Trial Metadata
    for trial_id, trial_metadata in delta.on_trials.items():
      for ns in trial_metadata.namespaces():
        for k, v in trial_metadata.abs_ns(ns).items():
          utils.AssignKeyValuePlus(
              request, trial_id=trial_id, key=k, ns=ns, value=v)
    waiter = utils.ResponseWaiter()
    # In a real server, this happens in another thread:
    response = self._vizier_service.UpdateMetadata(request, None)
    waiter.Report(response)
    # Wait for Vizier to reply with a UpdateMetadataResponse packet.
    waiter.WaitForResponse()
    self.CheckCancelled('UpdateMetadata exit')
