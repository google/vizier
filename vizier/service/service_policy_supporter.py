"""Policy Supporter used for the OSS Vizier service.

The Policy can use these methods to communicate with Vizier.
"""
import datetime
from typing import Iterable, List, Optional

from vizier import pythia
from vizier import pyvizier as vz
from vizier.service import pyvizier
from vizier.service import stubs_util
from vizier.service import utils
from vizier.service import vizier_service_pb2
from vizier.service import vizier_service_pb2_grpc


# TODO: Consider replacing protos with Pyvizier clients.py.
class ServicePolicySupporter(pythia.PolicySupporter):
  """Service version of the PolicySupporter."""

  def __init__(self, study_guid: str):
    """Initalization.

    Args:
      study_guid: A default study_name; the name of this study.
    """
    self._study_guid = study_guid
    self._vizier_service_stub: Optional[
        vizier_service_pb2_grpc.VizierServiceStub] = None

  def connect_to_vizier(self, vizier_service_endpoint: str) -> None:
    self._vizier_service_stub = stubs_util.create_vizier_server_stub(
        vizier_service_endpoint)

  def GetStudyConfig(self,
                     study_guid: Optional[str] = None) -> vz.ProblemStatement:
    if study_guid is None:
      study_guid = self._study_guid
    request = vizier_service_pb2.GetStudyRequest(name=study_guid)
    study = self._vizier_service_stub.GetStudy(request)
    return pyvizier.StudyConfig.from_proto(study.study_spec).to_problem()

  # TODO: Use TrialsFilter instead.
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
    trials = self._vizier_service_stub.ListTrials(request).trials
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

  def SendMetadata(self, delta: vz.MetadataDelta) -> None:
    """Updates the metadata."""
    self.CheckCancelled('UpdateMetadata entry')
    request = pyvizier.metadata_util.to_request_proto(self._study_guid, delta)
    waiter = utils.ResponseWaiter()
    # In a real server, this happens in another thread:
    response = self._vizier_service_stub.UpdateMetadata(request)
    waiter.Report(response)
    # Wait for Vizier to reply with a UpdateMetadataResponse packet.
    waiter.WaitForResponse()
    self.CheckCancelled('UpdateMetadata exit')
