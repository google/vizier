"""Policy Supporter used for the OSS Vizier service.

The Policy can use these methods to communicate with Vizier.
"""
import datetime
from typing import Iterable, List, Optional, Union

from vizier import pythia
from vizier import pyvizier as vz
from vizier.service import pythia_util
from vizier.service import pyvizier
from vizier.service import vizier_service_pb2
from vizier.service import vizier_service_pb2_grpc

VizierService = Union[vizier_service_pb2_grpc.VizierServiceStub,
                      vizier_service_pb2_grpc.VizierServiceServicer]


# TODO: Consider replacing protos with Pyvizier clients.py.
class ServicePolicySupporter(pythia.PolicySupporter):
  """Service version of the PolicySupporter."""

  def __init__(self, study_guid: str, vizier_service: VizierService):
    """Initalization.

    Args:
      study_guid: A default study_name; the name of this study.
      vizier_service: Vizier Service, in the form of a GRPC stub or actual
        class.
    """
    self._study_guid = study_guid
    self._vizier_service = vizier_service

  def GetStudyConfig(self, study_guid: str) -> vz.ProblemStatement:
    request = vizier_service_pb2.GetStudyRequest(name=study_guid)
    study = self._vizier_service.GetStudy(request)
    return pyvizier.StudyConfig.from_proto(study.study_spec).to_problem()

  # TODO: Support filters in ListTrialsRequest.
  def GetTrials(
      self,
      *,
      study_guid: Optional[str] = None,  # Equivalent to Study.name in OSS.
      trial_ids: Optional[Iterable[int]] = None,
      min_trial_id: Optional[int] = None,
      max_trial_id: Optional[int] = None,
      status_matches: Optional[vz.TrialStatus] = None,
      include_intermediate_measurements: bool = True) -> List[vz.Trial]:
    """Fetch all trials and then apply the filter."""

    if study_guid is None:
      study_guid = self._study_guid
    request = vizier_service_pb2.ListTrialsRequest(parent=study_guid)
    trials = self._vizier_service.ListTrials(request).trials
    all_pytrials = pyvizier.TrialConverter.from_protos(trials)

    trial_filter = vz.TrialFilter(
        ids=trial_ids,
        min_id=min_trial_id,
        max_id=max_trial_id,
        status=[status_matches] if status_matches else None)
    filtered_pytrials = [t for t in all_pytrials if trial_filter(t)]

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
    waiter = pythia_util.ResponseWaiter()
    # In a real server, this happens in another thread:
    response = self._vizier_service.UpdateMetadata(request)
    waiter.Report(response)
    # Wait for Vizier to reply with a UpdateMetadataResponse packet.
    waiter.WaitForResponse()
    self.CheckCancelled('UpdateMetadata exit')
