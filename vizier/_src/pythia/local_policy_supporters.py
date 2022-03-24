"""Policy supporters that keep data in RAM."""
import datetime
from typing import Iterable, List, Optional, Sequence

import attr
from vizier._src.pythia import policy
from vizier._src.pythia import policy_supporter
from vizier.pyvizier import pythia as vz


# TODO: Keep the Pareto frontier trials.
@attr.s(frozen=True, init=True, slots=True)
class LocalPolicyRunner(policy_supporter.PolicySupporter):
  """Runs a fresh Study in RAM using a Policy.

  LocalPolicyRunner acts as a limited vizier service + client that runs in RAM.
  Trials can only be added and never removed.

  Example of using a policy to run a Study for 100 iterations, 1 trial each:
    runner = LocalPolicyRunner(my_study_config)
    policy = MyPolicy(runner)
    for _ in range(100):
      trials = runner.SuggestTrials(policy, count=1)
      if not trials:
        break
      for t in trials:
        t.complete(vz.Measurement(
            {'my_objective': my_objective(t)}, inplace=True))

  Attributes:
    study_config: Study config.
    study_guid: Unique identifier for the study.
  """

  study_config: vz.StudyConfig = attr.ib(
      init=True, validator=attr.validators.instance_of(vz.StudyConfig))
  study_guid: str = attr.ib(init=True, kw_only=True, default='')
  _trials: List[vz.Trial] = attr.ib(init=False, factory=list)

  @property
  def trials(self) -> Sequence[vz.Trial]:
    return self._trials

  def study_descriptor(self) -> vz.StudyDescriptor:
    return vz.StudyDescriptor(
        self.study_config, self.study_guid, max_trial_id=len(self._trials))

  def _check_study_guid(self, study_guid: Optional[str]) -> None:
    if study_guid is not None and self.study_guid != study_guid:
      raise ValueError('LocalPolicyRunner does not support accessing '
                       'other studies than the current one, which has '
                       'guid={self.study_guid}')

  def GetStudyConfig(self, study_guid: Optional[str] = None) -> vz.StudyConfig:
    self._check_study_guid(study_guid)
    return self.study_config

  def GetTrials(
      self,
      *,
      study_guid: Optional[str] = None,
      trial_ids: Optional[Iterable[int]] = None,
      min_trial_id: Optional[int] = None,
      max_trial_id: Optional[int] = None,
      status_matches: Optional[vz.TrialStatus] = None,
      include_intermediate_measurements: bool = True) -> List[vz.Trial]:
    self._check_study_guid(study_guid)
    min_trial_id = min_trial_id or 1
    max_trial_id = max_trial_id or (len(self._trials))
    trials = [
        t for t in self._trials[min_trial_id - 1:max_trial_id]
        if (status_matches is None or t.status == status_matches)
    ]
    if trial_ids is None:
      return trials
    else:
      trial_ids = set(trial_ids)
      return [t for t in trials if t.id in trial_ids]

  def CheckCancelled(self, note: Optional[str] = None) -> None:
    pass

  def TimeRemaining(self) -> datetime.timedelta:
    return datetime.timedelta(seconds=100.0)

  def UpdateMetadata(self, delta: policy_supporter.MetadataDelta) -> None:
    for ns in delta.on_study.namespaces():
      self.study_config.metadata.abs_ns(ns).update(delta.on_study.abs_ns(ns))

    for tid, deltum in delta.on_trials.items():
      for ns in deltum.namespaces():
        self._trials[tid - 1].metadata.abs_ns(ns).update(deltum.abs_ns(ns))

  def AddTrials(self, trials: Sequence[vz.Trial]) -> None:
    """(Re-)assigns ids to the trials and add them to the study."""
    for i, trial in enumerate(trials):
      trial.id = i + len(self.trials) + 1
    self._trials.extend(trials)

  def SuggestTrials(self, algorithm: policy.Policy,
                    count: int) -> Sequence[vz.Trial]:
    """Suggest and add new trials."""
    trials = []
    for suggestion in algorithm.suggest(
        policy.SuggestRequest(self.study_descriptor(), count)):
      # Assign temporary ids, which will be overwritten by AddTrials() method.
      trials.append(suggestion.to_trial(-1))
    self.AddTrials(trials)
    return trials
