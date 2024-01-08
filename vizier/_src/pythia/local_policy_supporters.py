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

"""Policy supporters that keep data in RAM."""

import copy
import datetime
from typing import Iterable, List, Optional, Sequence
import uuid

from absl import logging
import attr
import numpy as np
from vizier import pyvizier as vz
from vizier._src.pythia import policy
from vizier._src.pythia import policy_supporter
from vizier.pyvizier import converters
from vizier.pyvizier import multimetric


# TODO: Keep the Pareto frontier trials.
@attr.s(frozen=False, init=True, slots=True)
class InRamPolicySupporter(policy_supporter.PolicySupporter):
  """Runs a fresh Study in RAM using a Policy.

  InRamPolicySupporter acts as a limited vizier service + client that runs in
  RAM. Trials can only be added and never removed.

  Example of using a policy to run a Study for 100 iterations, 1 trial each:
    runner = InRamPolicySupporter(my_study_config)
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
    study_guid: Unique identifier for the current study.
  """

  study_config: vz.ProblemStatement = attr.ib(
      init=True,
      validator=attr.validators.instance_of(vz.ProblemStatement),
      on_setattr=attr.setters.frozen,
  )
  study_guid: str = attr.ib(
      init=True,
      kw_only=True,
      default='',
      converter=str,
      on_setattr=attr.setters.frozen,
  )
  _trials: dict[int, vz.Trial] = attr.ib(
      init=False, factory=dict, on_setattr=attr.setters.frozen
  )
  # Dictionary of study_guid to ProblemAndTrials.
  prior_studies: dict[str, vz.ProblemAndTrials] = attr.ib(
      init=False, factory=dict
  )

  def __str__(self) -> str:
    return (
        f'InRamPolicySupporter(study_guid={self.study_guid},'
        f' num_trials={len(self.trials)})'
    )

  @property
  def trials(self) -> Sequence[vz.Trial]:
    return list(self._trials.values())

  def study_descriptor(self) -> vz.StudyDescriptor:
    return vz.StudyDescriptor(
        self.study_config,
        guid=self.study_guid,
        max_trial_id=max(self._trials.keys()) if self._trials else 0,
    )

  def GetStudyConfig(
      self, study_guid: Optional[str] = None
  ) -> vz.ProblemStatement:
    if study_guid is None or study_guid == self.study_guid:
      return self.study_config
    elif study_guid in self.prior_studies:
      return self.prior_studies[study_guid].problem
    else:
      raise KeyError(f'Study does not exist in InRamSupporter: {study_guid}')

  def GetTrials(
      self,
      *,
      study_guid: Optional[str] = None,
      trial_ids: Optional[Iterable[int]] = None,
      min_trial_id: Optional[int] = None,
      max_trial_id: Optional[int] = None,
      status_matches: Optional[vz.TrialStatus] = None,
      include_intermediate_measurements: bool = True,
  ) -> List[vz.Trial]:
    """Returns trials by reference to allow changing their status and attributes."""
    self.CheckCancelled('GetTrials')
    if study_guid is not None and study_guid != self.study_guid:
      if study_guid not in self.prior_studies:
        raise KeyError(f'Study Guid does not exist {study_guid}')

      candidate_trials = self.prior_studies[study_guid].trials
    else:
      candidate_trials = self.trials

    trial_id_set = None
    if trial_ids is not None:
      trial_id_set = set(trial_ids)
    output: List[vz.Trial] = []
    for t in candidate_trials:
      if status_matches is not None and t.status != status_matches:
        continue
      if min_trial_id is not None and t.id < min_trial_id:
        continue
      if max_trial_id is not None and t.id > max_trial_id:
        continue
      if trial_id_set is not None and t.id not in trial_id_set:
        continue
      # NOTE: we ignore include_intermediate_measurements and always enclude
      # them.  That should be safe, and avoids a nasty conflict with the
      # pass-by-reference philosophy for Trials (you can't delete the
      # intermediate measurements without deleting them everywhere, and you
      # can't copy the trial without breaking desired reference connections).
      output.append(t)
    return output

  def CheckCancelled(self, note: Optional[str] = None) -> None:
    pass

  def TimeRemaining(self) -> datetime.timedelta:
    return datetime.timedelta(seconds=100.0)

  def _UpdateMetadata(self, delta: vz.MetadataDelta) -> None:
    """Assign metadata to trials."""
    for ns in delta.on_study.namespaces():
      self.study_config.metadata.abs_ns(ns).update(delta.on_study.abs_ns(ns))

    for tid, metadatum in delta.on_trials.items():
      if not tid > 0:
        raise ValueError(f'Bad Trial Id: {tid}')
      for ns in metadatum.namespaces():
        self._trials[tid].metadata.abs_ns(ns).update(metadatum.abs_ns(ns))

  # TODO: Return `count` trials for multi-objectives, when
  # `count` exceeds the size of the Pareto frontier.
  def GetBestTrials(self, *, count: Optional[int] = None) -> List[vz.Trial]:
    """Returns optimal trials.

    Single-objective study:
      * If `count` is unset, returns all tied top trials.
      * If `count` is set, returns top `count` trials, breaking ties
           arbitrarily.

    Multi-objective study:
      * If `count` is unset, returns all Pareto optimal trials.
      * If `count` is set, returns up to `count` of Pareto optimal trials that
          are arbitrarily selected.

    Args:
      count: If unset, returns Pareto optimal trials only. If set, returns the
        top "count" trials.

    Returns:
      Best trials.
    """
    if not self.study_config.metric_information.of_type(
        vz.MetricType.OBJECTIVE):
      raise ValueError('Requires at least one objective metric.')

    # Add safety warping and remove safety metrics from conversion.
    safety_checker = multimetric.SafetyChecker(
        self.study_config.metric_information
    )
    warped_trials = safety_checker.warp_unsafe_trials(
        copy.deepcopy(self.trials)
    )
    config_without_safe = copy.deepcopy(self.study_config)
    config_without_safe.metric_information = (
        self.study_config.metric_information.exclude_type(vz.MetricType.SAFETY)
    )
    converter = converters.TrialToArrayConverter.from_study_config(
        config_without_safe,
        flip_sign_for_minimization_metrics=True,
        dtype=np.float32,
    )

    if self.study_config.is_single_objective:
      # Single metric: Sort and take top N.
      count = count or 1  # Defaults to 1.
      labels = converter.to_labels(warped_trials).squeeze()
      sorted_idx = np.argsort(-labels)  # np.argsort sorts in ascending order.
      return list(np.asarray(self.trials)[sorted_idx[:count]])
    else:
      algorithm = multimetric.FastParetoOptimalAlgorithm()
      is_optimal = algorithm.is_pareto_optimal(
          points=converter.to_labels(warped_trials)
      )
      return list(np.asarray(self.trials)[is_optimal][:count])

  def SetPriorStudy(
      self, study: vz.ProblemAndTrials, study_guid: Optional[str] = None
  ) -> str:
    if study_guid is None:
      # Assign study_guid using unique identifier.
      study_guid = f'prior_{uuid.uuid1()}'
      if study_guid in self.prior_studies:
        raise RuntimeError(f'Cannot set unique id: {study_guid}')

    if study_guid in self.prior_studies:
      logging.warning('Prior study already exists with guid %s ', study_guid)

    self.prior_studies[study_guid] = study
    return study_guid

  def AddTrials(self, trials: Sequence[vz.Trial]) -> None:
    """Assigns ids to the trials and add them to the supporter (by reference).

    New IDs are always assigned in increasing order from the max id unless:
      * If an incoming Trial has an ID that matches an ACTIVE Trial, the ACTIVE
    Trial is replaced and the COMPLETED Trial keeps its ID.
      * If an incoming Trial has an ID that matches an COMPLETE Trial, no
    updates are done and there is a warning.

    Args:
      trials: Incoming Trials to add.
    """
    existing_trial_ids = self._trials.keys()
    next_trial_id = max(existing_trial_ids) + 1 if existing_trial_ids else 1
    for trial in trials:
      if trial.id in existing_trial_ids:
        if self._trials[trial.id].status == vz.TrialStatus.ACTIVE:
          self._trials[trial.id] = trial
        elif self._trials[trial.id].status == vz.TrialStatus.COMPLETED:
          logging.warning(
              'COMPLETED Trial %s cannot be overwritten by %s',
              self._trials[trial.id],
              trial,
          )
        continue

      trial.id = next_trial_id
      self._trials[next_trial_id] = trial
      next_trial_id += 1

  def AddSuggestions(
      self, suggestions: Iterable[vz.TrialSuggestion]) -> Sequence[vz.Trial]:
    """Assigns ids to suggestions and add them to the study."""
    trials = []
    for suggestion in suggestions:
      # Assign temporary ids, which will be overwritten by AddTrials() method.
      trials.append(suggestion.to_trial(0))
    self.AddTrials(trials)
    return trials

  def SuggestTrials(self, algorithm: policy.Policy,
                    count: int) -> Sequence[vz.Trial]:
    """Suggest and add new trials."""

    decisions = algorithm.suggest(
        policy.SuggestRequest(
            study_descriptor=self.study_descriptor(), count=count))
    self._UpdateMetadata(decisions.metadata)
    return self.AddSuggestions([
        vz.TrialSuggestion(d.parameters, metadata=d.metadata)
        for d in decisions.suggestions
    ])
