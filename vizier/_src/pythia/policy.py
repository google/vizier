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

"""Base class for all PythiaPolicies."""

import abc
from typing import Any, FrozenSet, Optional

import attr
from vizier._src.pyvizier.pythia import study
from vizier._src.pyvizier.shared import base_study_config
from vizier._src.pyvizier.shared import trial


def _is_positive(instance: Any, attribute: Any, value: Any):
  del instance, attribute
  if value <= 0:
    raise ValueError(f'value must be positive! given: {value}')


def _not_empty(instance: Any, attribute, value: str):
  del instance, attribute
  if not value:
    raise ValueError(f'value must be not nullable! given: {value}')


@attr.define
class EarlyStopDecision:
  """Stopping decision on a single trial.

  Attributes:
    id: Trial's id.
    reason: Explanations for how `should_stop` value was determined.
    should_stop:
    metadata:
    predicted_final_measurement: This is added to the Trial so that other
      algorithms can treat it as if it is the observed final measurement.
  """
  id: int = attr.ib(
      validator=[attr.validators.instance_of(int), _is_positive],
      on_setattr=attr.setters.validate)
  # TODO: Record this in DB even when `should_stop` == False.
  reason: str = attr.ib(
      validator=[attr.validators.instance_of(str), _not_empty],
      on_setattr=attr.setters.validate,
      converter=str)
  should_stop: bool = attr.ib(
      default=True,
      validator=[attr.validators.instance_of(bool)],
      on_setattr=attr.setters.validate)

  # TODO: Add a proper support for this in the service side.
  # NOTE: As of 2022Q3, the standard deviation field of Metrics in this value
  #   are ignored (i.e. $predicted_final_measurement.metrics[].std).
  predicted_final_measurement: Optional[trial.Measurement] = attr.ib(
      default=None,
      validator=attr.validators.optional(
          attr.validators.instance_of(trial.Measurement)
      ),
      on_setattr=attr.setters.validate,
  )


@attr.define
class EarlyStopDecisions:
  """This is the output of the Policy.early_stop() method.

  Attributes:
    decisions: For some Trials, a decision as to whether it should be stopped.
    metadata: Metadata that's associated with the Study or with already existing
      Trials.
  """

  decisions: list[EarlyStopDecision] = attr.field(
      factory=list,
      validator=attr.validators.deep_iterable(
          attr.validators.instance_of(EarlyStopDecision)),
      converter=list)

  metadata: trial.MetadataDelta = attr.field(
      default=attr.Factory(trial.MetadataDelta),
      validator=attr.validators.instance_of(trial.MetadataDelta),
  )


@attr.define
class EarlyStopRequest:
  """Early stopping request.

  Attributes:
    study_guid:
    trial_ids: Trials to be considered for stopping, or None meaning "all
      Trials". This is a hint; it is allowable to consider stopping more or
      fewer trials.
    study_config:
    checkpoint_dir: If the policy wishes to use a checkpoint, then this is the
      path to find one.
    max_trial_id: max(trial.id for all existing Trials in the Study)
  """
  _study_descriptor: study.StudyDescriptor = attr.field(
      kw_only=True, validator=attr.validators.instance_of(study.StudyDescriptor)
  )

  trial_ids: Optional[FrozenSet[int]] = attr.field(
      default=None,
      validator=lambda x, c, v: x is None or isinstance(x, FrozenSet),
      converter=lambda x: None if x is None else frozenset(x))

  checkpoint_dir: Optional[str] = attr.field(
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(str)))

  @property
  def study_guid(self) -> str:
    return self._study_descriptor.guid

  @property
  def study_config(self) -> base_study_config.ProblemStatement:
    return self._study_descriptor.config

  @property
  def max_trial_id(self) -> int:
    return self._study_descriptor.max_trial_id


@attr.define(init=True)
class SuggestDecision:
  """This is the output of the Policy.suggestion() method.

  Attributes:
    suggestions: Trials to be suggested to the user.
    metadata: Metadata that's associated with the Study or with already existing
      Trials.
  """

  suggestions: list[trial.TrialSuggestion] = attr.field(
      init=True,
      validator=attr.validators.deep_iterable(
          attr.validators.instance_of(trial.TrialSuggestion)
      ),
      converter=list,
  )

  metadata: trial.MetadataDelta = attr.field(
      init=True,
      default=attr.Factory(trial.MetadataDelta),
      validator=attr.validators.instance_of(trial.MetadataDelta),
  )


@attr.define
class SuggestRequest:
  """Suggestion Request.

  Attributes:
    study_descriptor: information about the Study.
    study_guid: Study id
    count: A recommendation for how many suggestions should be generated.
    study_config:
    checkpoint_dir: (If set) A system-provided directory where the policy can
      store a checkpoint.
    max_trial_id: max(trial.id for all existing Trials in the Study)
  """
  _study_descriptor: study.StudyDescriptor = attr.field(
      validator=attr.validators.instance_of(study.StudyDescriptor),
      on_setattr=attr.setters.frozen,
      kw_only=True,
  )

  count: int = attr.field(
      validator=[attr.validators.instance_of(int), _is_positive],
      on_setattr=attr.setters.validate,
      kw_only=True)

  checkpoint_dir: Optional[str] = attr.field(
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(str)),
      on_setattr=attr.setters.validate,
      kw_only=True)

  @property
  def study_config(self) -> base_study_config.ProblemStatement:
    return self._study_descriptor.config

  @property
  def study_guid(self) -> str:
    return str(self._study_descriptor.guid)

  @property
  def max_trial_id(self) -> int:
    return self._study_descriptor.max_trial_id


class Policy(abc.ABC):
  """Interface for Pythia Policy subclasses.

  Most Policy subclasses would wish to take `PolicySupporter` object in the
  `__init__` function. `PolicySupporter` provides an abstraction for how
  `Policy` reads more information about the Study in question beyond the basic
  information available in `SuggestRequest` and `EarlyStopRequest`. It allows
  the `Policy` to be compatible in multiple environments.
  """

  @abc.abstractmethod
  def suggest(self, request: SuggestRequest) -> SuggestDecision:
    """Compute suggestions that Vizier will eventually hand to the user.

    Args:
      request:

    Returns:
      A bundle of TrialSuggestions and MetadataDelta that will be passed on to
        the user.  (See caveats in the SuggestionAnswer proto.)

    Raises:
      TemporaryPythiaError:  Generic retryable error.
      InactivateStudyError:  Raise this to inactivate the Study (non-retryable
        error).
        E.g. if this Policy cannot handle this StudyConfig.
        E.g. if this StudyConfig is somehow invalid.
        E.g. if this no more suggestions will ever be generated.
    """

  @abc.abstractmethod
  def early_stop(self, request: EarlyStopRequest) -> EarlyStopDecisions:
    """Decide which Trials Vizier should stop.

    This returns a list of decisions on on-going Trials.
    Args:
      request:

    Returns:
      Decisions about stopping some Trials and metadata changes.  If no decision
      is reported for a Trial, it's treated as "do not stop".

    Raises:
      TemporaryPythiaError:  Generic retryable error.
      InactivateStudyError: If this Pythia is inappropriate for the StudyConfig.
        (Non-retryable error.)  E.g. raise this if your Policy does not
        support MakeEarlyEarlyStopDecisions().
      CachedPolicyIsStaleError: Causes the computation to be restarted with a
        freshly constructed Policy instance.  It is incorrect to raise
        this on the first use of a Policy; the Study will be inactivated.
    """

  @property
  def name(self) -> str:
    """Returns the policy name. Intended for monitoring purposes only."""
    # Derived classes should override this implementation.
    return __class__.__name__

  @property
  def should_be_cached(self) -> bool:
    """Returns True if it's safe & worthwhile to cache this Policy in RAM.

    This is called after MakeEarlyEarlyStopDecisions() and/or MakeSuggestions().
    If True, the policy may be stored in RAM (at least for a while), and state
    may be preserved for the next time that Study makes it to that Pythia
    server.
    """
    return False
