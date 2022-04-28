"""Base class for all PythiaPolicies."""

import abc
from collections import abc as cabc
from typing import Any, FrozenSet, Iterable, List, Optional, Type, TypeVar

import attr
from vizier import pyvizier as vz

_T = TypeVar('_T')


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
  metadata: vz.Metadata = attr.ib(
      default=attr.Factory(vz.Metadata),
      validator=[attr.validators.instance_of(vz.Metadata)],
      on_setattr=attr.setters.validate)

  # TODO: Add a proper support for this in the service side.
  predicted_final_measurement: Optional[vz.Measurement] = attr.ib(
      default=None,
      validator=attr.validators.optional(
          attr.validators.instance_of(vz.Measurement)),
      on_setattr=attr.setters.validate)


@attr.define
class EarlyStopRequest:
  """Early stopping request.

  Attributes:
    study_guid:
    trial_ids: Trials to be considered for stopping. Used as hints.
    study_config:
    checkpoint_dir: If the policy wishes to use a checkpoint, then this is the
      path to find one.
  """
  _study_descriptor: vz.StudyDescriptor = attr.field(
      validator=attr.validators.instance_of(vz.StudyDescriptor))
  trial_ids: FrozenSet[int] = attr.field(
      default=attr.Factory(frozenset),
      validator=attr.validators.instance_of(FrozenSet),
      converter=frozenset)

  checkpoint_dir: Optional[str] = attr.field(
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(str)))

  @property
  def study_guid(self) -> str:
    return self._study_descriptor.guid

  @property
  def study_config(self) -> vz.StudyConfig:
    return self._study_descriptor.config


@attr.define(init=True)
class SuggestDecision:
  """Suggest decision.

  Currently only supports suggesting new points to be evaluated. In the future,
  we want to support other operations such as: freeze/thaw, or warm-start.

  Attributes:
    parameters: Trial parameters to be newly evaluated.
    metadata:
  """
  parameters: vz.ParameterDict = attr.field(
      init=True,
      validator=attr.validators.instance_of(vz.ParameterDict),
      converter=vz.ParameterDict)
  metadata: vz.Metadata = attr.field(
      init=True,
      factory=vz.Metadata,
      validator=[attr.validators.instance_of(vz.Metadata)])

  def to_trial(self, trial_id: int) -> vz.Trial:
    return vz.Trial(
        id=trial_id, parameters=self.parameters, metadata=self.metadata)


class SuggestDecisions(cabc.Sequence[SuggestDecision]):
  """Sequence of suggestions."""

  def __init__(self, items: Iterable[SuggestDecision] = tuple()):
    self._container = tuple(items)

  def __len__(self) -> int:
    return len(self._container)

  def __getitem__(self, index: int) -> SuggestDecision:
    return self._container[index]

  # TODO: rename to create_new_trials.
  @classmethod
  def from_trials(cls: Type[_T], trials: Iterable[vz.TrialSuggestion]) -> _T:
    return [SuggestDecision(t.parameters, t.metadata) for t in trials]


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
  """
  study_descriptor: vz.StudyDescriptor = attr.field(
      validator=attr.validators.instance_of(vz.StudyDescriptor),
      on_setattr=attr.setters.frozen)

  count: int = attr.field(
      validator=[attr.validators.instance_of(int), _is_positive],
      on_setattr=attr.setters.validate)

  checkpoint_dir: Optional[str] = attr.field(
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(str)),
      on_setattr=attr.setters.validate)

  @property
  def study_config(self) -> vz.StudyConfig:
    return self.study_descriptor.config

  def study_guid(self) -> str:
    return f'{self.study_descriptor.guid}'


class Policy(abc.ABC):
  """Interface for Pythia Policy subclasses.

  Most Policy subclasses would wish to take `PolicySupporter` object in the
  `__init__` function. `PolicySupporter` provides an abstraction for how
  `Policy` reads more information about the Study in question beyond the basic
  information available in `SuggestRequest` and `EarlyStopRequest`. It allows
  the `Policy` to be compatible in multiple environments.
  """

  @abc.abstractmethod
  def suggest(self, request: SuggestRequest) -> SuggestDecisions:
    """Compute suggestions that Vizier will eventually hand to the user.

    Args:
      request:

    Returns:
      A list of Trials that will be passed on to the user.
      (See caveats in the SuggestionAnswer proto.)

    Raises:
      TemporaryPythiaError:  Generic retryable error.
      InactivateStudyError:  Raise this to inactivate the Study (non-retryable
        error).
        E.g. if this Policy cannot handle this StudyConfig.
        E.g. if this StudyConfig is somehow invalid.
        E.g. if this no more suggestions will ever be generated.
      CachedPolicyIsStaleError: Causes the computation to be restarted with a
        freshly constructed Policy instance.  It is incorrect to raise
        this on the first use of a Policy; the Study will be inactivated.

    NOTE: Trials should not have Trial.id set; these are new Trials rather than
      modifications of existing Trials.
    """
    pass

  def early_stop(self, request: EarlyStopRequest) -> List[EarlyStopDecision]:
    """Decide which Trials Vizier should stop.

    This returns a list of decisions on on-going Trials.
    Args:
      request:

    Returns:
      List of length up to `len(request.trial_ids)`. No decision is treated
      as "do not stop".

    Raises:
      TemporaryPythiaError:  Generic retryable error.
      InactivateStudyError: If this Pythia is inappropriate for the StudyConfig.
        (Non-retryable error.)  E.g. raise this if your Policy does not
        support MakeEarlyEarlyStopDecisions().
      CachedPolicyIsStaleError: Causes the computation to be restarted with a
        freshly constructed Policy instance.  It is incorrect to raise
        this on the first use of a Policy; the Study will be inactivated.
    """
    del request
    return []

  @property
  def should_be_cached(self) -> bool:
    """Returns True if it's safe & worthwhile to cache this Policy in RAM.

    This is called after MakeEarlyEarlyStopDecisions() and/or MakeSuggestions().
    If True, the policy may be stored in RAM (at least for a while), and state
    may be preserved for the next time that Study makes it to that Pythia
    server.
    """
    return False
