"""Wraps Designer into Policy."""
import abc
import json
from typing import Callable, Generic, Sequence, Type, TypeVar

from absl import logging
from vizier import algorithms as vza
from vizier import pythia
from vizier import pyvizier as vz
from vizier.interfaces import serializable

_T = TypeVar('_T')
Factory = Callable[[vz.StudyConfig], _T]


class DesignerPolicy(pythia.Policy):
  """Wraps a Designer into pythia Policy."""

  def __init__(self, supporter: pythia.PolicySupporter,
               designer_factory: Factory[vza.Designer]):
    self._supporter = supporter
    self._designer_factory = designer_factory

  def suggest(self, request: pythia.SuggestRequest) -> pythia.SuggestDecisions:
    self._designer = self._designer_factory(request.study_config)
    new_trials = self._supporter.GetTrials(
        status_matches=vz.TrialStatus.COMPLETED)
    self._designer.update(vza.CompletedTrials(new_trials))

    return pythia.SuggestDecisions.from_trials(
        self._designer.suggest(request.count))


class _SerializableDesignerPolicyBase(pythia.Policy,
                                      serializable.PartiallySerializable,
                                      Generic[_T], abc.ABC):
  """Wraps a PartiallySerializable into pythia Policy."""

  _ns_designer = 'designer'

  def __init__(self,
               supporter: pythia.PolicySupporter,
               designer_factory: Callable[[vz.StudyConfig], _T],
               *,
               ns_root: str = 'designer_policy_v0',
               verbose: int = 0):
    """Init.

    Args:
      supporter:
      designer_factory:
      ns_root: Root of the namespace where policy state is stored.
      verbose: Logging verbosity.
    """
    self._supporter = supporter
    self._designer_factory = designer_factory
    self._ns_root = ns_root
    self._incorporated_trial_ids = set()
    self._reference_study_config = supporter.GetStudyConfig()
    self._verbose = verbose
    self._designer = None

  @property
  def designer(self) -> _T:
    if self._designer is None:
      raise ValueError('`self._designer` has not been initialized!'
                       'Use self._create_designer(..) to initialize it.')
    return self._designer

  @abc.abstractmethod
  def _create_designer(self, designer_metadata: vz.Metadata) -> _T:
    """Creates a new Designer by restoring the state from `designer_metadata`.

    Args:
      designer_metadata:

    Returns:
      New Designer object.

    Raises:
      DecodeError: `designer_metadata` does not contain valid information
        to restore a Designer state.
    """
    pass

  # TODO: Use timestamps to avoid metadata blowup.
  def load(self, md: vz.Metadata) -> None:
    if 'incorporated_trial_ids' in md:
      try:
        self._incorporated_trial_ids = set(
            json.loads(md['incorporated_trial_ids']))
      except json.JSONDecodeError as e:
        raise serializable.HarmlessDecodeError from e
    else:
      raise serializable.HarmlessDecodeError()
    self._log(
        self._verbose,
        'Successfully recovered the policy state, which incorporated %s trials',
        len(self._incorporated_trial_ids))
    self._designer = self._create_designer(md.ns(self._ns_designer))

  def _log(self, level, *args):
    if level <= self._verbose:
      logging.info(*args)

  def _initialize_designer(self, study_config: vz.StudyConfig) -> None:
    """Populates `self._designer` with a new designer.

    This method catches all DecodeErrors and guarantees that `self.designer`
    does not raise an Exception.

    Args:
      study_config:

    Raises:
      ValueError: If study_config is differerent from the initially
        receieved study_config.
    """
    if self._designer is not None:
      # When the same policy object is maintained in RAM, prefer keeping
      # the designer object over restoring the state from metadata.
      # TOCONSIDER: Adding a boolean knob to turn off this behavior.
      self._log(
          2,
          "Policy already has a designer, and won't attempt to load from metadata"
      )
      return
    elif self._reference_study_config != study_config:
      raise ValueError(
          f'{type(self)} cannot be re-used for different study configs!'
          f'Policy: {self}, previous study: {self._reference_study_config} '
          f'new study: {study_config}')

    metadata = study_config.metadata.ns(self._ns_root)
    try:
      self.load(metadata)
      self._log(1, 'Succesfully decoded all states!',
                len(self._incorporated_trial_ids))
    except serializable.DecodeError as e:
      self._log(1, 'Failed to decode state. %s', e)
      self._designer = self._designer_factory(study_config)
      self._incorporated_trial_ids = set()

  def dump(self) -> vz.Metadata:
    """Dump state.

    Returns:
      Metadata has the following namespace hierarchy:
        Namespace([self._ns_root]): contains the policy's state.
        Namespace([self._ns_root, self._ns_designer]: contains the designer's
          state.
    """

    md = vz.Metadata()
    md.ns(self._ns_designer).attach(self.designer.dump())
    # TODO: Storing every id is inefficient. Optimize this.
    md['incorporated_trial_ids'] = json.dumps(
        list(self._incorporated_trial_ids))
    return md

  def _get_new_trials(self, max_trial_id: int) -> Sequence[vz.CompletedTrial]:
    """Returns new completed trials that designer should be updated with."""
    if len(self._incorporated_trial_ids) == max_trial_id:
      # no trials need to be loaded.
      return []
    all_trial_ids = set(range(1, max_trial_id + 1))
    trial_ids_to_load = all_trial_ids - self._incorporated_trial_ids

    trials = self._supporter.GetTrials(
        trial_ids=trial_ids_to_load, status_matches=vz.TrialStatus.COMPLETED)
    self._log(
        1, 'Loaded %s completed trials out of %s total unseen trials. '
        'Max trial id is %s.', len(trials), len(trial_ids_to_load),
        max_trial_id)
    return trials

  def suggest(self, request: pythia.SuggestRequest) -> pythia.SuggestDecisions:
    # Note that we can avoid O(Num trials) dependency in the standard scenario,
    # by storing only the last element in a consecutive sequence, e.g.,
    # instead of storing [1,2,3,4,11,12,13,21], store: [4,13,21], but
    # we keep things simple in this pseudocode.
    self._initialize_designer(request.study_descriptor.config)
    new_trials = self._get_new_trials(request.study_descriptor.max_trial_id)
    self.designer.update(vza.CompletedTrials(new_trials))
    self._incorporated_trial_ids |= set(t.id for t in new_trials)

    self._log(
        1, 'Updated with %s trials. Designer has seen a total of %s trials.',
        len(new_trials), len(self._incorporated_trial_ids))

    with self._supporter.MetadataUpdate() as mu:
      # pylint: disable=protected-access
      # TODO: Improve the MetadataUpdateContext API.
      mu._delta.on_study.ns(self._ns_root).attach(self.dump())

    return pythia.SuggestDecisions.from_trials(
        self.designer.suggest(request.count))


class PartiallySerializableDesignerPolicy(
    _SerializableDesignerPolicyBase[vza.PartiallySerializableDesigner]):
  """Wraps a PartiallySerializableDesigner."""

  def _create_designer(
      self,
      designer_metadata: vz.Metadata) -> vza.PartiallySerializableDesigner:
    designer = self._designer_factory(self._reference_study_config)
    designer.load(designer_metadata)
    return designer


class SerializableDesignerPolicy(
    _SerializableDesignerPolicyBase[vza.SerializableDesigner]):
  """Wraps a SerializableDesigner."""

  def __init__(self,
               supporter: pythia.PolicySupporter,
               designer_factory: Factory[vza.SerializableDesigner],
               designer_cls: Type[vza.SerializableDesigner],
               *,
               ns_root: str = 'designer_policy_v0',
               verbose: int = 0):
    """Init.

    Args:
      supporter:
      designer_factory: Used when designer state cannot be restored.
      designer_cls: Type name of the designer. Its load() classmethod is called
        to restore the designer state.
      ns_root: Root of the namespace where policy state is stored.
      verbose: Logging verbosity.
    """
    super().__init__(
        supporter, designer_factory, ns_root=ns_root, verbose=verbose)
    self._designer_cls = designer_cls

  def _create_designer(
      self, designer_metadata: vz.Metadata) -> vza.SerializableDesigner:
    return self._designer_cls.recover(designer_metadata)
