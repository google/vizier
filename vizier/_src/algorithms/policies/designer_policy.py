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

"""Wrappers for Designer into Policy.

This file should contain the core logic to run Designers
efficiently for testing, benchmarking, or production purposes. Typical
usage is via PolicySuggester.from_designer_factory.
"""

import abc
from typing import Generic, Optional, Type, TypeVar

from absl import logging
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.algorithms.core import abstractions as vza
from vizier._src.algorithms.policies import trial_caches
from vizier.interfaces import serializable


_T = TypeVar('_T')

_INCOPORATED_COMPLETED_TRIALS_IDS = 'incorporated_completed_trials_ids'


class DesignerPolicy(pythia.Policy):
  """Wraps a Designer into a pythia Policy.

  > IMPORTANT: If your Designer class is (partially) serializable, use
  > (Partially)SerializableDesignerPolicy instead.

  When a Designer cannot be (partially) serialized, we create a new Designer
  instance at the start of each `suggest()` call. The most recently used
  Designer instance can be saved in self._designer. However, it is for
  interactive analysis/debugging purposes only and never used in
  future `suggest()` calls.
  """

  def __init__(
      self,
      supporter: pythia.PolicySupporter,
      designer_factory: vza.DesignerFactory[vza.Designer],
      *,
      policy_name: str = 'DesignerPolicy',
      use_seeding: bool = True,
  ):
    """Init.

    Args:
      supporter:
      designer_factory:
      policy_name: A user visible name.
      use_seeding:
    """
    self._supporter = supporter
    self._designer_factory = designer_factory
    self._policy_name = policy_name
    self._use_seeding = use_seeding

    if self._use_seeding:
      self.suggest = pythia.seed_with_default(self.suggest)

  def suggest(self, request: pythia.SuggestRequest) -> pythia.SuggestDecision:
    logging.info(
        'Processing request for study guid %s, with max trial id %s',
        request.study_guid,
        request.max_trial_id,
    )
    designer = self._designer_factory(request.study_config)
    logging.info(
        'Study guid %s: Designer created. Getting trials from Vizier service.',
        request.study_guid,
    )
    completed = self._supporter.GetTrials(
        status_matches=vz.TrialStatus.COMPLETED
    )
    logging.info(
        'Study guid %s: got %s COMPLETED trials',
        request.study_guid,
        len(completed),
    )
    active = self._supporter.GetTrials(status_matches=vz.TrialStatus.ACTIVE)
    logging.info(
        'Study guid %s: got %s ACTIVE trials', request.study_guid, len(active)
    )
    logging.info(
        'Study guid %s: Updating designer with trials.', request.study_guid
    )
    designer.update(vza.CompletedTrials(completed), vza.ActiveTrials(active))
    self._designer = designer  # saved for debugging purposes only.
    logging.info(
        'Study guid %s: Asking designer for %s suggestions.',
        request.study_guid,
        request.count,
    )
    return pythia.SuggestDecision(
        designer.suggest(request.count), metadata=vz.MetadataDelta()
    )

  def early_stop(
      self, request: pythia.EarlyStopRequest
  ) -> pythia.EarlyStopDecisions:
    raise NotImplementedError(
        'DesignerPolicy does not support the early_stop() method.'
    )

  @property
  def name(self) -> str:
    return self._policy_name


class _SerializableDesignerPolicyBase(
    pythia.Policy, serializable.PartiallySerializable, Generic[_T], abc.ABC
):
  """Partially implemented class for wrapping a (Partially)SerializableDesigner.

  Inherited by (Partially)SerializableDesignerPolicy which is fully implemented.

  (Partially)SerializableDesignerPolicy maintains a synchronized state between
  the set of trial ids that were passed to Designer via `update()` call, and
  the Designer instance that was used for during last `suggest()` call.

  The policy's `dump()` contains the trial ids passed to update() and the
  result of `dump()` called on the wrapped Designer.

  Unlike in the basic DesignerPolicy class, this class tries to minimize
  the computation by following these steps in order:
    * Re-use the saved Designer instance from the last `suggest()` call.
    * `load()` the Designer state from the study-level metadata.
    * If either of the first two steps succeeds, then we update the Designer
    with **newly** completed and **all** active (pending) trials.
    * Otherwise, we update the Designer with all trials.

  > NOTE: This Policy itself is PartiallySerializable, i.e. it implements its
  own 'load' and 'dump' which wrap the designer_factory designer's 'load' and
  'dump' methods.
  """

  _ns_designer = 'designer'
  _ns_cache = 'cache'

  def __init__(
      self,
      problem_statement: vz.ProblemStatement,
      supporter: pythia.PolicySupporter,
      designer_factory: vza.DesignerFactory[_T],
      *,
      ns_root: str = 'designer_policy_v0',
      verbose: int = 0,
      seed: Optional[int] = None,
      policy_name: str = '_SerializableDesignerPolicyBase',
  ):
    """Init.

    Args:
      problem_statement:
      supporter:
      designer_factory: Factory function to create.
        (Partially)SerializableDesigner designers.
      ns_root: Root of the namespace where policy state is stored.
      verbose: Logging verbosity.
      seed: Random seed to be passed to the designer factory.
      policy_name: A user visible name.
    """
    self._supporter = supporter
    self._designer_factory = designer_factory
    self._ns_root = ns_root
    self._cache: trial_caches.IdDeduplicatingTrialLoader = (
        trial_caches.IdDeduplicatingTrialLoader(
            supporter, include_intermediate_measurements=False
        )
    )
    self._problem_statement = problem_statement
    self._verbose = verbose
    self._designer = None
    self._seed = seed
    self._policy_name = policy_name

  def suggest(self, request: pythia.SuggestRequest) -> pythia.SuggestDecision:
    """Perform a suggest operation.

    The order of operations is:
    1. Initialize the designer and load its state from metadata.
    2. Update the designer with newly completed trials.
    3. Generate suggestions from the designer.
    4. Dump the state of the designer and store it in metadata.

    Arguments:
      request: Pythia suggestion request objects.

    Returns:
      The suggestions from the designer.
    """
    # Note that we can avoid O(Num trials) dependency in the standard scenario,
    # by storing only the last element in a consecutive sequence, e.g.,
    # instead of storing [1,2,3,4,11,12,13,21], store: [4,13,21], but
    # we keep things simple in this pseudocode.
    self._initialize_designer(request.study_config)
    new_completed_trials = self._cache.get_newly_completed_trials(
        request.max_trial_id
    )
    active_trials = self._cache.get_active_trials()
    self.designer.update(
        completed=vza.CompletedTrials(new_completed_trials),
        all_active=vza.ActiveTrials(active_trials),
    )
    logging.info(
        (
            'Updated with %s new completed trials and %s new active trials. '
            'Trial Id cache state: %s'
        ),
        len(new_completed_trials),
        len(active_trials),
        str(self._cache),
    )
    metadata_delta = vz.MetadataDelta()
    # During the 'suggest' call the designer's state could be changed, therefore
    # the state is dumped and stored only after 'suggest' was called.
    suggestions = pythia.SuggestDecision(
        self.designer.suggest(request.count), metadata=metadata_delta
    )
    metadata_delta.on_study.ns(self._ns_root).attach(self.dump())
    return suggestions

  def early_stop(
      self, request: pythia.EarlyStopRequest
  ) -> pythia.EarlyStopDecisions:
    raise NotImplementedError(
        'PartiallySerializableDesignerPolicy does not implement early_stop().'
    )

  @property
  def designer(self) -> _T:
    if self._designer is None:
      raise ValueError(
          '`self._designer` has not been initialized!'
          'Use self._restore_designer(..) to initialize it.'
      )
    return self._designer

  @abc.abstractmethod
  def _restore_designer(self, designer_metadata: vz.Metadata) -> _T:
    """Creates a new Designer by restoring the state from `designer_metadata`.

    Args:
      designer_metadata:

    Returns:
      New Designer object.

    Raises:
      DecodeError: `designer_metadata` does not contain valid information
        to restore a Designer state.
    """

  def _initialize_designer(
      self, problem_statement: vz.ProblemStatement
  ) -> None:
    """Guarantees that `self._designer` is populated.

    This method guarantees that after a successful call, `self.designer` does
    not raise an Exception.
    This method catches all DecodeErrors.

    Args:
      problem_statement:

    Raises:
      ValueError: If problem_statement is different from the initially
        received problem_statement.
    """
    if self._designer is not None:
      # When the same policy object is maintained in RAM, prefer keeping
      # the designer object over restoring the state from metadata.
      # TOCONSIDER: Adding a boolean knob to turn off this behavior.
      logging.log_if(
          logging.INFO,
          (
              'Policy already has a designer. '
              'It will not attempt to load from metadata.'
          ),
          self._verbose >= 2,
      )
      return
    elif self._problem_statement != problem_statement:
      raise ValueError(
          f'{type(self)} cannot be re-used for different study configs!'
          f'Policy: {self}, previous study: {self._problem_statement} '
          f'new study: {problem_statement}'
      )

    metadata = problem_statement.metadata.ns(self._ns_root)
    try:
      self.load(metadata)
      logging.log_if(
          logging.INFO, 'Successfully decoded all states!', self._verbose >= 1
      )
    except serializable.DecodeError as e:
      logging.log_if(
          logging.INFO, 'Failed to decode state. %s', self._verbose >= 1, e
      )
      # Restart the state of the policy.
      self._designer = self._designer_factory(
          problem_statement, seed=self._seed
      )
      self._cache.clear()

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
    md.ns(self._ns_cache).attach(self._cache.dump())
    return md

  def load(self, md: vz.Metadata) -> None:
    self._cache.load(md.ns(self._ns_cache))
    self._designer = self._restore_designer(md.ns(self._ns_designer))

  @property
  def name(self) -> str:
    return self._policy_name


# TODO: Rewrite base to separate incorporated trial logic from
# serialization logic. For now, it simply disables the serialization.
class InRamDesignerPolicy(_SerializableDesignerPolicyBase[vza.Designer]):
  """Wraps a Designer in RAM for efficient updates()."""

  # Override restore() to simply return the designer.
  def _restore_designer(self, designer_metadata: vz.Metadata) -> vza.Designer:
    if self._designer is None:
      raise ValueError(
          '`self._designer` has not been initialized!'
          ' Call self._initialize_designer(..) first.'
      )
    return self._designer

  # Override dump() since this is in ram.
  def dump(self) -> vz.Metadata:
    return vz.Metadata()


class PartiallySerializableDesignerPolicy(
    _SerializableDesignerPolicyBase[vza.PartiallySerializableDesigner]
):
  """Wraps a PartiallySerializableDesigner."""

  def _restore_designer(
      self, designer_metadata: vz.Metadata
  ) -> vza.PartiallySerializableDesigner:
    designer = self._designer_factory(self._problem_statement)
    designer.load(designer_metadata)
    return designer


class SerializableDesignerPolicy(
    _SerializableDesignerPolicyBase[vza.SerializableDesigner]
):
  """Wraps a SerializableDesigner."""

  def __init__(
      self,
      problem_statement: vz.ProblemStatement,
      supporter: pythia.PolicySupporter,
      designer_factory: vza.DesignerFactory[vza.SerializableDesigner],
      designer_cls: Type[vza.SerializableDesigner],
      *,
      ns_root: str = 'designer_policy_v0',
      verbose: int = 0,
  ):
    """Init.

    Args:
      problem_statement:
      supporter:
      designer_factory: Used when designer state cannot be restored.
      designer_cls: Type name of the designer. Its load() classmethod is called
        to restore the designer state.
      ns_root: Root of the namespace where policy state is stored.
      verbose: Logging verbosity.
    """
    super().__init__(
        problem_statement,
        supporter,
        designer_factory,
        ns_root=ns_root,
        verbose=verbose,
    )
    self._designer_cls = designer_cls

  def _restore_designer(
      self, designer_metadata: vz.Metadata
  ) -> vza.SerializableDesigner:
    return self._designer_cls.recover(designer_metadata)
