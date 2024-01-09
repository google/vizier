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

"""Cross-Platform Backend."""
import enum
import functools
import getpass
import random
import threading
import time
from typing import Any, Dict, Optional, Sequence, Type, Union

from absl import logging
import attrs
import pyglove as pg
from vizier import pyvizier as vz
from vizier._src.pyglove import algorithms
from vizier._src.pyglove import client
from vizier._src.pyglove import constants
from vizier._src.pyglove import converters
from vizier._src.pyglove import core
from vizier._src.pyglove import pythia as pyglove_pythia
from vizier.client import client_abc


TunerPolicy = pyglove_pythia.TunerPolicy
BuiltinAlgorithm = algorithms.BuiltinAlgorithm

ExpandedStudyName = client.ExpandedStudyName
StudyKey = client.StudyKey


class TunerMode(enum.Enum):
  """Mode for Tuner."""

  # Automatic select primary mode when study is new or study TUNER_ID equals
  # to current tuner ID (for failover) or the primary tuner is not accessible
  # via its BNS. Otherwise select secondary mode.
  AUTO = 0

  # Work as primary tuner, which will host pythia service when algorithm is
  # non-Vizier-builtin. When using Vizier built-in algorithms, all tuners are
  # in secondary mode.
  PRIMARY = 1

  # Work as secondary tuner, which can query or stop a tuner task but not
  # hosting pythia service.
  SECONDARY = 2


# Global policy cache.
_global_policy_cache: Dict[StudyKey, TunerPolicy] = dict()


@attrs.define(auto_attribs=False)
class VizierBackend(pg.tuning.Backend):
  """Vizier backend."""

  # Class-level variables.
  default_owner: str = getpass.getuser()
  default_study_prefix: Optional[str] = None
  tuner_cls: Type[client.VizierTuner] = attrs.field()

  # Instance-level variables.

  #
  # Flags passed from `pg.sample`:
  #

  # Worker group - workers that belong to the same group will share the same
  # Vizier client ID.
  _group: Union[None, int, str] = attrs.field()

  # Max number of examples to sample.
  _num_examples: Optional[int] = attrs.field()

  # Prior study IDs for transfer learning.
  _prior_study_ids: Optional[Sequence[str]] = attrs.field()

  # If True, add the completed trials from prior studies. Otherwise, simply
  # warm up the algorithm using these trials without adding them to the current
  # study.
  _add_prior_trials: bool = attrs.field()

  #
  # Internal states.
  #

  _tuner: client.VizierTuner
  _dna_spec: pg.DNASpec
  _algorithm: pg.geno.DNAGenerator = attrs.field()
  _early_stopping_policy: Optional[pg.tuning.EarlyStoppingPolicy] = (
      attrs.field()
  )

  _study_owner: str = attrs.field()
  _study_name: ExpandedStudyName = attrs.field()
  _converter: converters.VizierConverter = attrs.field()

  _study: client_abc.StudyInterface = attrs.field()
  _suggestion_generator: Any = attrs.field()

  _run_mode: TunerMode = attrs.field()
  _auto_election_thread: Optional[threading.Thread] = attrs.field()
  _is_active: bool = attrs.field()

  def __init__(
      self,
      name: Optional[str],
      group: Union[None, int, str],
      dna_spec: pg.DNASpec,
      algorithm: pg.DNAGenerator,
      metrics_to_optimize: Sequence[str],
      early_stopping_policy: Optional[pg.tuning.EarlyStoppingPolicy] = None,
      num_examples: Optional[int] = None,
      study_owner: Optional[str] = None,
      prior_study_ids: Optional[Sequence[str]] = None,
      add_prior_trials: bool = False,
      is_chief: Optional[bool] = None,
  ):
    self._tuner = self.tuner_cls()

    self._dna_spec = dna_spec
    self._algorithm = algorithm
    self._early_stopping_policy = early_stopping_policy

    self._group = group
    self._num_examples = num_examples
    self._prior_study_ids = prior_study_ids or tuple()
    self._add_prior_trials = add_prior_trials

    self._study_owner = study_owner or self.default_owner
    self._study_name = self._expand_name(name)

    # Set up converter based on the search space and metrics to optimize.
    self._converter = converters.VizierConverter.from_dna_spec(
        dna_spec, metrics_to_optimize
    )

    # Detecting mode based on is_chef flag.
    if is_chief:
      mode = TunerMode.PRIMARY
    elif is_chief is None:
      mode = TunerMode.AUTO
    else:
      mode = TunerMode.SECONDARY

    self._run_mode = mode

    # Setup the Vizier study and the suggestion generator.
    is_chief = self._setup_study()
    self._suggestion_generator = self._create_suggestion_generator()

    # Start Pythia if needed.
    self._auto_election_thread = None
    self._is_active = True

    if self._need_pythia_service:
      pg.logging.info(
          "Study '%s/%s' will be served by Pythia service hosted on %r. "
          'Algorithm=%r, EarlyStoppingPolicy=%r.',
          self._study_owner,
          self._study_name,
          self._get_chief_tuner_id(),
          self._algorithm,
          self._early_stopping_policy,
      )

      if is_chief:
        pg.logging.info(
            "Starting hosting Pythia service for study '%s/%s' as PRIMARY. ",
            self._study_owner,
            self._study_name,
        )
        self._start_pythia()
      if self._run_mode == TunerMode.AUTO:
        self._auto_election_thread = threading.Thread(
            target=self._auto_elect_primary_if_needed
        )
        self._auto_election_thread.start()
    else:
      pg.logging.info(
          'Pythia service is not required as both the search algorithm and '
          'early stopping policy are either Vizier built-in or served from '
          'remote Pythia endpoint. Algorithm=%r, EarlyStoppingPolicy=%r.',
          self._algorithm,
          self._early_stopping_policy,
      )

  @property
  def _host_pythia_algorithm(self) -> bool:
    """Returns True if the algorithm is hosted on Pythia."""
    return not isinstance(self._algorithm, algorithms.PseudoAlgorithm)

  @property
  def _need_pythia_service(self) -> bool:
    """Returns True if pythia service is needed."""
    return self._host_pythia_algorithm

  def _setup_study(self) -> bool:
    """Sets up Vizier study, returns True if current worker is the chief."""
    study_descriptor = f'{self._study_owner}/{self._study_name}'

    try:
      self._study = self._tuner.load_study(self._study_owner, self._study_name)
      pg.logging.info(
          'Connecting tuner to existing study %r... ', study_descriptor
      )

      #
      # Ensure the client-side search space matches with the server-side search
      # space.
      #
      stored_dna_spec = self._get_stored_dna_spec()

      # CustomDecisionPoint could be non-serializable and non-compariable.
      if pg.contains(self._dna_spec, pg.geno.CustomDecisionPoint):
        pg.logging.info(
            'There is a CustomDecisionPoint. Skipping the check to ensure the '
            'client-side search space matches the server-side search space.'
        )
      elif pg.eq(self._dna_spec, stored_dna_spec):
        pg.logging.info(
            'Verified that the client-side search space matches the '
            'server-side.'
        )
      else:
        raise ValueError(
            'The client-side search space is different from the search space '
            'from the study stored in Vizier. Try launching the experiment '
            'with a different study name. '
            f'Diff: {pg.diff(stored_dna_spec, self._dna_spec)}.'
        )

      chief_tuner_id = self._get_chief_tuner_id()

      if self._run_mode != TunerMode.SECONDARY and not self._tuner.ping_tuner(
          chief_tuner_id
      ):
        # NOTE(daiyip): there could be a race condition that multiple workers
        # elect themselves as the new primary, and all of them regard
        # themselves as the elected. When this happens, multiple workers
        # may be hosting the Pythia service for this study. However, the study
        # will use one address (BNS of the latest elected worker) as the
        # Pythia endpoint. Besides, all states are stored in the study,
        # restart of workers will pick up these states from the study.
        # This is done by replaying existing trials to recompute the states of
        # the search algorithm. Therefore, it does not matter which worker is
        # chosen to serve the study, which should work equally well.
        # On the other hand, it is cheap to serve a PythiaService without
        # incoming queries. Therefore, we do not handle this race conditions
        # with expensive distributed locks.
        chief_tuner_id = self._register_self_as_primary()

      is_chief = chief_tuner_id == self._tuner_id
      if self._run_mode == TunerMode.SECONDARY and is_chief:
        raise ValueError(
            f'Inconsistent primary tuner: {self._tuner_id!r} is running as '
            f'secondary but study {study_descriptor!r} indicates otherwise.'
        )
      elif self._run_mode == TunerMode.PRIMARY and not is_chief:
        raise ValueError(
            f'Inconsistent tuner mode: {self._tuner_id!r} is running in '
            f'PRIMARY mode but study {study_descriptor!r} already has a '
            f'different primary {chief_tuner_id!r}.\n'
            'Please check if you are launching different experiments using '
            'the same Vizier study.'
        )

      pg.logging.info(
          'Tuner is running in %s mode with existing study %r.',
          'PRIMARY' if is_chief else 'SECONDARY',
          study_descriptor,
      )

      return is_chief
    except KeyError:
      # Study does not exist.
      if self._run_mode == TunerMode.SECONDARY:
        pg.logging.info(
            'Start tuner as secondary. Waiting for study %r to be created...',
            study_descriptor,
        )
        self._study = self._wait_for_study(self._study_owner, self._study_name)
      else:
        mode_str = 'PRIMARY' if self._run_mode == TunerMode.PRIMARY else 'AUTO'
        pg.logging.info(
            'Start tuner in %s mode. Attempting to create study %r...',
            mode_str,
            study_descriptor,
        )

        # Create a new study.
        problem = self._converter.problem_or_dummy
        problem.metadata.ns(constants.METADATA_NAMESPACE)[
            constants.STUDY_METADATA_KEY_TUNER_ID
        ] = self._tuner_id
        self._study = self._tuner.create_study(
            problem,
            self._converter,
            self._study_owner,
            self._study_name,
            self._algorithm,
            self._early_stopping_policy,
        )

      # Perform post-study actions.
      is_chief = self._tuner_id == self._get_chief_tuner_id()
      if is_chief:
        self._on_study_created()

      pg.logging.info(
          'Tuner is running in %s mode with newly created study %r.',
          'PRIMARY' if is_chief else 'SECONDARY',
          study_descriptor,
      )
      return is_chief

  def _on_study_created(self):
    # For multi-thread scenario, `local_tuner_id` will be the same for
    # all the worker threads, therefore there is a chance that multiple
    # worker threads consider themselves as PRIMARY. This does not matter
    # since there is only one Pythia service shared across them.
    if self._add_prior_trials:
      # Trials are added to the study directly upon creation.
      trials: Sequence[vz.Trial] = self._load_prior_trials()
      for trial in trials:
        self._study._add_trial(trial)  # pylint: disable=protected-access

  def _create_suggestion_generator(self):
    """Creates a suggestion generator."""
    while (
        self._num_examples is None
        or len(list(self._study.trials())) < self._num_examples
    ):
      trials = self._study.suggest(
          count=1, client_id=self._tuner.get_group_id(self._group)
      )
      if not trials:
        return
      for trial in trials:
        yield core.Feedback(trial, self._converter)

  def _start_pythia(self) -> None:
    """Starts Pythia service."""
    # Start pythia service.
    self._tuner.start_pythia_service(_global_policy_cache)

    # Set up the policy.
    self._algorithm.setup(self._dna_spec)
    prior_trials = tuple()
    if not self._add_prior_trials:
      # Trials are added to the algorithm only.
      prior_trials: Sequence[vz.Trial] = self._load_prior_trials()
    policy = self._create_policy(prior_trials)
    # Connect the service with policy.
    self._register(self._study_owner, self._study_name, policy)

  def _create_policy(
      self,
      prior_trials: Sequence[vz.Trial],
  ) -> TunerPolicy:
    """Creates a pythia policy."""
    if prior_trials:

      def get_trial_history(vizier_trials):
        for trial in vizier_trials:
          tuner_trial = core.VizierTrial(self._converter, trial)
          reward = tuner_trial.get_reward_for_feedback(
              self._converter.metrics_to_optimize
          )
          yield (tuner_trial.dna, reward)

      self._algorithm.recover(get_trial_history(prior_trials))

    return TunerPolicy(
        self._tuner.pythia_supporter(self._study),
        self._converter,
        self._algorithm,
        early_stopping_policy=self._early_stopping_policy,
    )

  def _get_stored_dna_spec(self) -> pg.DNASpec:
    metadata = self._study.materialize_problem_statement().metadata.ns(
        constants.METADATA_NAMESPACE
    )
    try:
      return converters.restore_dna_spec(
          metadata[constants.STUDY_METADATA_KEY_DNA_SPEC]
      )
    except KeyError as e:
      raise RuntimeError(
          f'Metadata {constants.STUDY_METADATA_KEY_DNA_SPEC} does not exist '
          f'in study: {self._study.resource_name}.'
      ) from e

  def _get_chief_tuner_id(self) -> str:
    metadata = self._study.materialize_problem_statement().metadata.ns(
        constants.METADATA_NAMESPACE
    )
    try:
      return str(metadata[constants.STUDY_METADATA_KEY_TUNER_ID])
    except KeyError as e:
      raise RuntimeError(
          f'Metadata {constants.STUDY_METADATA_KEY_TUNER_ID} does not exist '
          f'in study: {self._study.resource_name}.'
      ) from e

  @functools.cached_property
  def _tuner_id(self) -> str:
    """Returns the tuner id of current ."""
    return self._tuner.get_tuner_id(self._algorithm)

  def _register_self_as_primary(self) -> str:
    metadata = vz.Metadata()
    metadata.ns(constants.METADATA_NAMESPACE)[
        constants.STUDY_METADATA_KEY_TUNER_ID
    ] = self._tuner_id
    self._study.update_metadata(metadata)
    self._tuner.use_pythia_for_study(self._study)
    return self._tuner_id

  def _auto_elect_primary_if_needed(self) -> None:
    """Automatically elect primary tuner if previous primary is offline."""
    assert self._need_pythia_service
    assert self._run_mode == TunerMode.AUTO

    while self._is_active:
      chief_tuner_id = self._get_chief_tuner_id()
      if not self._tuner.ping_tuner(chief_tuner_id):
        # NOTE(daiyip): there could be a race condition that multiple workers
        # elect themselves as the new primary, and all of them regard
        # themselves as the elected. When this happens, multiple workers
        # may be hosting the Pythia service for this study. However, the study
        # will use one address (BNS of the latest elected worker) as the
        # Pythia endpoint. Besides, all states are stored in the study,
        # restart of workers will pick up these states from the study.
        # This is done by replaying existing trials to recompute the states of
        # the search algorithm. Therefore, it does not matter which worker is
        # chosen to serve the study, which should work equally well.
        # On the other hand, it is cheap to serve a PythiaService without
        # incoming queries. Therefore, we do not handle this race conditions
        # with expensive distributed locks.
        new_chief_tuner_id = self._register_self_as_primary()
        self._is_primary = new_chief_tuner_id == self._tuner_id
        if self._is_primary:
          pg.logging.warning(
              'Primary tuner has been switched from %s to %s.',
              chief_tuner_id,
              new_chief_tuner_id,
          )
          self._start_pythia()
      time.sleep(random.randint(50, 70))

  def next(self) -> pg.tuning.Feedback:
    """Gets the next tuning feedback object."""
    try:
      trial = next(self._suggestion_generator)  # pytype: disable=wrong-arg-types
      return core.Feedback(self._study.get_trial(trial.id), self._converter)
    except StopIteration as e:
      self._is_active = False
      raise e

  def _wait_for_study(
      self, owner: str, name: ExpandedStudyName
  ) -> client_abc.StudyInterface:
    """Wait for the study in a loop."""
    while True:
      try:
        return self._tuner.load_study(owner, name)
      except KeyError:
        logging.info(
            'Study %s (owner=%s) does not exist. Retrying after 10 seconds.',
            name,
            owner,
        )
        time.sleep(10)
      except Exception as e:  # pylint:disable=broad-except
        logging.warn(
            'Could not look up study: %s. Retrying after 60 seconds', e
        )
        time.sleep(60)

  def _load_prior_trials(self) -> list[vz.Trial]:
    trials = []
    for prior in self._prior_study_ids:
      trials.extend(
          self._tuner.load_prior_study(prior)
          .trials(vz.TrialFilter(status=vz.TrialStatus.COMPLETED))
          .get()
      )
    return trials

  def _register(
      self, owner: str, name: ExpandedStudyName, policy: TunerPolicy
  ) -> None:
    """Registers the algorithm for a specific study."""
    study_key = StudyKey(owner, name)

    if study_key in _global_policy_cache:
      existing = _global_policy_cache[study_key]
      if existing.algorithm != policy.algorithm:
        raise ValueError(
            f'Different algorithms are used for the same study {study_key!r}. '
            f'Previous: {existing.algorithm!r}, Current: {policy.algorithm!r}.'
        )
      if existing.early_stopping_policy != policy.early_stopping_policy:
        raise ValueError(
            'Different early stopping policy are used for the same study '
            f'{study_key!r}. Previous: {existing.early_stopping_policy!r}, '
            f'Current: {policy.early_stopping_policy!r}.'
        )

    _global_policy_cache[study_key] = policy

  #
  # Class methods.
  #

  @classmethod
  def use_study_prefix(cls, study_prefix: Optional[str]):
    cls.default_study_prefix = study_prefix or ''

  @classmethod
  def _expand_name(cls, name: Optional[str]) -> ExpandedStudyName:
    """Expand the pyglove study name into the full name in Vizier DB.

    Args:
      name: Name as passed into pyglove.

    Returns:
      Study name to use for Vizier interactions.
    """
    components = []
    if cls.default_study_prefix:
      components.append(cls.default_study_prefix)
    if name:
      components.append(name)
    return ExpandedStudyName('.'.join(components))

  @classmethod
  def _get_study_resource_name(cls, name: str) -> str:
    """Use for testing only."""
    return cls.tuner_cls.load_study(
        owner=cls.default_owner,
        name=ExpandedStudyName(name),
    ).resource_name

  @classmethod
  def poll_result(
      cls, name: str, study_owner: Optional[str] = None
  ) -> pg.tuning.Result:
    """Polls result of a study."""
    return core.Result.from_study(
        cls.tuner_cls.load_study(
            study_owner or cls.default_owner, cls._expand_name(name)
        )
    )
