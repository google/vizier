# Copyright 2023 Google LLC.
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

# Copyright 2019 The PyGlove Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Cross-Platform Backend."""
import enum
import getpass
import time

from typing import Any, Optional, Sequence, Union

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


@attrs.define(auto_attribs=False)
class VizierBackend(pg.tuning.Backend):
  """Vizier backend."""

  # ------------------------------------
  # Class-level methods and variables begin here.
  # ------------------------------------

  default_owner: str = getpass.getuser()
  default_study_prefix: Optional[str] = None
  policy_cache: dict[StudyKey, TunerPolicy] = dict()
  tuner: client.VizierTuner

  @classmethod
  def _wait_for_study(
      cls, owner: str, name: ExpandedStudyName
  ) -> client_abc.StudyInterface:
    """Wait for the study in a loop."""
    while True:
      try:
        return cls.tuner.load_study(owner, name)
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

  @classmethod
  def _load_prior_trials(
      cls, prior_study_ids: Optional[Sequence[str]]
  ) -> list[vz.Trial]:
    trials = []
    for prior in prior_study_ids:
      trials.extend(
          cls.tuner.load_prior_study(prior)
          .trials(vz.TrialFilter(status=vz.TrialStatus.COMPLETED))
          .get()
      )
    return trials

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
  def _register(
      cls, owner: str, name: ExpandedStudyName, policy: TunerPolicy
  ) -> None:
    """Registers the algorithm for a specific study."""
    study_key = StudyKey(owner, name)

    if study_key in cls.policy_cache:
      existing = cls.policy_cache[study_key]
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

    cls.policy_cache[study_key] = policy

  @classmethod
  def _get_study_resource_name(cls, name: str) -> str:
    """Use for testing only."""
    return cls.tuner.load_study(
        owner=cls.default_owner,
        name=ExpandedStudyName(name),
    ).resource_name

  @classmethod
  def use_study_prefix(cls, study_prefix: Optional[str]):
    cls.default_study_prefix = study_prefix or ''

  @classmethod
  def poll_result(
      cls, name: str, study_owner: Optional[str] = None
  ) -> pg.tuning.Result:
    """Gets tuning result by a unique tuning identifier."""
    name = cls._expand_name(name)
    return core.Result.from_study(
        cls.tuner.load_study(study_owner or cls.default_owner, name)
    )

  # ------------------------------------
  # Instance-level methods and variables begin here.
  # ------------------------------------
  _name: str = attrs.field()
  _study: client_abc.StudyInterface = attrs.field()
  _converter: converters.VizierConverter = attrs.field()
  _algorithm: pg.geno.DNAGenerator = attrs.field()
  _suggestion_generator: Any = attrs.field()

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
    self._algorithm = algorithm
    study_owner = study_owner or self.default_owner
    prior_study_ids = prior_study_ids or tuple()
    name = self._expand_name(name)

    if is_chief:
      mode = TunerMode.PRIMARY
    elif is_chief is None:
      mode = TunerMode.AUTO
    else:
      mode = TunerMode.SECONDARY

    self._converter = converters.VizierConverter.from_dna_spec(
        dna_spec, metrics_to_optimize
    )

    # Load or create study.
    try:
      self._study = self.tuner.load_study(
          study_owner,
          name,
      )
      # Study exists.
      if mode == TunerMode.AUTO:
        if self.tuner.ping_tuner(self._get_chief_tuner_id()):
          mode = TunerMode.SECONDARY
        else:
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
          mode = TunerMode.PRIMARY
          self._register_self_as_primary()
    except KeyError:
      # Study does not exist.
      if mode == TunerMode.SECONDARY:
        self._study = self._wait_for_study(study_owner, name)
      else:
        mode = TunerMode.PRIMARY  # We will make this a chief
        problem = self._converter.problem_or_dummy
        local_tuner_id = self.tuner.get_tuner_id(self._algorithm)
        problem.metadata.ns(constants.METADATA_NAMESPACE)[
            constants.STUDY_METADATA_KEY_TUNER_ID
        ] = local_tuner_id
        self._study = self.tuner.create_study(
            problem, self._converter, study_owner, name, self._algorithm
        )
        if local_tuner_id == self._get_chief_tuner_id():
          # For multi-thread scenario, `local_tuner_id` will be the same for
          # all the worker threads, therefore there is a chance that multiple
          # worker threads consider themselves as PRIMARY. This does not matter
          # since there is only one Pythia service shared across them.
          mode = TunerMode.PRIMARY  # We will make this a chief
          if add_prior_trials:
            # Trials are added to the study directly upon creation.
            trials: Sequence[vz.Trial] = self._load_prior_trials(
                prior_study_ids
            )
            for trial in trials:
              self._study._add_trial(trial)
        else:
          mode = TunerMode.SECONDARY

    # Set up the generator.
    def _suggestion_generator():
      while (
          num_examples is None or len(list(self._study.trials())) < num_examples
      ):
        trials = self._study.suggest(
            count=1, client_id=self.tuner.get_group_id(group)
        )
        if not trials:
          return
        for trial in trials:
          yield core.Feedback(trial, self._converter)

    self._suggestion_generator = _suggestion_generator()

    if mode != TunerMode.PRIMARY or (
        isinstance(self._algorithm, algorithms.PseudoAlgorithm)
    ):
      # nothing more to do
      return

    # Start pythia service.
    self.tuner.start_pythia_service(VizierBackend.policy_cache)
    # Set up the policy.
    self._algorithm.setup(dna_spec)
    prior_trials = tuple()
    if not add_prior_trials:
      # Trials are added to the algorithm only.
      prior_trials: Sequence[vz.Trial] = self._load_prior_trials(
          prior_study_ids
      )
    policy = self._create_policy(early_stopping_policy, prior_trials)
    # Connect the service with policy.
    self._register(study_owner, name, policy)

  def _create_policy(
      self,
      early_stopping_policy: Optional[pg.tuning.EarlyStoppingPolicy],
      prior_trials: Sequence[vz.Trial],
  ) -> TunerPolicy:
    """Creates a pythia policy.

    Args:
      early_stopping_policy:
      prior_trials:

    Returns:
      Policy.
    """
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
        self.tuner.pythia_supporter(self._study),
        self._converter,
        self._algorithm,
        early_stopping_policy=early_stopping_policy,
    )

  def _get_chief_tuner_id(self) -> str:
    metadata = self._study.materialize_problem_statement().metadata.ns(
        constants.METADATA_NAMESPACE
    )
    try:
      return str(metadata[constants.STUDY_METADATA_KEY_TUNER_ID])
    except KeyError as e:
      raise RuntimeError(
          f'Metadata does not exist in study: {self._study.resource_name}'
      ) from e

  def _register_self_as_primary(self) -> str:
    metadata = vz.Metadata()
    tuner_id = self.tuner.get_tuner_id(self._algorithm)
    metadata.ns(constants.METADATA_NAMESPACE)[
        constants.STUDY_METADATA_KEY_TUNER_ID
    ] = self.tuner.get_tuner_id(self._algorithm)
    self._study.update_metadata(metadata)
    self.tuner.use_pythia_for_study(self._study)
    return tuner_id

  def next(self) -> pg.tuning.Feedback:
    """Gets the next tuning feedback object."""
    trial = next(self._suggestion_generator)  # pytype: disable=wrong-arg-types
    return core.Feedback(self._study.get_trial(trial.id), self._converter)
