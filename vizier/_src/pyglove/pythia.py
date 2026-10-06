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

"""TunerPolicy."""

from typing import Optional, Sequence, cast

from absl import logging
import attr
import pyglove as pg
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.algorithms.policies import trial_caches
from vizier._src.pyglove import constants
from vizier._src.pyglove import converters
from vizier._src.pyglove import core


@attr.define
class TunerPolicy(pythia.Policy):
  """Pythia policy for custom multi-trial tuner algorithm.

  Note that study_config should be used if the user needs the Trials to
  faithfully use the parameter names from the original StudyConfig.
  """

  supporter: pythia.PolicySupporter = attr.field()
  _converter: converters.VizierConverter = attr.field()
  algorithm: pg.geno.DNAGenerator = attr.field()
  early_stopping_policy: Optional[pg.tuning.EarlyStoppingPolicy] = attr.field(
      default=None
  )

  def __attrs_post_init__(self):
    # Initialize the caches. Unlike Vizier's Pythia service implementation,
    # Pyglove's pythia service maintains a single Policy instance for each
    # study (unless the binary crashes). We can therefore use an in-ram cache
    # and avoid re-loading the same trials over and over again.
    self._suggestion_cache = trial_caches.IdDeduplicatingTrialLoader(
        self.supporter, include_intermediate_measurements=False
    )
    self._stopping_cache = trial_caches.IdDeduplicatingTrialLoader(
        self.supporter, include_intermediate_measurements=True
    )

  @property
  def _metric_names(self) -> Sequence[str]:
    return self._converter.metrics_to_optimize

  def _update(self, tuner_trial: pg.tuning.Trial) -> bool:
    """Update a single tuner Trial.

    Args:
      tuner_trial: If the trial id was previously seen, update is no-op.

    Returns:
      True if the trial was added.
    """
    logging.info(
        'Updating TunerTrial id=%s to algorithm: %s',
        tuner_trial.id,
        self.algorithm,
    )
    reward = tuner_trial.get_reward_for_feedback(self._metric_names)
    if reward is not None:
      self.algorithm.feedback(tuner_trial.dna, reward)
      return True
    return False

  def suggest(self, request: pythia.SuggestRequest) -> pythia.SuggestDecision:
    logging.info('TunerPolicy:suggest started')
    newly_completed_trials = self._suggestion_cache.get_newly_completed_trials(
        request.max_trial_id
    )
    n_trials_updated = 0
    metadata_updates = vz.MetadataDelta()
    for vizier_trial in newly_completed_trials:
      tuner_trial = core.VizierTrial(self._converter, vizier_trial)
      metadata = dict(tuner_trial.dna.metadata)  # make a deep copy.
      if self._update(tuner_trial):
        n_trials_updated += 1
        # Serialize DNA metadata to Vizier trial metadata upon change.
        if pg.ne(tuner_trial.dna.metadata, metadata):
          metadata_updates.assign(
              namespace=constants.METADATA_NAMESPACE,
              key=constants.TRIAL_METADATA_KEY_DNA_METADATA,
              value=pg.to_json_str(tuner_trial.dna.metadata),
              trial=vizier_trial,
          )

    new_trials = []
    count = request.count or 1

    logging.info(
        (
            'The algorithm is updated with %s new trials out of %s newly'
            ' completed trials. It has seen %s trials total. Now generating %s'
            ' new suggestions.'
        ),
        n_trials_updated,
        len(newly_completed_trials),
        self._suggestion_cache.num_incorporated_trials,
        count,
    )
    for _ in range(request.count or 1):
      try:
        dna = self.algorithm.propose()
      except StopIteration:
        logging.info(
            'Search algorithm %s has no new suggestions.', self.algorithm
        )
        break

      if dna.spec is None:
        dna.use_spec(self._converter.dna_spec)
      trial = self._converter.to_trial(dna, fallback='return_dummy')
      new_trials.append(trial)
      logging.info(
          'algorithm : %s proposed DNA: %s which is translated to trial: %s',
          self.algorithm,
          repr(dna),
          trial,
      )

    logging.info('TunerPolicy:suggest ended')
    return pythia.SuggestDecision(new_trials, metadata_updates)

  def early_stop(
      self, request: pythia.EarlyStopRequest
  ) -> pythia.EarlyStopDecisions:
    if self.early_stopping_policy is None:
      return pythia.EarlyStopDecisions()

    early_stopping_policy = cast(
        pg.tuning.EarlyStoppingPolicy, self.early_stopping_policy
    )

    newly_completed_trials = self._stopping_cache.get_newly_completed_trials(
        request.max_trial_id
    )
    for vizier_trial in newly_completed_trials:
      tuner_trial = core.VizierTrial(self._converter, vizier_trial)
      # For completed trials, we do not need the actual stopping decision but
      # we still have to feed them in and update the stopping policy's state.
      early_stopping_policy.should_stop_early(tuner_trial)

    active_trials = self._stopping_cache.get_active_trials()

    decisions = pythia.EarlyStopDecisions()
    for vizier_trial in active_trials:
      tuner_trial = core.VizierTrial(self._converter, vizier_trial)
      should_stop = early_stopping_policy.should_stop_early(tuner_trial)
      decisions.decisions.append(
          pythia.EarlyStopDecision(
              vizier_trial.id,
              should_stop=should_stop,
              reason='Pyglove stopping policy stopped the trial.',
          )
      )
    return decisions


def create_policy(
    supporter: pythia.PolicySupporter,
    problem_statement: vz.ProblemStatement,
    algorithm: pg.geno.DNAGenerator,
    early_stopping_policy: Optional[pg.tuning.EarlyStoppingPolicy] = None,
    prior_trials: Optional[Sequence[vz.Trial]] = None,
) -> pythia.Policy:
  """Creates a Pythia policy that uses PyGlove algorithms."""
  converter = converters.VizierConverter.from_problem(problem_statement)

  # Bind the algorithm with the search space before usage.
  algorithm.setup(converter.dna_spec)

  # Warm up algorithm if prior trials are present.
  if prior_trials:

    def get_trial_history():
      for trial in prior_trials:
        tuner_trial = core.VizierTrial(converter, trial)
        reward = tuner_trial.get_reward_for_feedback(
            converter.metrics_to_optimize
        )
        yield (tuner_trial.dna, reward)

    algorithm.recover(get_trial_history())

  return TunerPolicy(supporter, converter, algorithm, early_stopping_policy)
