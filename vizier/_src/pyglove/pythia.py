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

"""TunerPolicy."""

from typing import Optional, Sequence

from absl import logging
import attr
import pyglove as pg
from vizier import pythia
from vizier import pyvizier as vz
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
  _algorithm: pg.geno.DNAGenerator = attr.field()
  _incorporated_trial_ids: set[int] = attr.field(factory=set)
  _early_stopping_policy: Optional[pg.tuning.EarlyStoppingPolicy] = attr.field(
      default=None
  )

  @property
  def algorithm(self) -> pg.geno.DNAGenerator:
    return self._algorithm

  @property
  def early_stopping_policy(self) -> Optional[pg.tuning.EarlyStoppingPolicy]:
    return self._early_stopping_policy

  @property
  def _metric_names(self) -> Sequence[str]:
    return self._converter.metrics_to_optimize

  def update(self, tuner_trial: pg.tuning.Trial) -> bool:
    """Update a single tuner Trial.

    Args:
      tuner_trial: If the trial id was previously seen, update is no-op.

    Returns:
      True if the trial was added.
    """
    if tuner_trial.id in self._incorporated_trial_ids:
      return False
    logging.info(
        'Updating TunerTrial %s to algorithm: %s', tuner_trial, self._algorithm
    )
    reward = tuner_trial.get_reward_for_feedback(self._metric_names)
    if reward is not None:
      self._algorithm.feedback(tuner_trial.dna, reward)
      self._incorporated_trial_ids.add(tuner_trial.id)
      return True
    return False

  def suggest(self, request: pythia.SuggestRequest) -> pythia.SuggestDecision:
    logging.info('Tuner policy get new suggestions')
    completed_trials = self.supporter.GetTrials(
        status_matches=vz.TrialStatus.COMPLETED
    )

    mu = vz.MetadataDelta()
    n_trials_updated = 0
    for vizier_trial in completed_trials:
      tuner_trial = core.VizierTrial(self._converter, vizier_trial)
      metadata = dict(tuner_trial.dna.metadata)  # make a deep copy.
      if self.update(tuner_trial):
        n_trials_updated += 1
        # Serialize DNA metadata to Vizier trial metadata upon change.
        if pg.ne(tuner_trial.dna.metadata, metadata):
          mu.assign(
              constants.METADATA_NAMESPACE,
              constants.TRIAL_METADATA_KEY_DNA_METADATA,
              pg.to_json_str(tuner_trial.dna.metadata),
              trial=vizier_trial,
          )

    new_trials = []
    count = request.count or 1

    logging.info(
        (
            'The algorithm now reflects %s new trials.'
            'The algorithm has seen %s trials total and '
            'there are %s total completed trials in the study '
            'Now generating %s new suggestions.'
        ),
        n_trials_updated,
        len(self._incorporated_trial_ids),
        len(completed_trials),
        count,
    )
    for _ in range(request.count or 1):
      dna = self._algorithm.propose()
      if dna.spec is None:
        dna.use_spec(self._converter.dna_spec)
      trial = self._converter.to_trial(dna, fallback='return_dummy')
      new_trials.append(trial)
      logging.info(
          'algorithm : %s proposed DNA: %s which is translated to trial: %s',
          self._algorithm,
          repr(dna),
          trial,
      )

    return pythia.SuggestDecision(new_trials, mu)

  def early_stop(
      self, request: pythia.EarlyStopRequest
  ) -> pythia.EarlyStopDecisions:
    # TODO: Support early stopping in the future.
    return pythia.EarlyStopDecisions()
