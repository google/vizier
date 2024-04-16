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

"""A meta-Designer to adaptively ensemble multiple Designers."""

from typing import Callable, Optional, Sequence

from absl import logging
import attrs
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.ensemble import ensemble_design
from vizier.benchmarks import analyzers

ENSEMBLE_NS = 'ens'
EXPERT_KEY = 'expert'


@attrs.define
class ObjectiveRewardGenerator:
  """Stateful rewards generator using the objective curves of Trials."""

  problem: vz.ProblemStatement = attrs.field()
  all_trials: list[vz.Trial] = attrs.field(factory=list)
  reward_regularization: float = attrs.field(
      default=0.0,
      validator=[attrs.validators.instance_of(float), attrs.validators.ge(0)],
  )
  min_reward: float = attrs.field(
      default=0.0,
      validator=[attrs.validators.instance_of(float), attrs.validators.ge(0)],
  )
  # Arguments passed to the hypervolume converter.
  reference_value: Optional[np.ndarray] = attrs.field(
      default=None,
      kw_only=True,
      validator=[
          attrs.validators.optional(attrs.validators.instance_of(np.ndarray))
      ],
  )
  num_vectors: int = attrs.field(
      default=100, kw_only=True, validator=[attrs.validators.ge(0)]
  )

  def __call__(self, trials: list[vz.Trial]) -> list[float]:
    """Generate rewards from trials."""
    # Using MultiMetricConverter for safety understanding.
    if self.problem.is_single_objective:

      def curve_generator() -> analyzers.StatefulCurveConverter:
        return analyzers.MultiMetricCurveConverter.from_metrics_config(
            self.problem.metric_information, flip_signs_for_min=True
        )

    else:

      def curve_generator() -> analyzers.StatefulCurveConverter:
        return analyzers.MultiMetricCurveConverter.from_metrics_config(
            self.problem.metric_information,
            reference_value=self.reference_value,
            num_vectors=self.num_vectors,
            infer_origin_factor=0.1,
        )

    stateful_curve_generator = analyzers.RestartingCurveConverter(
        curve_generator, restart_min_trials=10, restart_rate=1.5
    )

    original_num_trials = len(self.all_trials) - len(trials)
    if original_num_trials <= 0:
      # First round so there is no reward signal.
      rewards = [self.min_reward] * len(trials)
    else:
      # Objective curve is a 1 x (1+len(trials)) array.
      objective_curve = stateful_curve_generator.convert(
          self.all_trials[-1:] + trials
      )
      rewards = []
      for idx in range(len(trials)):
        obj_reward = objective_curve.ys[:, idx + 1] - objective_curve.ys[:, idx]
        regularized_reward = (
            obj_reward
            + self.reward_regularization * objective_curve.ys[:, idx + 1]
        )
        if np.isfinite(regularized_reward):
          rewards.append(
              max(self.min_reward, float(np.squeeze(regularized_reward)))
          )
        else:
          rewards.append(self.min_reward)

    self.all_trials.extend(trials)
    return rewards


class EnsembleDesigner(vza.Designer):
  """Ensembling of Designers."""

  def __init__(
      self,
      designers: dict[str, vza.Designer],
      ensemble_design_factory: Callable[
          [list[int]], ensemble_design.EnsembleDesign
      ] = ensemble_design.RandomEnsembleDesign,
      reward_generator: Optional[ObjectiveRewardGenerator] = None,
      *,
      use_diverse_suggest: bool = False,
      use_separate_update: bool = False,
  ):
    """Creates a wrapper that uses an ensemble of pydesigners.

    Args:
      designers: Dictionary of PyDesigners to be ensembled.
      ensemble_design_factory: EnsembleDesign factory that takes in list of
        indices.
      reward_generator: Generates rewards for use in ensembling strategy.
      use_diverse_suggest: Whether to use a diverse set of designers for batched
        Suggests by repeatedly calling Suggest(1) in round robin fashion.
      use_separate_update: If false, update all Designers with Trials that were
        Suggested by other Designers.

    Raises:
      ValueError when designers is empty.
    """
    if not designers:
      raise ValueError('Empty list of Designers for ensembling.')
    self._designers = designers
    self._ensemble_design_factory = ensemble_design_factory
    # Each index corresponds to the designer order in the dictionary.
    self._strategy = ensemble_design_factory(list(range(len(self._designers))))
    self._use_diverse_suggest = use_diverse_suggest
    self._reward_generator = reward_generator
    self._use_separate_update = use_separate_update

  def suggest(self, num_suggestions: int) -> Sequence[vz.TrialSuggestion]:
    """Randomly chooses a designer and Suggests from the chosen designer.

    Args:
      num_suggestions: Number of suggested trials desired.

    Returns:
      Batch of suggested trials.
    """

    if num_suggestions == 1 or not self._use_diverse_suggest:
      probs = self._strategy.ensemble_probs
      logging.info(
          'Choosing ensemble with probabilities %s over %s',
          probs,
          self._designers.keys(),
      )
      designer_key = np.random.choice(list(self._designers.keys()), p=probs)
      trials = self._designers[designer_key].suggest(num_suggestions)
      for t in trials:
        t.metadata.ns(ENSEMBLE_NS)[EXPERT_KEY] = f'{designer_key}'
      return trials
    else:
      # Apply diverse Suggestions.
      diverse_suggestions = []
      for _ in range(num_suggestions):
        diverse_suggestions.extend(self.suggest(1))
      return diverse_suggestions

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    # With no completed Trials, simply update with reward calcuations.
    if not completed.trials:
      for designer in self._designers.values():
        designer.update(completed, all_active)
      return

    if self._reward_generator is None:
      rewards = []
      for t in completed.trials:
        if t.infeasible:
          rewards.append(0.0)
        elif t.final_measurement is not None and t.final_measurement.metrics:
          metrics = t.final_measurement.metrics
          first_key = list(metrics.keys())[0]
          rewards.append(metrics.get_value(first_key, default=0.0))
        else:
          raise ValueError(
              'A completed Trial is feasible but has no final measurement'
              f' metrics: {t}.'
          )
    else:
      rewards = self._reward_generator(list(completed.trials))

    if len(completed.trials) != len(rewards):
      # If lengths are different zip(..) would silently truncate the lists.
      raise RuntimeError(
          'Internal error: Mismatched completed trials count'
          f' (len(completed.trials)) and rewards counts {(len(rewards))}.'
      )
    # Update the EnsembleDesign strategy.
    for trial, reward in zip(completed.trials, rewards):
      designer_key = trial.metadata.ns(ENSEMBLE_NS).get(EXPERT_KEY)
      if designer_key is None:
        logging.warning(
            'Trial does not contain required metadata with key %s: %s',
            EXPERT_KEY,
            trial,
        )
        continue

      all_keys = list(self._designers.keys())
      if designer_key not in all_keys:
        logging.warning(
            'Trial does not contain designer-specific metadata for designer key'
            ' %s in all designer keys (skipping update as default): \n %s',
            designer_key,
            all_keys,
        )
        if self._use_separate_update:
          raise ValueError(
              'Separate update algorithms require designer-specific metadata'
              f'{designer_key} in all designer keys: \n {all_keys}'
          )
        continue

      observation = (all_keys.index(designer_key), reward)
      logging.info('Updating with expert, rewards: %s', observation)
      self._strategy.update(observation)

    # Update the underlying designers.
    if self._use_separate_update:
      for designer_key, designer in self._designers.items():
        filtered_completed = [
            t
            for t in completed.trials
            if t.metadata.ns(ENSEMBLE_NS).get(EXPERT_KEY) == designer_key
        ]
        filtered_active = [
            t
            for t in all_active.trials
            if t.metadata.ns(ENSEMBLE_NS).get(EXPERT_KEY) == designer_key
        ]
        designer.update(
            vza.CompletedTrials(filtered_completed),
            vza.ActiveTrials(filtered_active),
        )
    else:
      for designer in self._designers.values():
        designer.update(completed, all_active)
