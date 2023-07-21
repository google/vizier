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
  """Generates rewards using the objective curves of Trials."""

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

  def __call__(self, trials: Sequence[vz.Trial]) -> list[float]:
    """Generate rewards from trials."""
    self.all_trials.extend(trials)
    objectives = self.problem.metric_information.of_type(
        vz.MetricType.OBJECTIVE
    )
    if self.problem.is_single_objective:
      metric_information = objectives.item()
      curve_generator = analyzers.ConvergenceCurveConverter(
          metric_information, flip_signs_for_min=True
      )
    else:
      curve_generator = analyzers.HypervolumeCurveConverter(
          list(objectives),
          reference_value=self.reference_value,
          num_vectors=self.num_vectors,
      )

    # Objective curve is a 1 x len(all_trials) array.
    objective_curve = curve_generator.convert(self.all_trials)
    original_num_trials = len(self.all_trials) - len(trials)
    if original_num_trials <= 0:
      # First round so there is no reward signal.
      return [self.min_reward] * len(trials)
    else:
      rewards = []
      for idx in range(original_num_trials, len(self.all_trials)):
        obj_reward = objective_curve.ys[:, idx] - objective_curve.ys[:, idx - 1]
        regularized_reward = (
            obj_reward + self.reward_regularization * objective_curve.ys[:, idx]
        )
        rewards.append(max(self.min_reward, float(regularized_reward)))
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
    self._rewards = []
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
    if self._reward_generator is None:
      rewards = []
      for t in completed.trials:
        if t.final_measurement is not None:
          metrics = t.final_measurement.metrics
          first_key = list(metrics.keys())[0]
          rewards.append(metrics.get_value(first_key, default=0.0))
    else:
      rewards = self._reward_generator(completed.trials)

    # Update the EnsembleDesign strategy.
    for trial, reward in zip(completed.trials, rewards):
      designer_key = trial.metadata.ns(ENSEMBLE_NS).get(EXPERT_KEY)
      if designer_key is None:
        raise RuntimeError(
            'Trial does not contain required metadata with key'
            f' {EXPERT_KEY}: {trial}'
        )

      observation = (list(self._designers.keys()).index(designer_key), reward)
      logging.info('Updating with expert, rewards: %s', observation)
      self._rewards.append(observation)
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
