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

"""Tests for meta learning designer."""
from typing import Sequence

import attr
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import grid
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.designers.meta_learning import meta_learning

from absl.testing import absltest
from absl.testing import parameterized


MetaLearningDesigner = meta_learning.MetaLearningDesigner
MetaLearningConfig = meta_learning.MetaLearningConfig
MetaLearningState = meta_learning.MetaLearningState


@attr.define
class FakeDesigner(vza.Designer):
  """Fake designer."""

  problem: vz.ProblemStatement
  designer_param: float = attr.field(
      default=0.5, validator=attr.validators.instance_of(float)
  )

  def suggest(self, count: int = 1) -> Sequence[vz.TrialSuggestion]:
    search_space = self.problem.search_space
    if np.random.random() < self.designer_param:
      return random.RandomDesigner(search_space).suggest(count)
    else:
      return grid.GridSearchDesigner(search_space).suggest(count)

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    pass


@attr.define
class FakeMetaDesigner(vza.Designer):
  """Fake meta designer."""

  problem: vz.ProblemStatement
  meta_param1: float = 0.7
  meta_param2: float = 1.0

  def suggest(self, count: int = 1) -> Sequence[vz.TrialSuggestion]:
    search_space = self.problem.search_space
    if np.random.random() * self.meta_param1 < self.meta_param2:
      return random.RandomDesigner(search_space).suggest(count)
    else:
      return quasi_random.QuasiRandomDesigner(search_space).suggest(count)

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    pass


def fake_designer_factory(
    problem: vz.ProblemStatement, **kwargs
) -> vza.Designer:
  if 'designer_param' not in kwargs:
    return FakeDesigner(problem=problem)
  else:
    return FakeDesigner(
        problem=problem, designer_param=kwargs['designer_param']
    )


def fake_meta_designer_factory(
    problem: vz.ProblemStatement, **kwargs
) -> vza.Designer:
  del kwargs
  return FakeMetaDesigner(problem=problem)


def _create_fake_problem() -> vz.ProblemStatement:
  problem = vz.ProblemStatement()
  problem.search_space.root.add_float_param('x', 0.0, 15.0)
  problem.search_space.root.add_float_param('y', -5.0, 10.0)
  problem.search_space.root.add_categorical_param('c', ['a', 'b', 'c'])
  problem.metric_information.append(
      vz.MetricInformation(
          name='objective',
          goal=vz.ObjectiveMetricGoal.MAXIMIZE,
      )
  )
  return problem


def _create_meta_learning_designer() -> MetaLearningDesigner:
  problem = _create_fake_problem()
  meta_config = MetaLearningConfig(
      num_trials_per_tuning=10,
      tuning_min_num_trials=100,
      tuning_max_num_trials=500,
  )
  tuning_params = vz.SearchSpace()
  tuning_params.root.add_float_param(
      'designer_param', 0.0, 1.0, default_value=0.5
  )

  meta_learning_designer = MetaLearningDesigner(
      problem=problem,
      tuning_hyperparams=tuning_params,
      tuned_designer_factory=fake_designer_factory,
      meta_designer_factory=fake_meta_designer_factory,
      config=meta_config,
  )
  return meta_learning_designer


def meta_learning_suggest_update_loop(
    meta_learning_designer: MetaLearningDesigner,
    num_suggestions: int,
    batch_size: int,
):
  trial_id = 0
  for _ in range(num_suggestions // batch_size):
    suggestions = meta_learning_designer.suggest(batch_size)
    trials = []
    for i in range(batch_size):
      trial = suggestions[i].to_trial(trial_id)
      trial_id += 1
      trial.complete(vz.Measurement(metrics={'objective': np.random.random()}))
      trials.append(trial)
    meta_learning_designer.update(
        vza.CompletedTrials(trials), vza.ActiveTrials()
    )


class MetaLearningDesignerTest(parameterized.TestCase):
  """Tests for AutoTunerDesinger."""

  @parameterized.parameters([1, 3, 5, 10])
  def test_state_transitions(self, batch_size):
    meta_learning_designer = _create_meta_learning_designer()
    self.assertEqual(
        meta_learning_designer._state,
        MetaLearningState.INITIALIZE,
    )
    meta_learning_suggest_update_loop(meta_learning_designer, 50, batch_size)
    self.assertEqual(
        meta_learning_designer._state,
        MetaLearningState.INITIALIZE,
    )
    meta_learning_suggest_update_loop(meta_learning_designer, 200, batch_size)
    self.assertEqual(
        meta_learning_designer._state, meta_learning.MetaLearningState.TUNE
    )
    meta_learning_suggest_update_loop(meta_learning_designer, 600, batch_size)
    self.assertEqual(
        meta_learning_designer._state,
        MetaLearningState.USE_BEST_PARAMS,
    )


if __name__ == '__main__':
  absltest.main()
