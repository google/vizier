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

"""Tests for meta learning designer."""
from typing import Optional

import attrs
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.designers.meta_learning import meta_learning

from absl.testing import absltest
from absl.testing import parameterized


MetaLearningDesigner = meta_learning.MetaLearningDesigner
MetaLearningConfig = meta_learning.MetaLearningConfig
MetaLearningState = meta_learning.MetaLearningState


def _eagle_designer_factory(
    problem: vz.ProblemStatement, seed: Optional[int], **kwargs
):
  """Creates an EagleStrategyDesigner with hyper-parameters and seed."""
  config = eagle_strategy.FireflyAlgorithmConfig()
  # Unpack the hyperparameters into the Eagle config class.
  for param_name, param_value in kwargs.items():
    if param_name not in attrs.asdict(config):
      raise ValueError(f"'{param_name}' is not in FireflyAlgorithmConfig!")
    setattr(config, param_name, param_value)
  return eagle_strategy.EagleStrategyDesigner(
      problem_statement=problem,
      seed=seed,
      config=config,
  )


def _quasirandom_designer_factory(
    problem: vz.ProblemStatement, seed: Optional[int] = None
):
  """Creates a QuasiRandomDesigner with seed."""
  return quasi_random.QuasiRandomDesigner(problem.search_space, seed=seed)


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
      # Complete the trial with additional metric to test 'tuned_metric'.
      trial.complete(
          vz.Measurement(
              metrics={
                  'objective': np.random.random(),
                  'secondary_metric': np.random.random(),
              }
          )
      )
      trials.append(trial)
    meta_learning_designer.update(
        vza.CompletedTrials(trials), vza.ActiveTrials()
    )


class MetaLearningDesignerTest(parameterized.TestCase):
  """Tests for AutoTunerDesinger."""

  def setUp(self):
    super().setUp()
    self.problem = vz.ProblemStatement()
    self.problem.search_space.root.add_float_param('x', 0.0, 15.0)
    self.problem.search_space.root.add_float_param('y', -5.0, 10.0)
    self.problem.search_space.root.add_categorical_param('c', ['a', 'b', 'c'])
    self.problem.metric_information.append(
        vz.MetricInformation(
            name='objective',
            goal=vz.ObjectiveMetricGoal.MAXIMIZE,
        )
    )
    self.meta_config = MetaLearningConfig(
        num_trials_per_tuning=10,
        tuning_min_num_trials=100,
        tuning_max_num_trials=500,
    )
    self.tuning_params = vz.SearchSpace()
    self.tuning_params.root.add_float_param(
        'visibility', 0.0, 10.0, default_value=2.22
    )
    self.tuning_params.root.add_float_param(
        'gravity', 0.0, 10.0, default_value=3.33
    )

  def test_initialize_designer(self):
    meta_learning_designer = MetaLearningDesigner(
        problem=self.problem,
        tuning_hyperparams=self.tuning_params,
        tuned_designer_factory=_eagle_designer_factory,
        meta_designer_factory=_quasirandom_designer_factory,
        config=self.meta_config,
    )
    self.assertEqual(
        meta_learning_designer._state, MetaLearningState.INITIALIZE
    )
    # type: ignore[attribute-error]  # pylint: disable=protected-access
    self.assertEqual(
        meta_learning_designer._curr_tuned_designer._config.visibility, 2.22
    )
    self.assertEqual(
        meta_learning_designer._curr_tuned_designer._config.gravity, 3.33
    )

  @parameterized.parameters([1, 5])
  def test_state_transitions(self, batch_size):
    meta_learning_designer = MetaLearningDesigner(
        problem=self.problem,
        tuning_hyperparams=self.tuning_params,
        tuned_designer_factory=_eagle_designer_factory,
        meta_designer_factory=_quasirandom_designer_factory,
        config=self.meta_config,
    )
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
