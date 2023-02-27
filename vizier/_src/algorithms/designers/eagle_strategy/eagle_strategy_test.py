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

"""Tests for eagle_strategy."""

import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.designers.eagle_strategy import testing

from absl.testing import absltest
from absl.testing import parameterized


EagleStrategyDesiger = eagle_strategy.EagleStrategyDesigner


class EagleStrategyTest(parameterized.TestCase):

  def test_dump_and_load(self):
    eagle_designer = testing.create_fake_populated_eagle_designer()
    partial_serialized_eagle = eagle_designer.dump()
    # Create a new eagle designer and load state
    eagle_designer_restored = testing.create_fake_empty_eagle_designer()
    eagle_designer_restored.load(partial_serialized_eagle)
    # Generate suggestions from the two designers
    trial_suggestions = eagle_designer.suggest(count=1)
    trial_suggestions_recovered = eagle_designer_restored.suggest(count=1)
    # Test if the suggestion from the two designers equal
    self.assertEqual(trial_suggestions[0].parameters,
                     trial_suggestions_recovered[0].parameters)

  def test_load_with_no_state(self):
    problem = testing.create_fake_problem_statement()
    eagle_designer = EagleStrategyDesiger(problem)
    metadata = vz.Metadata()
    # Check that the designer can accept empty metadata in load.
    eagle_designer.load(metadata)

  def test_suggest_one(self):
    eagle_designer = testing.create_fake_populated_eagle_designer()
    trial_suggestion = eagle_designer._suggest_one()
    self.assertIsInstance(trial_suggestion, vz.TrialSuggestion)
    self.assertIsNotNone(
        trial_suggestion.metadata.ns('eagle').get('parent_fly_id')
    )

  def test_embedding(self):
    eagle_designer = testing.create_fake_populated_eagle_designer()
    # Check that the problem was converted.
    self.assertEqual(
        eagle_designer._problem.search_space.parameters[0].bounds, (0.0, 1.0)
    )
    # Check that internal suggestions are in normalized range.
    for _ in range(10):
      trial_suggestion = eagle_designer._suggest_one()
      self.assertBetween(trial_suggestion.parameters['x'].value, 0.0, 1.0)
    # Check that update maps the trials correctly to the normalized space.
    eagle_designer = testing.create_fake_empty_eagle_designer()
    trial = vz.Trial({'x': 10.0})
    trial = trial.complete(
        vz.Measurement(metrics={'objective': np.random.uniform()})
    )
    complete_trials = vza.CompletedTrials([trial])
    eagle_designer.update(complete_trials, vza.ActiveTrials())
    print(eagle_designer._firefly_pool)
    self.assertEqual(
        eagle_designer._firefly_pool._pool[0].trial.parameters['x'].value, 1.0
    )

  @parameterized.parameters(1e-4, 1.0)
  def test_penalize_parent_fly_no_trial_change(self, perturbation):
    eagle_designer = testing.create_fake_populated_eagle_designer(
        x_values=[1.0, 2.0, 3.0], obj_values=[1, 2, 3], parent_fly_ids=[1, 2, 3]
    )
    trial = testing.create_fake_trial(parent_fly_id=2, x_value=2.0, obj_value=2)
    parent_fly = eagle_designer._firefly_pool._pool[2]
    # Set the perturbation.
    parent_fly.perturbation = perturbation
    before_perturbation = parent_fly.perturbation
    eagle_designer._penalize_parent_fly(parent_fly, trial)
    after_perturbation = parent_fly.perturbation
    if perturbation == 1.0:
      # Perturbation is already high so capped by the maximimum.
      self.assertEqual(
          after_perturbation,
          before_perturbation * eagle_designer._config.max_perturbation,
      )
    elif perturbation == 1e-4:
      # Not reaching the maximum yet, multiply by 10.
      self.assertEqual(after_perturbation, 1e-3)

  def test_penalize_parent_fly(self):
    # Capacitated pool size has 11 fireflies.
    eagle_designer = testing.create_fake_populated_eagle_designer(
        x_values=[1.0, 2.0, 3.0], obj_values=[1, 2, 3], parent_fly_ids=[1, 2, 3]
    )
    trial = testing.create_fake_trial(
        parent_fly_id=2, x_value=1.42, obj_value=0.5
    )
    parent_fly = eagle_designer._firefly_pool._pool[2]
    before_perturbation = parent_fly.perturbation
    eagle_designer._penalize_parent_fly(parent_fly, trial)
    after_perturbation = parent_fly.perturbation
    self.assertEqual(after_perturbation, before_perturbation * 0.9)

  def test_suggest(self):
    eagle_designer = testing.create_fake_populated_eagle_designer()
    trial_suggestions = eagle_designer.suggest(count=10)
    self.assertLen(trial_suggestions, 10)
    self.assertIsInstance(trial_suggestions[0], vz.TrialSuggestion)

  def test_update_capacitated_pool_no_parent_fly_trial_is_better(self):
    # Capacitated pool size has 11 fireflies.
    eagle_designer = testing.create_fake_populated_eagle_designer(
        x_values=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
        obj_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    trial = testing.create_fake_trial(
        parent_fly_id=98, x_value=1.42, obj_value=100.0)
    eagle_designer._update_one(trial)
    self.assertIs(eagle_designer._firefly_pool._pool[3].trial, trial)

  def test_update_capacitated_pool_no_parent_fly_trial_is_not_better(self):
    eagle_designer = testing.create_fake_populated_eagle_designer(
        x_values=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
        obj_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    trial = testing.create_fake_trial(
        parent_fly_id=98, x_value=1.42, obj_value=-80.0)
    prev_trial = eagle_designer._firefly_pool._pool[3].trial
    eagle_designer._update_one(trial)
    self.assertIs(eagle_designer._firefly_pool._pool[3].trial, prev_trial)

  def test_update_capacitated_pool_with_parent_fly_trial_is_better(self):
    eagle_designer = testing.create_fake_populated_eagle_designer(
        x_values=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
        obj_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    trial = testing.create_fake_trial(
        parent_fly_id=2, x_value=3.3, obj_value=80.0)
    eagle_designer._update_one(trial)
    self.assertIs(eagle_designer._firefly_pool._pool[2].trial, trial)

  def test_update_capacitated_pool_with_parent_fly_trial_is_not_better(self):
    eagle_designer = testing.create_fake_populated_eagle_designer(
        x_values=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
        obj_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    trial = testing.create_fake_trial(
        parent_fly_id=2, x_value=3.3, obj_value=-80.0)
    prev_trial = eagle_designer._firefly_pool._pool[2].trial
    eagle_designer._update_one(trial)
    self.assertIs(eagle_designer._firefly_pool._pool[2].trial, prev_trial)

  def test_update_empty_pool(self):
    eagle_designer = testing.create_fake_empty_eagle_designer()
    trial = testing.create_fake_trial(
        parent_fly_id=0, x_value=3.3, obj_value=0.0)
    eagle_designer._update_one(trial)
    self.assertIs(eagle_designer._firefly_pool._pool[0].trial, trial)

  @parameterized.parameters(1, 3, 5)
  def test_suggest_update(self, batch_size):
    problem = vz.ProblemStatement()
    problem.search_space.select_root().add_float_param('float', -5.0, 5.0)
    problem.search_space.select_root().add_int_param(
        'int', min_value=0, max_value=10
    )
    problem.search_space.select_root().add_discrete_param(
        'discrete', feasible_values=[0.0, 0.6]
    )
    problem.search_space.select_root().add_categorical_param(
        'categorical', feasible_values=['a', 'b', 'c']
    )
    problem.metric_information.append(
        vz.MetricInformation(goal=vz.ObjectiveMetricGoal.MINIMIZE, name='')
    )
    eagle_designer = EagleStrategyDesiger(problem)

    tid = 1
    # Simulate running the designer for 3 suggestions each with a batch.
    for _ in range(3):
      suggestions = eagle_designer.suggest(batch_size)
      completed = []
      # Completing the suggestions while assigning unique trial id.
      for suggestion in suggestions:
        completed.append(
            suggestion.to_trial(tid).complete(
                vz.Measurement(metrics={'': np.random.uniform()})
            )
        )
        tid += 1
      eagle_designer.update(vza.CompletedTrials(completed), vza.ActiveTrials())


if __name__ == '__main__':
  absltest.main()
