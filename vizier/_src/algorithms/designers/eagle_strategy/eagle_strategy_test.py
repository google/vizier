# Copyright 2022 Google LLC.
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

"""Tests for eagle_strategy."""

from vizier import algorithms as vza
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.designers.eagle_strategy import testing
from vizier._src.algorithms.testing import comparator_runner
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest
from absl.testing import parameterized

EagleStrategyDesiger = eagle_strategy.EagleStrategyDesiger


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
        trial_suggestion.metadata.ns('eagle').get('parent_fly_id'))

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

  def test_linear_scale(self):
    problem = vz.ProblemStatement(metric_information=[
        vz.MetricInformation(name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    ])
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 5.0)
    problem.search_space.root.add_categorical_param('c1', ['a', 'b', 'c'])
    problem.search_space.root.add_int_param('i1', 0, 10)
    problem.search_space.root.add_discrete_param('d1', [1, 5, 10])
    EagleStrategyDesiger(problem)
    problem.search_space.root.add_float_param(
        'f3', 0.0, 10.0, scale_type=vz.ScaleType.LOG)
    with self.assertRaises(ValueError):
      EagleStrategyDesiger(problem)

  @parameterized.parameters(
      testing.create_continuous_exptr(bbob.Sphere),
      testing.create_categorical_exptr())
  def test_convergence_confidence_check(self, exptr):
    """A simple and small convergence test for basic confidence check."""

    def _random_designer_factory(problem: vz.ProblemStatement) -> vza.Designer:
      return random.RandomDesigner(problem.search_space, seed=1)

    def _eagle_designer_factory(problem: vz.ProblemStatement) -> vza.Designer:
      return eagle_strategy.EagleStrategyDesiger(problem, seed=1)

    random_benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=_random_designer_factory,
        experimenter=exptr,
    )
    eagle_benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=_eagle_designer_factory,
        experimenter=exptr,
    )
    evaluations = 1000
    # Random designer batch size is large to expedite run time.
    comparator_runner.SimpleRegretComparisonTester(
        baseline_num_trials=evaluations,
        candidate_num_trials=evaluations,
        baseline_suggestion_batch_size=evaluations,
        candidate_suggestion_batch_size=5,
        baseline_num_repeats=5,
        candidate_num_repeats=1,
        alpha=0.05,
        goal=vz.ObjectiveMetricGoal.MINIMIZE
    ).assert_benchmark_state_better_simple_regret(
        baseline_benchmark_state_factory=random_benchmark_state_factory,
        candidate_benchmark_state_factory=eagle_benchmark_state_factory)


if __name__ == '__main__':
  absltest.main()
