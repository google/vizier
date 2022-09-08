"""Tests for eagle_strategy."""

from typing import List, Optional, Any
from jax import random
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy

from absl.testing import absltest

EagleStrategyDesiger = eagle_strategy.EagleStrategyDesiger
PRNGKey = Any


def _create_dummy_trial(
    parent_fly_id: int,
    x_value: float,
    obj_value: float,
) -> vz.Trial:
  """Create a dummy completed trial."""
  trial = vz.Trial()
  measurement = vz.Measurement(metrics={'obj': vz.Metric(value=obj_value)})
  trial.parameters['x'] = x_value
  trial.complete(measurement, inplace=True)
  trial.metadata.ns('eagle')['parent_fly_id'] = str(parent_fly_id)
  return trial


def _create_dummy_problem_statement() -> vz.ProblemStatement:
  """Create a dummy problem statement."""
  problem = vz.ProblemStatement()
  problem.search_space.root.add_float_param('x', 0.0, 10.0)
  problem.metric_information.append(
      vz.MetricInformation(name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE))
  return problem


def _create_dummy_fly(
    parent_fly_id: int,
    x_value: float,
    obj_value: float,
) -> eagle_strategy._Firefly:
  """"Create a dummy firefly with a dummy completed trial."""
  trial = _create_dummy_trial(parent_fly_id, x_value, obj_value)
  return eagle_strategy._Firefly(
      id_=parent_fly_id, perturbation_factor=1.0, generation=1, trial=trial)


def _create_dummy_empty_firefly_pool(
    capacity: int = 10) -> eagle_strategy._FireflyPool:
  """Create a dummy empty Firefly pool."""
  problem = _create_dummy_problem_statement()
  config = eagle_strategy.FireflyAlgorithmConfig()
  return eagle_strategy._FireflyPool(problem, config, capacity)


def _create_dummy_populated_firefly_pool(
    *,
    capacity: int,
    x_values: Optional[List[float]] = None,
    obj_values: Optional[List[float]] = None,
) -> eagle_strategy._FireflyPool:
  """Create a dummy populated Firefly pool with a given capacity."""
  firefly_pool = _create_dummy_empty_firefly_pool(capacity=capacity)
  key = random.PRNGKey(0)
  if not x_values:
    x_values = [
        float(x) for x in random.uniform(key, shape=(5,), minval=0, maxval=10)
    ]
  if not obj_values:
    obj_values = [
        float(o)
        for o in random.uniform(key, shape=(5,), minval=-1.5, maxval=1.5)
    ]
  for parent_fly_id, (obj_val, x_val) in enumerate(zip(obj_values, x_values)):
    firefly_pool._pool[parent_fly_id] = _create_dummy_fly(
        parent_fly_id=parent_fly_id, x_value=x_val, obj_value=obj_val)
  firefly_pool._max_pool_id = capacity
  return firefly_pool


def _create_dummy_empty_eagle_designer(*,
                                       key: Optional[PRNGKey] = None
                                      ) -> EagleStrategyDesiger:
  """"Create a dummy empty eagle designer."""
  problem = _create_dummy_problem_statement()
  key = key or random.PRNGKey(0)
  return EagleStrategyDesiger(problem_statement=problem, key=key)


def _create_dummy_populated_eagle_designer(
    *,
    x_values: Optional[List[float]] = None,
    obj_values: Optional[List[float]] = None,
    key: Optional[PRNGKey] = None) -> EagleStrategyDesiger:
  """Create a dummy populated eagle designer."""
  problem = _create_dummy_problem_statement()
  key = key or random.PRNGKey(0)
  eagle_designer = EagleStrategyDesiger(problem_statement=problem, key=key)
  pool_capacity = eagle_designer._firefly_pool.capacity
  # Override the eagle designer's firefly pool with a populated firefly pool.
  eagle_designer._firefly_pool = _create_dummy_populated_firefly_pool(
      x_values=x_values, obj_values=obj_values, capacity=pool_capacity)
  return eagle_designer


class FireflyPoolTest(absltest.TestCase):

  def test_create_or_update_fly(self):
    # Test creating a new fly in the pool.
    firefly_pool = _create_dummy_empty_firefly_pool()
    trial = _create_dummy_trial(parent_fly_id=112, x_value=0, obj_value=0.8)
    firefly_pool.create_or_update_fly(trial)
    self.assertEqual(firefly_pool.size, 1)
    self.assertLen(firefly_pool._pool, 1)
    self.assertIs(firefly_pool._pool[112].trial, trial)
    # Test that another trial with the same parent id updates the fly.
    trial2 = _create_dummy_trial(parent_fly_id=112, x_value=1, obj_value=1.5)
    firefly_pool.create_or_update_fly(trial2)
    self.assertEqual(firefly_pool.size, 1)
    self.assertLen(firefly_pool._pool, 1)
    self.assertIs(firefly_pool._pool[112].trial, trial2)

  def test_find_closest_parent(self):
    firefly_pool = _create_dummy_populated_firefly_pool(
        x_values=[1, 2, 5], obj_values=[2, 10, -2], capacity=4)
    trial = _create_dummy_trial(parent_fly_id=123, x_value=4.2, obj_value=8)
    parent_fly = firefly_pool.find_closest_parent(trial)
    self.assertEqual(parent_fly.id_, 2)

  def test_is_best_fly(self):
    firefly_pool = _create_dummy_populated_firefly_pool(
        x_values=[1, 2, 5], obj_values=[2, 10, -2], capacity=4)
    self.assertTrue(firefly_pool.is_best_fly(firefly_pool._pool[1]))
    self.assertFalse(firefly_pool.is_best_fly(firefly_pool._pool[0]))
    self.assertFalse(firefly_pool.is_best_fly(firefly_pool._pool[2]))

  def test_get_next_moving_fly_copy(self):
    firefly_pool = _create_dummy_populated_firefly_pool(
        x_values=[1, 2, 5], obj_values=[2, 10, -2], capacity=5)
    firefly_pool._last_id = 1
    moving_fly1 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly1.id_, 2)
    moving_fly2 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly2.id_, 0)
    moving_fly3 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly3.id_, 1)

  def test_get_next_moving_fly_copy_after_removing_last_id_fly(self):
    firefly_pool = _create_dummy_populated_firefly_pool(
        x_values=[1, 2, 5], obj_values=[2, 10, -2], capacity=5)
    firefly_pool._last_id = 1
    # Remove the fly associated with `_last_id` from the pool.
    del firefly_pool._pool[1]
    moving_fly1 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly1.id_, 2)
    moving_fly2 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly2.id_, 0)
    moving_fly3 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly3.id_, 2)

  def test_get_next_moving_fly_copy_after_removing_multiple_flies(self):
    firefly_pool = _create_dummy_populated_firefly_pool(
        x_values=[1, 2, 5, -1], obj_values=[2, 10, -2, 8], capacity=5)
    firefly_pool._last_id = 3
    # Remove the several flies
    del firefly_pool._pool[0]
    del firefly_pool._pool[2]
    moving_fly1 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly1.id_, 1)
    moving_fly2 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly2.id_, 3)
    moving_fly3 = firefly_pool.get_next_moving_fly_copy()
    self.assertEqual(moving_fly3.id_, 1)


class EagleStrategyTest(absltest.TestCase):

  def test_dump_load_state(self):
    eagle_designer = _create_dummy_populated_eagle_designer()
    eagle_state = eagle_designer.dump()
    # Create a new eagle designer and load state
    eagle_designer_recovered = _create_dummy_empty_eagle_designer()
    eagle_designer_recovered.load(eagle_state)
    # Generate suggestions from the two designers
    trial_suggestions = eagle_designer.suggest(count=1)
    trial_suggestions_recovered = eagle_designer_recovered.suggest(count=1)
    # Test if the suggestion from the two designers equal
    self.assertEqual(trial_suggestions[0].parameters,
                     trial_suggestions_recovered[0].parameters)

  def test_suggest_one(self):
    eagle_designer = _create_dummy_populated_eagle_designer()
    trial_suggestion = eagle_designer._suggest_one()
    self.assertIsInstance(trial_suggestion, vz.TrialSuggestion)
    self.assertIsNotNone(
        trial_suggestion.metadata.ns('eagle').get('parent_fly_id'))

  def test_suggest(self):
    eagle_designer = _create_dummy_populated_eagle_designer()
    trial_suggestions = eagle_designer.suggest(count=10)
    self.assertLen(trial_suggestions, 10)
    self.assertIsInstance(trial_suggestions[0], vz.TrialSuggestion)

  def test_update_capacitated_pool_no_parent_fly_trial_is_better(self):
    # Capacitated pool size has 11 fireflies.
    eagle_designer = _create_dummy_populated_eagle_designer(
        x_values=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
        obj_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    trial = _create_dummy_trial(parent_fly_id=98, x_value=1.42, obj_value=100.0)
    eagle_designer._update_one(trial)
    self.assertIs(eagle_designer._firefly_pool._pool[3].trial, trial)

  def test_update_capacitated_pool_no_parent_fly_trial_is_not_better(self):
    eagle_designer = _create_dummy_populated_eagle_designer(
        x_values=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
        obj_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    trial = _create_dummy_trial(parent_fly_id=98, x_value=1.42, obj_value=-80.0)
    prev_trial = eagle_designer._firefly_pool._pool[3].trial
    eagle_designer._update_one(trial)
    self.assertIs(eagle_designer._firefly_pool._pool[3].trial, prev_trial)

  def test_update_capacitated_pool_with_parent_fly_trial_is_better(self):
    eagle_designer = _create_dummy_populated_eagle_designer(
        x_values=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
        obj_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    trial = _create_dummy_trial(parent_fly_id=2, x_value=3.3, obj_value=80.0)
    eagle_designer._update_one(trial)
    self.assertIs(eagle_designer._firefly_pool._pool[2].trial, trial)

  def test_update_capacitated_pool_with_parent_fly_trial_is_not_better(self):
    eagle_designer = _create_dummy_populated_eagle_designer(
        x_values=[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1],
        obj_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    trial = _create_dummy_trial(parent_fly_id=2, x_value=3.3, obj_value=-80.0)
    prev_trial = eagle_designer._firefly_pool._pool[2].trial
    eagle_designer._update_one(trial)
    self.assertIs(eagle_designer._firefly_pool._pool[2].trial, prev_trial)

  def test_update_empty_pool(self):
    eagle_designer = _create_dummy_empty_eagle_designer()
    trial = _create_dummy_trial(parent_fly_id=0, x_value=3.3, obj_value=0.0)
    eagle_designer._update_one(trial)
    self.assertIs(eagle_designer._firefly_pool._pool[0].trial, trial)


if __name__ == '__main__':
  absltest.main()
