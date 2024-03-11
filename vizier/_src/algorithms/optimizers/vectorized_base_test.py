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

"""Tests for vectorized_base."""

from typing import Optional

import chex
import jax
from jax import numpy as jnp
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.jax import types
from vizier.pyvizier import converters

from absl.testing import absltest
from absl.testing import parameterized

# pylint: disable=g-long-lambda


@chex.dataclass(frozen=True)
class FakeIncrementVectorizedStrategyState:
  """State for FakeIncrementVectorizedStrategy."""

  iterations: int


class FakeIncrementVectorizedStrategy(
    vb.VectorizedStrategy[FakeIncrementVectorizedStrategyState]
):
  """Fake vectorized strategy with incrementing suggestions."""

  def __init__(self, *args, **kwargs):
    pass

  def suggest(
      self,
      seed: jax.Array,
      state: FakeIncrementVectorizedStrategyState,
      n_parallel: int = 1,
  ) -> vb.VectorizedOptimizerInput:
    # The following structure allows to test the top K results.
    i = state.iterations
    suggestions = (
        jnp.array([
            [i % 10, i % 10],
            [(i + 1) % 10, (i + 1) % 10],
            [(i + 2) % 10, (i + 2) % 10],
            [(i + 3) % 10, (i + 3) % 10],
            [(i + 4) % 10, (i + 4) % 10],
        ])
        / 10
    )
    return vb.VectorizedOptimizerInput(
        jnp.repeat(suggestions[:, jnp.newaxis, :], n_parallel, axis=1),
        jnp.zeros([5, n_parallel, 0], dtype=types.INT_DTYPE),
    )

  @property
  def suggestion_batch_size(self) -> int:
    return 5

  def update(
      self,
      seed: jax.Array,
      state: FakeIncrementVectorizedStrategyState,
      batch_features: vb.VectorizedOptimizerInput,
      batch_rewards: types.Array,
  ) -> FakeIncrementVectorizedStrategyState:
    return FakeIncrementVectorizedStrategyState(iterations=state.iterations + 5)

  def init_state(
      self,
      seed: jax.Array,
      n_parallel: int = 1,
      *,
      prior_features: Optional[vb.VectorizedOptimizerInput] = None,
      prior_rewards: Optional[types.Array] = None,
  ) -> FakeIncrementVectorizedStrategyState:
    del seed
    return FakeIncrementVectorizedStrategyState(iterations=0)


# pylint: disable=unused-argument
def fake_increment_strategy_factory(
    converter: converters.TrialToModelInputConverter,
    suggestion_batch_size: int,
) -> vb.VectorizedStrategy:
  return FakeIncrementVectorizedStrategy()


@chex.dataclass(frozen=True)
class FakePriorTrialsStrategyState:
  """State for FakeIncrementVectorizedStrategy."""

  features: vb.VectorizedOptimizerInput
  rewards: types.Array


class FakePriorTrialsVectorizedStrategy(
    vb.VectorizedStrategy[FakePriorTrialsStrategyState]
):
  """Fake vectorized strategy to test prior trials."""

  def init_state(
      self,
      seed: jax.Array,
      n_parallel: int = 1,
      *,
      prior_features: Optional[vb.VectorizedOptimizerInput] = None,
      prior_rewards: Optional[types.Array] = None,
  ):
    if prior_rewards is not None and len(prior_rewards.shape) != 1:
      raise ValueError('Expected seed labels to have 1D dimension!')
    return FakePriorTrialsStrategyState(
        features=prior_features, rewards=prior_rewards
    )

  def suggest(
      self,
      seed: jax.Array,
      state: FakePriorTrialsStrategyState,
      n_parallel: int = 1,
  ) -> vb.VectorizedOptimizerInput:
    return vb.VectorizedOptimizerInput(
        continuous=state.features.continuous[
            jnp.argmax(state.rewards, axis=-1)
        ][jnp.newaxis, :],
        categorical=jnp.zeros([1, n_parallel, 0], dtype=types.INT_DTYPE),
    )

  @property
  def suggestion_batch_size(self) -> int:
    return 1

  def update(
      self,
      seed: jax.Array,
      state: FakePriorTrialsStrategyState,
      batch_features: vb.VectorizedOptimizerInput,
      batch_rewards: types.Array,
  ) -> FakePriorTrialsStrategyState:
    return state


# pylint: disable=unused-argument
def fake_prior_trials_strategy_factory(
    converter: converters.TrialToModelInputConverter,
    suggestion_batch_size: int,
) -> vb.VectorizedStrategy:
  return FakePriorTrialsVectorizedStrategy()


class VectorizedBaseTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  def test_optimize_candidates_len(self, count):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    converter = converters.TrialToModelInputConverter.from_problem(problem)
    score_fn = lambda x, _: jnp.sum(x.continuous.padded_array, axis=-1)
    optimizer = vb.VectorizedOptimizerFactory(
        strategy_factory=fake_increment_strategy_factory,
        max_evaluations=100,
    )(converter=converter)
    res_array = optimizer(score_fn=score_fn, count=count)
    res = vb.best_candidates_to_trials(res_array, converter=converter)
    self.assertLen(res, count)

  @parameterized.parameters(
      (1, 3),
      (2, 4),
  )
  def test_optimize_parallel_candidates_len(self, count, n_parallel):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    converter = converters.TrialToModelInputConverter.from_problem(problem)
    score_fn = lambda x, _: jnp.sum(x.continuous.padded_array, axis=(-1, -2))
    optimizer = vb.VectorizedOptimizerFactory(
        strategy_factory=fake_increment_strategy_factory,
        max_evaluations=100,
    )(converter=converter)
    res_array = optimizer(score_fn=score_fn, count=count, n_parallel=n_parallel)
    res = vb.best_candidates_to_trials(res_array, converter=converter)
    self.assertLen(res, count * n_parallel)

  @parameterized.parameters(True, False)
  def test_best_candidates_count_is_1(self, use_fori):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    converter = converters.TrialToModelInputConverter.from_problem(problem)
    score_fn = lambda x, _: -jnp.max(
        jnp.square(x.continuous.padded_array - 0.52), axis=-1
    )
    strategy_factory = FakeIncrementVectorizedStrategy
    optimizer = vb.VectorizedOptimizerFactory(
        strategy_factory=strategy_factory,
        suggestion_batch_size=5,
        max_evaluations=10,
        use_fori=use_fori,
    )(converter=converter)
    best_candidates_array = optimizer(score_fn=score_fn, count=1)
    best_candidates = vb.best_candidates_to_trials(
        best_candidates_array, converter=converter
    )
    # check the best candidate
    self.assertEqual(best_candidates[0].parameters['f1'].value, 0.5)
    self.assertEqual(best_candidates[0].parameters['f2'].value, 0.5)
    self.assertAlmostEqual(
        best_candidates[0]
        .final_measurement_or_die.metrics['acquisition']
        .value,
        -((0.5 - 0.52) ** 2),
    )

  @parameterized.parameters(True, False)
  def test_best_candidates_count_is_3(self, use_fori):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    converter = converters.TrialToModelInputConverter.from_problem(problem)
    score_fn = lambda x, _: -jnp.max(
        jnp.square(x.continuous.padded_array - 0.52), axis=-1
    )
    optimizer = vb.VectorizedOptimizerFactory(
        strategy_factory=fake_increment_strategy_factory,
        suggestion_batch_size=5,
        max_evaluations=10,
        use_fori=use_fori,
    )(converter=converter)
    best_candidates_array = optimizer(score_fn=score_fn, count=3)
    best_candidates = vb.best_candidates_to_trials(
        best_candidates_array, converter=converter
    )
    # check 1st best candidate
    self.assertAlmostEqual(best_candidates[0].parameters['f1'].value, 0.5)
    self.assertAlmostEqual(best_candidates[0].parameters['f2'].value, 0.5)
    self.assertAlmostEqual(
        best_candidates[0]
        .final_measurement_or_die.metrics['acquisition']
        .value,
        -((0.5 - 0.52) ** 2),
    )
    # check 2nd best candidate
    self.assertAlmostEqual(best_candidates[1].parameters['f1'].value, 0.6)
    self.assertAlmostEqual(best_candidates[1].parameters['f2'].value, 0.6)
    self.assertAlmostEqual(
        best_candidates[1]
        .final_measurement_or_die.metrics['acquisition']
        .value,
        -((0.6 - 0.52) ** 2),
    )
    # check 3rd best candidate
    self.assertAlmostEqual(best_candidates[2].parameters['f1'].value, 0.4)
    self.assertAlmostEqual(best_candidates[2].parameters['f2'].value, 0.4)
    self.assertAlmostEqual(
        best_candidates[2]
        .final_measurement_or_die.metrics['acquisition']
        .value,
        -((0.4 - 0.52) ** 2),
    )

  def test_vectorized_optimizer_factory(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    converter = converters.TrialToModelInputConverter.from_problem(problem)
    optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=fake_increment_strategy_factory,
        suggestion_batch_size=5,
        max_evaluations=1000,
    )
    optimizer = optimizer_factory(converter)
    self.assertEqual(optimizer.max_evaluations, 1000)
    self.assertEqual(optimizer.suggestion_batch_size, 5)

  @parameterized.parameters(True, False)
  def test_prior_trials(self, use_fori):
    """Test that the optimizer can correctly parsae and pass seed trials."""
    optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=fake_prior_trials_strategy_factory,
        suggestion_batch_size=5,
        max_evaluations=100,
        use_fori=use_fori,
    )

    study_config = vz.ProblemStatement(
        metric_information=[
            vz.MetricInformation(
                name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = study_config.search_space.root
    root.add_float_param('x1', 0.0, 10.0)
    root.add_float_param('x2', 0.0, 10.0)
    converter = converters.TrialToModelInputConverter.from_problem(study_config)
    optimizer = optimizer_factory(converter)

    trial1 = vz.Trial(parameters={'x1': 1, 'x2': 1})
    measurement1 = vz.Measurement(metrics={'obj': vz.Metric(value=-10.33)})
    trial1.complete(measurement1, inplace=True)

    trial2 = vz.Trial(parameters={'x1': 2, 'x2': 2})
    measurement2 = vz.Measurement(metrics={'obj': vz.Metric(value=5.0)})
    trial2.complete(measurement2, inplace=True)

    prior_features = vb.trials_to_sorted_array(
        [trial1, trial2, trial1], converter=converter
    )
    best_trial_array = optimizer(
        lambda x, _: -jnp.max(
            jnp.square(x.continuous.padded_array - 0.52), axis=-1
        ),
        count=1,
        prior_features=prior_features,
    )
    best_trial = vb.best_candidates_to_trials(
        best_trial_array, converter=converter
    )
    self.assertEqual(best_trial[0].parameters['x1'].value, 2)
    self.assertEqual(best_trial[0].parameters['x2'].value, 2)

    best_trial_array = optimizer(
        lambda x, _: -jnp.max(
            jnp.square(x.continuous.padded_array - 0.52), axis=-1
        ),
        count=1,
        prior_features=vb.trials_to_sorted_array([trial1], converter=converter),
    )
    best_trial = vb.best_candidates_to_trials(
        best_trial_array, converter=converter
    )
    self.assertEqual(best_trial[0].parameters['x1'].value, 1)
    self.assertEqual(best_trial[0].parameters['x2'].value, 1)

  @parameterized.parameters(True, False)
  def test_prior_trials_parallel(self, use_fori):
    optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=fake_prior_trials_strategy_factory,
        suggestion_batch_size=5,
        max_evaluations=100,
    )

    study_config = vz.ProblemStatement(
        metric_information=[
            vz.MetricInformation(
                name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = study_config.search_space.root
    root.add_float_param('x1', 0.0, 10.0)
    root.add_float_param('x2', 0.0, 10.0)
    root.add_float_param('x3', 0.0, 10.0)
    converter = converters.TrialToModelInputConverter.from_problem(study_config)
    optimizer = optimizer_factory(converter)
    prior_features = types.ModelInput(
        continuous=types.PaddedArray.as_padded(
            jax.random.uniform(jax.random.PRNGKey(0), (14, 3))
        ),
        categorical=types.PaddedArray.as_padded(
            jnp.zeros([14, 0], dtype=types.INT_DTYPE)
        ),
    )
    suggestions = optimizer(
        lambda x, _: -jnp.max(
            jnp.square(x.continuous.padded_array - 0.52), axis=(-1, -2)
        ),
        prior_features=prior_features,
        n_parallel=2,
    )
    self.assertSequenceEqual(suggestions.features.continuous.shape, (1, 2, 3))
    self.assertSequenceEqual(suggestions.rewards.shape, (1,))


if __name__ == '__main__':
  absltest.main()
