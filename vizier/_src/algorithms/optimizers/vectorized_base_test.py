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


@chex.dataclass(frozen=True)
class FakeIncrementVectorizedStrategyState:
  """State for FakeIncrementVectorizedStrategy."""

  iterations: int


class FakeIncrementVectorizedStrategy(vb.VectorizedStrategy):
  """Fake vectorized strategy with incrementing suggestions."""

  def __init__(self, *args, **kwargs):
    pass

  def suggest(
      self,
      state: FakeIncrementVectorizedStrategyState,
      seed: jax.random.KeyArray,
  ) -> jax.Array:
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
    return suggestions

  @property
  def suggestion_batch_size(self) -> int:
    return 5

  def update(
      self,
      state: FakeIncrementVectorizedStrategyState,
      batch_features: types.Array,
      batch_rewards: types.Array,
      seed: jax.random.KeyArray,
  ) -> FakeIncrementVectorizedStrategyState:
    return FakeIncrementVectorizedStrategyState(iterations=state.iterations + 5)

  def init_state(
      self,
      seed: jax.random.KeyArray,
      prior_features: Optional[types.Array] = None,
      prior_rewards: Optional[types.Array] = None,
  ) -> FakeIncrementVectorizedStrategyState:
    del seed
    return FakeIncrementVectorizedStrategyState(iterations=0)


# pylint: disable=unused-argument
def fake_increment_strategy_factory(
    converter: converters.TrialToArrayConverter,
    suggestion_batch_size: int,
) -> vb.VectorizedStrategy:
  return FakeIncrementVectorizedStrategy()


@chex.dataclass(frozen=True)
class FakePriorTrialsStrategyState:
  """State for FakeIncrementVectorizedStrategy."""

  features: types.Array
  rewards: types.Array


class FakePriorTrialsVectorizedStrategy(vb.VectorizedStrategy):
  """Fake vectorized strategy to test prior trials."""

  def init_state(
      self,
      seed: jax.random.KeyArray,
      prior_features: Optional[types.Array] = None,
      prior_rewards: Optional[types.Array] = None,
  ):
    if prior_rewards is not None and len(prior_rewards.shape) != 1:
      raise ValueError('Expected seed labels to have 1D dimension!')
    return FakePriorTrialsStrategyState(
        features=prior_features, rewards=prior_rewards
    )

  def suggest(
      self, state: FakePriorTrialsStrategyState, seed: jax.random.KeyArray
  ) -> jax.Array:
    return state.features[jnp.argmax(state.rewards, axis=-1)].reshape(1, -1)

  @property
  def suggestion_batch_size(self) -> int:
    return 1

  def update(
      self,
      state: FakePriorTrialsStrategyState,
      batch_features: types.Array,
      batch_rewards: types.Array,
      seed: jax.random.KeyArray,
  ) -> FakePriorTrialsStrategyState:
    return state


# pylint: disable=unused-argument
def fake_prior_trials_strategy_factory(
    converter: converters.TrialToArrayConverter,
    suggestion_batch_size: int,
) -> vb.VectorizedStrategy:
  return FakePriorTrialsVectorizedStrategy()


class VectorizedBaseTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  def test_optimize_candidates_len(self, count):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: jnp.sum(x, axis=-1)
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=fake_increment_strategy_factory,
        max_evaluations=100,
    )
    res = optimizer.optimize(
        converter=converter, score_fn=score_fn, count=count
    )
    self.assertLen(res, count)

  def test_best_candidates_count_is_1(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: -jnp.max(jnp.square(x - 0.52), axis=-1)
    strategy_factory = FakeIncrementVectorizedStrategy
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=strategy_factory,
        suggestion_batch_size=5,
        max_evaluations=10,
    )
    best_candidates = optimizer.optimize(
        converter=converter, score_fn=score_fn, count=1
    )
    # check the best candidate
    self.assertEqual(best_candidates[0].parameters['f1'].value, 0.5)
    self.assertEqual(best_candidates[0].parameters['f2'].value, 0.5)
    self.assertAlmostEqual(
        best_candidates[0].final_measurement.metrics['acquisition'].value,
        -((0.5 - 0.52) ** 2),
    )

  def test_best_candidates_count_is_3(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: -jnp.max(jnp.square(x - 0.52), axis=-1)
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=fake_increment_strategy_factory,
        suggestion_batch_size=5,
        max_evaluations=10,
    )
    best_candidates = optimizer.optimize(
        converter=converter, score_fn=score_fn, count=3
    )
    # check 1st best candidate
    self.assertAlmostEqual(best_candidates[0].parameters['f1'].value, 0.5)
    self.assertAlmostEqual(best_candidates[0].parameters['f2'].value, 0.5)
    self.assertAlmostEqual(
        best_candidates[0].final_measurement.metrics['acquisition'].value,
        -((0.5 - 0.52) ** 2),
    )
    # check 2nd best candidate
    self.assertAlmostEqual(best_candidates[1].parameters['f1'].value, 0.6)
    self.assertAlmostEqual(best_candidates[1].parameters['f2'].value, 0.6)
    self.assertAlmostEqual(
        best_candidates[1].final_measurement.metrics['acquisition'].value,
        -((0.6 - 0.52) ** 2),
    )
    # check 3rd best candidate
    self.assertAlmostEqual(best_candidates[2].parameters['f1'].value, 0.4)
    self.assertAlmostEqual(best_candidates[2].parameters['f2'].value, 0.4)
    self.assertAlmostEqual(
        best_candidates[2].final_measurement.metrics['acquisition'].value,
        -((0.4 - 0.52) ** 2),
    )

  def test_vectorized_optimizer_factory(self):
    optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=fake_increment_strategy_factory
    )
    optimizer = optimizer_factory(suggestion_batch_size=5, max_evaluations=1000)
    self.assertEqual(optimizer.max_evaluations, 1000)
    self.assertEqual(optimizer.suggestion_batch_size, 5)

  def test_prior_trials(self):
    """Test that the optimizer can correctly parsae and pass seed trials."""
    optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=fake_prior_trials_strategy_factory
    )
    optimizer = optimizer_factory(suggestion_batch_size=5, max_evaluations=100)

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
    converter = converters.TrialToArrayConverter.from_study_config(study_config)

    trial1 = vz.Trial(parameters={'x1': 1, 'x2': 1})
    measurement1 = vz.Measurement(metrics={'obj': vz.Metric(value=-10.33)})
    trial1.complete(measurement1, inplace=True)

    trial2 = vz.Trial(parameters={'x1': 2, 'x2': 2})
    measurement2 = vz.Measurement(metrics={'obj': vz.Metric(value=5.0)})
    trial2.complete(measurement2, inplace=True)

    best_trial = optimizer.optimize(
        converter,
        lambda x: -jnp.max(jnp.square(x - 0.52), axis=-1),
        count=1,
        prior_trials=[trial1, trial2, trial1],
    )
    self.assertEqual(best_trial[0].parameters['x1'].value, 2)
    self.assertEqual(best_trial[0].parameters['x2'].value, 2)

    best_trial = optimizer.optimize(
        converter,
        lambda x: -jnp.max(jnp.square(x - 0.52), axis=-1),
        count=1,
        prior_trials=[trial1],
    )
    self.assertEqual(best_trial[0].parameters['x1'].value, 1)
    self.assertEqual(best_trial[0].parameters['x2'].value, 1)


if __name__ == '__main__':
  absltest.main()
