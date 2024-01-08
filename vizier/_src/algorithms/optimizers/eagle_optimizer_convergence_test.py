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

"""Tests for eagle_optimizer."""

import logging
from typing import Optional

import jax
from jax import numpy as jnp
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import eagle_strategy
from vizier._src.algorithms.optimizers import random_vectorized_optimizer as rvo
from vizier._src.algorithms.testing import comparator_runner
from vizier._src.jax import types
from vizier.pyvizier import converters

from absl.testing import absltest
from absl.testing import parameterized


def create_continuous_problem(
    n_features: int,
    problem: Optional[vz.ProblemStatement] = None) -> vz.ProblemStatement:
  if not problem:
    problem = vz.ProblemStatement()
  root = problem.search_space.select_root()
  for i in range(n_features):
    root.add_float_param('x%d' % i, -5.0, 5.0)
  return problem


def create_categorical_problem(
    n_features: int,
    categorical_dim: int = 6,
    problem: Optional[vz.ProblemStatement] = None) -> vz.ProblemStatement:
  if not problem:
    problem = vz.ProblemStatement()
  root = problem.search_space.select_root()
  for i in range(n_features):
    root.add_categorical_param(
        'c%d' % i, feasible_values=[str(i) for i in range(categorical_dim)])
  return problem


def create_mix_problem(n_features: int,
                       categorical_dim: int = 8) -> vz.ProblemStatement:
  problem = create_continuous_problem(n_features // 2)
  return create_categorical_problem(n_features // 2, categorical_dim, problem)


# TODO: Change to bbob functions when they can support batching.
def sphere(x: types.ModelInput) -> jax.Array:
  return -(
      jnp.sum(jnp.square(x.continuous.padded_array), axis=-1)
      + 0.1 * jnp.sum(jnp.square(x.categorical.padded_array), axis=-1)
  )


def _rastrigin_d10_part(x: types.Array) -> jax.Array:
  return 10 * jnp.sum(jnp.cos(2 * np.pi * x), axis=-1) - jnp.sum(
      jnp.square(x), axis=-1
  )


def rastrigin_d10(x: types.ModelInput) -> jax.Array:
  return _rastrigin_d10_part(x.continuous.padded_array) + _rastrigin_d10_part(
      0.01 * x.categorical.padded_array
  )


class EagleOptimizerConvegenceTest(parameterized.TestCase):
  """Test optimizing an acquisition functions using vectorized Eagle Strategy.
  """

  @absltest.skip("Test takes too long externally.")
  @parameterized.product(
      create_problem_fn=[
          create_continuous_problem,
          create_categorical_problem,
          create_mix_problem,
      ],
      n_features=[10, 20],
      score_fn=[sphere, rastrigin_d10],
  )
  def test_converges(self, create_problem_fn, n_features, score_fn):
    logging.info('Starting a new convergence test (n_features: %s)', n_features)
    evaluations = 20_000
    problem = create_problem_fn(n_features)
    converter = converters.TrialToModelInputConverter.from_problem(problem)
    eagle_strategy_factory = eagle_strategy.VectorizedEagleStrategyFactory(
        eagle_config=eagle_strategy.EagleStrategyConfig())
    shifted_score_fn = score_fn
    random_strategy_factory = rvo.random_strategy_factory
    # Run simple regret convergence test.
    comparator_runner.SimpleRegretComparisonTester(
        baseline_num_trials=2 * evaluations,
        candidate_num_trials=evaluations,
        baseline_suggestion_batch_size=5,
        candidate_suggestion_batch_size=5,
        baseline_num_repeats=5,
        candidate_num_repeats=3,
        alpha=0.05,
        goal=vz.ObjectiveMetricGoal.MAXIMIZE,
    ).assert_optimizer_better_simple_regret(
        converter=converter,
        score_fn=shifted_score_fn,
        baseline_strategy_factory=random_strategy_factory,
        candidate_strategy_factory=eagle_strategy_factory,
    )


if __name__ == '__main__':
  absltest.main()
