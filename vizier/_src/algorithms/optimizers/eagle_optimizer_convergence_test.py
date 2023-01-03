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

"""Tests for eagle_optimizer."""

import logging
from typing import Optional

import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import eagle_strategy
from vizier._src.algorithms.optimizers import random_vectorized_optimizer as rvo
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.algorithms.testing import comparator_runner
from vizier.pyvizier import converters

from absl.testing import absltest
from absl.testing import parameterized


def randomize_array(converter: converters.TrialToArrayConverter) -> np.ndarray:
  """Generate a random array of features to be used as score_fn shift."""
  features_arrays = []
  for spec in converter.output_specs:
    if spec.type == converters.NumpyArraySpecType.ONEHOT_EMBEDDING:
      dim = spec.num_dimensions - spec.num_oovs
      features_arrays.append(
          np.eye(spec.num_dimensions)[np.random.randint(0, dim)])
    elif spec.type == converters.NumpyArraySpecType.CONTINUOUS:
      features_arrays.append(np.random.uniform(0.4, 0.6, size=(1,)))
    else:
      raise ValueError(f'The type {spec.type} is not supported!')
  return np.hstack(features_arrays)


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
def sphere(x: np.ndarray) -> np.ndarray:
  return -np.sum(np.square(x), axis=-1)


def rastrigin_d10(x: np.ndarray) -> np.ndarray:
  return 10 * np.sum(
      np.cos(2 * np.pi * x), axis=-1) - np.sum(
          np.square(x), axis=-1)


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
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    eagle_strategy_factory = eagle_strategy.VectorizedEagleStrategyFactory(
        eagle_config=eagle_strategy.EagleStrategyConfig())
    eagle_optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=eagle_strategy_factory)
    optimum_features = randomize_array(converter)
    shifted_score_fn = lambda x, shift=optimum_features: score_fn(x - shift)
    shifted_score_fn = score_fn
    random_optimizer_factory = rvo.create_random_optimizer_factory()
    # Run simple regret convergence test.
    comparator_runner.SimpleRegretComparisonTester(
        baseline_num_trials=2 * evaluations,
        candidate_num_trials=evaluations,
        baseline_suggestion_batch_size=5,
        candidate_suggestion_batch_size=5,
        baseline_num_repeats=5,
        candidate_num_repeats=3,
        alpha=0.05,
        goal=vz.ObjectiveMetricGoal.MAXIMIZE
    ).assert_optimizer_better_simple_regret(
        converter=converter,
        score_fn=shifted_score_fn,
        baseline_optimizer_factory=random_optimizer_factory,
        candidate_optimizer_factory=eagle_optimizer_factory)


if __name__ == '__main__':
  absltest.main()
