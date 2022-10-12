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

"""Tests for eagle_optimizer."""

import logging
from typing import Optional

import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import eagle_strategy
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.algorithms.testing import simple_regret_convergence_runner as srcr
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
    root.add_float_param('x%d' % i, 0.0, 1.0)
  return problem


def create_categorical_problem(
    n_features: int,
    categorical_dim: int = 8,
    problem: Optional[vz.ProblemStatement] = None) -> vz.ProblemStatement:
  if not problem:
    problem = vz.ProblemStatement()
  root = problem.search_space.select_root()
  for i in range(n_features):
    root.add_categorical_param(
        'c%d' % i, feasible_values=[str(i) for i in range(categorical_dim)])
  return problem


def create_mix_problem(n_features: int,
                       categrocial_dim: int = 8) -> vz.ProblemStatement:
  problem = create_continuous_problem(n_features)
  return create_categorical_problem(n_features, categrocial_dim, problem)


# TODO: Change to bbob functions when they can support batching.
def sphere_score_fn(x: np.ndarray) -> np.ndarray:
  return -np.sum(np.square(x), axis=-1)


class EagleOptimizerConvegenceTest(parameterized.TestCase):
  """Test optimizing an acquisition functions using vectorized Eagle Strategy.
  """

  @parameterized.product(
      create_problem=[
          create_continuous_problem,
          create_categorical_problem,
          create_mix_problem,
      ],
      n_features=list(range(10, 20)),
  )
  def test_converges(self, create_problem, n_features):
    logging.info('Starting a new convergence test (n_features: %s)', n_features)
    evaluations = 20_000
    problem = create_problem(n_features)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    eagle_factory = eagle_strategy.VectorizedEagleStrategyFactory(
        eagle_config=eagle_strategy.EagleStrategyConfig(),
        pool_size=50,
        batch_size=10,
        seed=1)
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=eagle_factory, max_evaluations=evaluations)
    srcr.assert_converges(
        converter,
        optimizer,
        sphere_score_fn,
        evaluations,
        num_repeats=1,
        success_threshold=1)


if __name__ == '__main__':
  absltest.main()
