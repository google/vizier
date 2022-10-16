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
from vizier._src.algorithms.testing import convergence_runner
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
      raise ValueError('The type %s is not supported!' % spec.type)
  return np.hstack(features_arrays)


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
def sphere(x: np.ndarray) -> np.ndarray:
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
    optimum_features = randomize_array(converter)
    shifted_score_fn = lambda x, shift=optimum_features: sphere(x - shift)
    convergence_runner.assert_optimizer_converges(
        converter,
        optimizer,
        shifted_score_fn,
        optimum_features,
        evaluations,
        num_repeats=1,
        success_threshold=1)


if __name__ == '__main__':
  absltest.main()
