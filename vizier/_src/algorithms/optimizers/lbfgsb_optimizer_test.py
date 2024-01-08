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

"""Tests for lbfgsb_optimizer."""

import jax.numpy as jnp
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import lbfgsb_optimizer as lo
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier.pyvizier import converters

from absl.testing import absltest
from absl.testing import parameterized


class LBFGSBOptimizer(parameterized.TestCase):

  def test_optimize_candidates_len(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    problem.search_space.root.add_float_param('f3', 0.0, 10.0)
    converter = converters.TrialToModelInputConverter.from_problem(problem)
    score_fn = lambda x, _: jnp.sum(x.continuous.padded_array, axis=-1)
    optimizer = lo.LBFGSBOptimizerFactory(random_restarts=10, maxiter=20)(
        converter
    )
    res = optimizer(score_fn=score_fn)
    self.assertLen(res.rewards, 1)

  def test_best_candidates_count_is_1(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    converter = converters.TrialToModelInputConverter.from_problem(problem)
    optimizer = lo.LBFGSBOptimizerFactory(random_restarts=10, maxiter=20)(
        converter
    )
    score_fn = lambda x, _: -jnp.sum(  # pylint: disable=g-long-lambda
        jnp.square(x.continuous.padded_array - 0.52), axis=-1
    )
    results = optimizer(score_fn=score_fn)
    candidates = vb.best_candidates_to_trials(results, converter)
    # check the best candidate
    self.assertLessEqual(
        np.abs(candidates[0].parameters['f1'].value - 0.52), 1e-6
    )
    self.assertLessEqual(
        np.abs(candidates[0].parameters['f2'].value - 0.52), 1e-6
    )
    self.assertIsNotNone(candidates[0].final_measurement)
    if candidates[0].final_measurement:
      self.assertLessEqual(
          np.abs(candidates[0].final_measurement.metrics['acquisition'].value),
          1e-6,
      )

  def test_batch_candidates(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    problem.search_space.root.add_float_param('f3', 0.0, 1.0)
    converter = converters.TrialToModelInputConverter.from_problem(problem)
    # Minimize sum over all features.
    score_fn = lambda x, _: -jnp.sum(  # pylint: disable=g-long-lambda
        jnp.square(x.continuous.padded_array - 0.52),
        axis=[-1, -2],
    )
    optimizer = lo.LBFGSBOptimizerFactory(random_restarts=10, maxiter=20)(
        converter
    )
    best_results = optimizer(score_fn=score_fn, count=1, n_parallel=3)
    best_candidates = vb.best_candidates_to_trials(best_results, converter)
    self.assertLen(best_candidates, 3)
    # check the best candidates
    for b in best_candidates:
      self.assertLessEqual(np.abs(b.parameters['f1'].value - 0.52), 1e-6)
      self.assertLessEqual(np.abs(b.parameters['f2'].value - 0.52), 1e-6)
      self.assertIsNotNone(b.final_measurement)
      if b.final_measurement:
        self.assertLessEqual(
            np.abs(b.final_measurement.metrics['acquisition'].value), 1e-6
        )


if __name__ == '__main__':
  absltest.main()
