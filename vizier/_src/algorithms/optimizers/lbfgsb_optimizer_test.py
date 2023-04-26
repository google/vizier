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

"""Tests for lbfgsb_optimizer."""

import jax.numpy as jnp
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import lbfgsb_optimizer as lo
from vizier.pyvizier import converters

from absl.testing import absltest
from absl.testing import parameterized


class LBFGSBOptimizer(parameterized.TestCase):

  def test_optimize_candidates_len(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    problem.search_space.root.add_float_param('f3', 0.0, 10.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: jnp.sum(x, axis=-1)
    optimizer = lo.LBFGSBOptimizer(random_restarts=10)
    res = optimizer.optimize(converter=converter, score_fn=score_fn, count=1)
    self.assertLen(res, 1)

  def test_best_candidates_count_is_1(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: -jnp.sum(jnp.square(x - 0.52), axis=-1)
    optimizer = lo.LBFGSBOptimizer(random_restarts=10)
    candidates = optimizer.optimize(
        converter=converter, score_fn=score_fn, count=1
    )
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
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    # Minimize sum over all features.
    score_fn = lambda x: -jnp.sum(jnp.square(x - 0.52), axis=[-1, -2])
    optimizer = lo.LBFGSBOptimizer(parallel_batch_size=3, random_restarts=10)
    best_candidates = optimizer.optimize(
        converter=converter, score_fn=score_fn, count=1
    )
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
