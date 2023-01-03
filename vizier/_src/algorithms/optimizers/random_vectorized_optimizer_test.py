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

"""Tests for random_vectorized_optimizer."""

import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import random_vectorized_optimizer as rvo
from vizier.pyvizier import converters

from absl.testing import absltest


class RandomVectorizedOptimizerTest(absltest.TestCase):
  """Tests for random_vectorized_optimizer."""

  def test_random_optimizer(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: np.sum(x, axis=-1)
    random_optimizer = rvo.create_random_optimizer(
        max_evaluations=100, suggestion_batch_size=10)
    res = random_optimizer.optimize(
        converter=converter, score_fn=score_fn, count=5)
    self.assertLen(res, 5)

  def test_random_optimizer_factory(self):
    random_optimizer_factory = rvo.create_random_optimizer_factory()
    random_optimizer = random_optimizer_factory(
        max_evaluations=100, suggestion_batch_size=10)
    self.assertEqual(random_optimizer.suggestion_batch_size, 10)
    self.assertEqual(random_optimizer.max_evaluations, 100)


if __name__ == '__main__':
  absltest.main()
