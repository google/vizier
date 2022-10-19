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

"""Tests for vectorized_base."""

import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier.pyvizier import converters

from absl.testing import absltest


class DummyVectorizedStrategy(vb.VectorizedStrategy):

  def __init__(self, count: int):
    self._count = count

  def suggest(self) -> np.ndarray:
    return np.ones((5, 2))

  @property
  def suggestion_count(self) -> int:
    return 5

  @property
  def best_features_results(self) -> list[vb.VectorizedStrategyResult]:
    return [vb.VectorizedStrategyResult(np.ones(2), 0.0)] * self._count

  def update(self, rewards: np.ndarray) -> None:
    pass


class VectorizedBaseTest(absltest.TestCase):

  def test_optimize_candidates_len(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: np.sum(x, axis=-1)
    strategy_factory = lambda converter, count: DummyVectorizedStrategy(count)
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=strategy_factory, max_evaluations=100)
    res = optimizer.optimize(converter=converter, score_fn=score_fn, count=5)
    self.assertLen(res, 5)
    self.assertLen(optimizer.best_candidates, 5)

  def test_should_stop(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: np.sum(x, axis=-1)
    strategy_factory = lambda converter, count: DummyVectorizedStrategy(count)
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=strategy_factory, max_evaluations=100)
    optimizer.optimize(converter=converter, score_fn=score_fn, count=5)
    self.assertEqual(optimizer._evaluated_count, 100)


if __name__ == '__main__':
  absltest.main()
