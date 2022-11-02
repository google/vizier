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

import datetime
import time

import mock
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier.pyvizier import converters

from absl.testing import absltest
from absl.testing import parameterized


class FakeVectorizedStrategy(vb.VectorizedStrategy):

  def __init__(self):
    self._iter = 0

  def suggest(self) -> np.ndarray:
    # The following structure allows to test the top K results.
    suggestions = np.array([
        [self._iter % 10, self._iter % 10],
        [(self._iter + 1) % 10, (self._iter + 1) % 10],
        [(self._iter + 2) % 10, (self._iter + 2) % 10],
        [(self._iter + 3) % 10, (self._iter + 3) % 10],
        [(self._iter + 4) % 10, (self._iter + 4) % 10],
    ]) / 10
    self._iter += 5
    return suggestions

  @property
  def suggestion_batch_size(self) -> int:
    return 5

  def update(self, rewards: np.ndarray) -> None:
    pass


class VectorizedBaseTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  def test_optimize_candidates_len(self, count):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: np.sum(x, axis=-1)
    strategy_factory = lambda converter, batch, seed: FakeVectorizedStrategy()
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=strategy_factory,
        max_evaluations=100,
    )
    res = optimizer.optimize(
        converter=converter, score_fn=score_fn, count=count)
    self.assertLen(res, count)

  @parameterized.parameters([None, datetime.timedelta(minutes=10)])
  def test_should_stop_max_evaluations(self, max_duration):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = mock.Mock()
    score_fn.side_effect = lambda x: np.sum(x, axis=-1)
    strategy_factory = lambda converter, batch, seed: FakeVectorizedStrategy()
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=strategy_factory,
        max_evaluations=100,
        max_duration=max_duration,
    )
    optimizer.optimize(converter=converter, score_fn=score_fn, count=3)
    # The batch size is 5, so we expect 100/5 = 20 calls
    self.assertEqual(score_fn.call_count, 20)

  def test_best_candidates(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: -np.max(np.square(x - 0.52), axis=-1)
    strategy_factory = lambda converter, batch, seed: FakeVectorizedStrategy()
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=strategy_factory, max_evaluations=10)
    best_candidates = optimizer.optimize(
        converter=converter, score_fn=score_fn, count=3)
    # check 1st best candidate
    self.assertEqual(best_candidates[0].parameters['f1'].value, 0.5)
    self.assertEqual(best_candidates[0].parameters['f2'].value, 0.5)
    self.assertEqual(
        best_candidates[0].final_measurement.metrics['acquisition'].value,
        -(0.5 - 0.52)**2)
    # check 2nd best candidate
    self.assertEqual(best_candidates[1].parameters['f1'].value, 0.6)
    self.assertEqual(best_candidates[1].parameters['f2'].value, 0.6)
    self.assertEqual(
        best_candidates[1].final_measurement.metrics['acquisition'].value,
        -(0.6 - 0.52)**2)
    # check 3rd best candidate
    self.assertEqual(best_candidates[2].parameters['f1'].value, 0.4)
    self.assertEqual(best_candidates[2].parameters['f2'].value, 0.4)
    self.assertEqual(
        best_candidates[2].final_measurement.metrics['acquisition'].value,
        -(0.4 - 0.52)**2)

  def test_should_stop_max_duration(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = mock.Mock()

    def slow_score_fn(x):
      time.sleep(1)
      return np.sum(x, axis=-1)

    score_fn.side_effect = slow_score_fn
    strategy_factory = lambda converter, batch, seed: FakeVectorizedStrategy()
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=strategy_factory,
        max_evaluations=100,
        max_duration=datetime.timedelta(seconds=3))
    optimizer.optimize(converter=converter, score_fn=score_fn)
    # Test the optimization stopped after ~3 seconds based on function calls.
    self.assertLess(score_fn.call_count, 4)

  def test_vectorized_optimizer_factory(self):
    strategy_factory = lambda converter, batch, seed: FakeVectorizedStrategy()
    optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=strategy_factory)
    optimizer = optimizer_factory(suggestion_batch_size=5, max_evaluations=1000)
    self.assertEqual(optimizer.max_evaluations, 1000)
    self.assertEqual(optimizer.suggestion_batch_size, 5)


if __name__ == '__main__':
  absltest.main()
