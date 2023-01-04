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

import datetime
import time
from typing import Optional

import mock
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier.pyvizier import converters

from absl.testing import absltest
from absl.testing import parameterized


class FakeIncrementVectorizedStrategy(vb.VectorizedStrategy):
  """Fake vectorized strategy with incrementing suggestions."""

  def __init__(self, *args, **kwargs):
    self._iter = 0

  def suggest(self) -> np.ndarray:
    # The following structure allows to test the top K results.
    suggestions = (
        np.array([
            [self._iter % 10, self._iter % 10],
            [(self._iter + 1) % 10, (self._iter + 1) % 10],
            [(self._iter + 2) % 10, (self._iter + 2) % 10],
            [(self._iter + 3) % 10, (self._iter + 3) % 10],
            [(self._iter + 4) % 10, (self._iter + 4) % 10],
        ])
        / 10
    )
    self._iter += 5
    return suggestions

  @property
  def suggestion_batch_size(self) -> int:
    return 5

  def update(self, rewards: np.ndarray) -> None:
    pass


# pylint: disable=unused-argument
def fake_increment_strategy_factory(
    converter: converters.TrialToArrayConverter,
    suggestion_batch_size: int,
    seed: Optional[int] = None,
    prior_features: Optional[np.ndarray] = None,
    prior_rewards: Optional[np.ndarray] = None,
) -> vb.VectorizedStrategy:
  return FakeIncrementVectorizedStrategy()


class FakePriorTrialsVectorizedStrategy(vb.VectorizedStrategy):
  """Fake vectorized strategy to test prior trials."""

  def __init__(self, prior_features: np.ndarray, prior_rewards: np.ndarray):
    self.seed_features, self.seed_rewards = prior_features, prior_rewards
    if len(self.seed_rewards.shape) != 1:
      raise ValueError('Expected seed labels to have 1D dimension!')

  def suggest(self) -> np.ndarray:
    return self.seed_features[np.argmax(self.seed_rewards, axis=-1)].reshape(
        1, -1
    )

  @property
  def suggestion_batch_size(self) -> int:
    return 1

  def update(self, rewards: np.ndarray) -> None:
    pass


# pylint: disable=unused-argument
def fake_prior_trials_strategy_factory(
    converter: converters.TrialToArrayConverter,
    suggestion_batch_size: int,
    seed: Optional[int] = None,
    prior_features: Optional[np.ndarray] = None,
    prior_rewards: Optional[np.ndarray] = None,
) -> vb.VectorizedStrategy:
  return FakePriorTrialsVectorizedStrategy(prior_features, prior_rewards)


class VectorizedBaseTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  def test_optimize_candidates_len(self, count):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: np.sum(x, axis=-1)
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=fake_increment_strategy_factory,
        max_evaluations=100,
    )
    res = optimizer.optimize(
        converter=converter, score_fn=score_fn, count=count
    )
    self.assertLen(res, count)

  @parameterized.parameters([None, datetime.timedelta(minutes=10)])
  def test_should_stop_max_evaluations(self, max_duration):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = mock.Mock()
    score_fn.side_effect = lambda x: np.sum(x, axis=-1)
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=fake_increment_strategy_factory,
        max_evaluations=100,
        max_duration=max_duration,
    )
    optimizer.optimize(converter=converter, score_fn=score_fn, count=3)
    # The batch size is 5, so we expect 100/5 = 20 calls
    self.assertEqual(score_fn.call_count, 20)

  def test_best_candidates_count_is_1(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: -np.max(np.square(x - 0.52), axis=-1)
    strategy_factory = FakeIncrementVectorizedStrategy
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=strategy_factory, max_evaluations=10
    )
    best_candidates = optimizer.optimize(
        converter=converter, score_fn=score_fn, count=1
    )
    # check the best candidate
    self.assertEqual(best_candidates[0].parameters['f1'].value, 0.5)
    self.assertEqual(best_candidates[0].parameters['f2'].value, 0.5)
    self.assertEqual(
        best_candidates[0].final_measurement.metrics['acquisition'].value,
        -((0.5 - 0.52) ** 2),
    )

  def test_best_candidates_count_is_3(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 1.0)
    problem.search_space.root.add_float_param('f2', 0.0, 1.0)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    score_fn = lambda x: -np.max(np.square(x - 0.52), axis=-1)
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=fake_increment_strategy_factory, max_evaluations=10
    )
    best_candidates = optimizer.optimize(
        converter=converter, score_fn=score_fn, count=3
    )
    # check 1st best candidate
    self.assertEqual(best_candidates[0].parameters['f1'].value, 0.5)
    self.assertEqual(best_candidates[0].parameters['f2'].value, 0.5)
    self.assertEqual(
        best_candidates[0].final_measurement.metrics['acquisition'].value,
        -((0.5 - 0.52) ** 2),
    )
    # check 2nd best candidate
    self.assertEqual(best_candidates[1].parameters['f1'].value, 0.6)
    self.assertEqual(best_candidates[1].parameters['f2'].value, 0.6)
    self.assertEqual(
        best_candidates[1].final_measurement.metrics['acquisition'].value,
        -((0.6 - 0.52) ** 2),
    )
    # check 3rd best candidate
    self.assertEqual(best_candidates[2].parameters['f1'].value, 0.4)
    self.assertEqual(best_candidates[2].parameters['f2'].value, 0.4)
    self.assertEqual(
        best_candidates[2].final_measurement.metrics['acquisition'].value,
        -((0.4 - 0.52) ** 2),
    )

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
    optimizer = vb.VectorizedOptimizer(
        strategy_factory=fake_increment_strategy_factory,
        max_evaluations=100,
        max_duration=datetime.timedelta(seconds=3),
    )
    optimizer.optimize(converter=converter, score_fn=score_fn)
    # Test the optimization stopped after ~3 seconds based on function calls.
    self.assertLess(score_fn.call_count, 4)

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
        lambda x: -np.max(np.square(x - 0.52), axis=-1),
        count=1,
        prior_trials=[trial1, trial2, trial1],
    )
    self.assertEqual(best_trial[0].parameters['x1'].value, 2)
    self.assertEqual(best_trial[0].parameters['x2'].value, 2)

    best_trial = optimizer.optimize(
        converter,
        lambda x: -np.max(np.square(x - 0.52), axis=-1),
        count=1,
        prior_trials=[trial1],
    )
    self.assertEqual(best_trial[0].parameters['x1'].value, 1)
    self.assertEqual(best_trial[0].parameters['x2'].value, 1)


if __name__ == '__main__':
  absltest.main()
