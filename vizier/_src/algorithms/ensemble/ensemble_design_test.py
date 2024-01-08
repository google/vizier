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

import numpy as np
from vizier._src.algorithms.ensemble import ensemble_design

from absl.testing import absltest


class EnsembleDesignTest(absltest.TestCase):

  def testRandom(self):
    indices = [0, 1]
    rewards = [(1, 1), (0, 0), (1, 1)]
    strategy = ensemble_design.RandomEnsembleDesign(indices=indices)

    for reward in rewards:
      strategy.update(reward)
      np.testing.assert_array_equal(strategy.ensemble_probs, [0.5, 0.5])

  def testEXP3IX(self):
    indices = [0, 1]
    rewards = [(1, 1), (0, 0), (1, 1)]
    strategy = ensemble_design.EXP3IXEnsembleDesign(
        indices=indices, stepsize=1.0
    )

    np.testing.assert_array_equal(strategy.ensemble_probs, [0.5, 0.5])

    for reward in rewards:
      strategy.update(reward)

    probs = strategy.ensemble_probs
    self.assertLen(probs, 2)
    self.assertGreater(probs[1], 0.55)
    self.assertGreater(probs[0], 0.3)

  def testEXP3Uniform(self):
    indices = [0, 1]
    rewards = [(1, 1), (0, 0), (1, 1)]
    strategy = ensemble_design.EXP3UniformEnsembleDesign(
        indices=indices, stepsize=1.0
    )

    np.testing.assert_array_equal(strategy.ensemble_probs, [0.5, 0.5])

    for reward in rewards:
      strategy.update(reward)

    probs = strategy.ensemble_probs
    self.assertLen(probs, 2)
    self.assertGreater(probs[1], 0.7)
    self.assertGreater(probs[0], 0.25)

  def testAdaptiveStrategyFixed(self):
    indices = [0, 1]
    rewards = [(1, 1), (0, 0.0), (1, 1)] * 3

    strategy = ensemble_design.AdaptiveEnsembleDesign(
        indices=indices,
        max_lengths=[5, 10],
    )

    np.testing.assert_array_equal(strategy.ensemble_probs, [0.5, 0.5])

    for reward in rewards:
      strategy.update(reward)

    # We should still favor 1 greatly.
    self.assertGreater(strategy.ensemble_probs[1], 0.7)
    self.assertGreater(strategy.observation_probs[1], 0.7)

  def testAdaptiveStrategy(self):
    indices = [0, 1]
    rewards = [(0, 0), (1, 1), (1, 0), (0, 1)]

    strategy = ensemble_design.AdaptiveEnsembleDesign(
        indices=indices,
        max_lengths=[2, 4],
        base_stepsize=1 / np.sqrt(2),
        meta_stepsize=1 / np.sqrt(2),
    )

    np.testing.assert_array_equal(strategy.ensemble_probs, [0.5, 0.5])

    for reward in rewards:
      strategy.update(reward)

    probs = strategy.ensemble_probs
    self.assertLen(probs, 2)
    # We have adapted to favor index 0 quickly.
    self.assertGreater(probs[0], 0.45)


if __name__ == "__main__":
  absltest.main()
