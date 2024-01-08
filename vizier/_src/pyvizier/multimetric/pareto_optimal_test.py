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

"""Tests for pareto_optimal."""

import numpy as np

from vizier._src.pyvizier.multimetric import pareto_optimal
from absl.testing import absltest


class ParetoOptimalTest(absltest.TestCase):

  def setUp(self):
    super(ParetoOptimalTest, self).setUp()
    self.algo = pareto_optimal.FastParetoOptimalAlgorithm(
        base_algorithm=pareto_optimal.NaiveParetoOptimalAlgorithm())

  def test_is_pareto_optimal(self):
    points = np.array([[1, 2, 3], [1, 2, 3], [2, 4, 1], [1, 2, -1]])
    np.testing.assert_array_equal(
        self.algo.is_pareto_optimal(points), [True, True, True, False])

    # Adding a point that dominates the first two values.
    points = np.vstack([points, [2, 3, 4]])
    np.testing.assert_array_equal(
        self.algo.is_pareto_optimal(points), [False, False, True, False, True])

  def test_is_pareto_optimal_all_optimal(self):
    # Generate points on positive orthant sphere.
    points = abs(np.random.normal(size=(1000, 3)))
    points /= np.linalg.norm(points, axis=1)[..., np.newaxis]
    algo = pareto_optimal.FastParetoOptimalAlgorithm(
        base_algorithm=pareto_optimal.NaiveParetoOptimalAlgorithm(),
        recursive_threshold=100)
    self.assertTrue(np.all(algo.is_pareto_optimal(points)))

  def test_is_pareto_optimal_randomized(self):
    dim = 10
    points = np.random.normal(size=(1000, dim))
    # Make a copy of points to test equality case.
    points = np.vstack([points, points])

    # Make sure the fast and base algorithms match.
    simple_test = self.algo.is_pareto_optimal(points)
    fast_test = self.algo._base_algorithm.is_pareto_optimal(points)

    self.assertTrue(np.all(simple_test == fast_test))

  def test_is_pareto_optimal_against(self):
    points = np.array([[1, 2, 3], [2, 4, 1], [1, 2, -1]])
    dominating_points = np.array([[1, 2, 3], [3, 4, 0]])

    np.testing.assert_array_equal(
        self.algo.is_pareto_optimal_against(
            points, dominating_points, strict=True), [True, True, False])
    np.testing.assert_array_equal(
        self.algo.is_pareto_optimal_against(
            points, dominating_points, strict=False), [False, True, False])

  def test_is_pareto_optimal_against_randomized(self):
    dim = 4
    dominating_points = np.random.normal(size=(10000, dim))
    points = np.random.normal(size=(1000, dim))

    # Make sure the fast and base algorithms match.
    simple_check = self.algo.is_pareto_optimal_against(
        points, dominating_points, strict=True)
    fast_check = self.algo._base_algorithm.is_pareto_optimal_against(
        points, dominating_points, strict=True)

    # Results should not be affected by changing recursive threshold.
    self.assertTrue(np.all(simple_check == fast_check))

  def test_update_pareto(self):
    points = np.array([[1, 2, 3], [1, 2, 3], [2, 4, 1], [1, 2, -1]])
    np.testing.assert_array_equal(
        self.algo.is_pareto_optimal(points), [True, True, True, False]
    )

    # Adding a point that dominates the first two values.
    point = np.array([[2, 3, 4]])
    np.testing.assert_array_equal(
        self.algo.update_pareto_optimal(points, point),
        [2, 4],
    )


if __name__ == '__main__':
  absltest.main()
