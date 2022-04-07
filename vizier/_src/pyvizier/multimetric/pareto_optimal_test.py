"""Tests for pareto_optimal."""

import numpy as np

from vizier._src.pyvizier.multimetric import pareto_optimal
from absl.testing import absltest


class ParetoOptimalTest(absltest.TestCase):

  def setUp(self):
    super(ParetoOptimalTest, self).setUp()
    self.algo = pareto_optimal.FastParetoOptimalAlgorithm(
        is_pareto_optimal_base=pareto_optimal.is_pareto_optimal_naive,
        is_pareto_optimal_against_base=pareto_optimal
        .is_pareto_optimal_against_naive)

  def test_is_pareto_optimal(self):
    points = np.array([[1, 2, 3], [1, 2, 3], [2, 4, 1], [1, 2, -1]])
    self.assertCountEqual(
        self.algo.is_pareto_optimal(points), [True, True, True, False])

    # Adding a point that dominates the first two values.
    points = np.vstack([points, [2, 3, 4]])
    self.assertCountEqual(
        self.algo.is_pareto_optimal(points), [False, False, True, False, True])

  def test_is_pareto_optimal_all_optimal(self):
    # Generate points on positive orthant sphere.
    points = abs(np.random.normal(size=(1000, 3)))
    points /= np.linalg.norm(points, axis=1)[..., np.newaxis]
    self.assertTrue(
        np.all(self.algo.is_pareto_optimal(points, recursive_threshold=100)))

  def test_is_pareto_optimal_randomized(self):
    dim = 10
    points = np.random.normal(size=(1000, dim))
    # Make a copy of points to test equality case.
    points = np.vstack([points, points])

    simple_test = self.algo.is_pareto_optimal(points)
    fast_test = self.algo.is_pareto_optimal(points, recursive_threshold=100)

    self.assertTrue(np.all(simple_test == fast_test))

  def test_is_pareto_optimal_against(self):
    points = np.array([[1, 2, 3], [2, 4, 1], [1, 2, -1]])
    dominating_points = np.array([[1, 2, 3], [3, 4, 0]])

    self.assertCountEqual(
        self.algo.is_pareto_optimal_against(points, dominating_points),
        [True, True, False])
    self.assertCountEqual(
        self.algo.is_pareto_optimal_against(
            points, dominating_points, strict=False), [False, True, False])

  def test_is_pareto_optimal_against_randomized(self):
    dim = 4
    dominating_points = np.random.normal(size=(10000, dim))
    points = np.random.normal(size=(1000, dim))

    simple_check = self.algo.is_pareto_optimal_against(points,
                                                       dominating_points)
    fast_check = self.algo.is_pareto_optimal_against(
        points, dominating_points, recursive_threshold=100)

    # Results should not be affected by changing recursive threshold.
    self.assertTrue(np.all(simple_check == fast_check))


if __name__ == '__main__':
  absltest.main()
