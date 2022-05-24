"""Tests for xla_pareto."""

import numpy as np
from vizier._src.jax import xla_pareto
from vizier._src.pyvizier.multimetric import hypervolume

from absl.testing import absltest


class XlaParetoTest(absltest.TestCase):

  def test_get_pareto(self):
    ys = np.random.random([20000, 2])
    is_frontier = xla_pareto.is_frontier(ys, num_shards=5)
    frontier = xla_pareto.get_frontier(ys, num_shards=5)

    # Two implementations should be consistent.
    self.assertEqual(frontier.shape[0], np.sum(is_frontier))

    # Sort by one of the dimensions and the other dimension must be sorted
    # in the reverse order.
    sid = frontier[:, 0].argsort()

    np.testing.assert_array_equal(frontier[sid, 1].argsort(),
                                  np.arange(len(frontier)-1, -1, -1))

  def test_get_pareto_rank(self):
    ys = np.array([[6, 6], [4, 8], [2, 6], [3, 6], [1, 8], [7, 0], [4, 4],
                   [9, 6], [5, 10], [8, 10]])
    np.testing.assert_array_equal(
        xla_pareto.pareto_rank(ys), np.array([2, 2, 6, 5, 3, 2, 5, 0, 1, 0]))

  def test_cum_hypervolume_origin(self):
    dim = 3
    vectors = abs(
        np.array(np.random.uniform(size=(100, dim)), dtype=np.float32))
    points = abs(
        np.array(np.random.uniform(size=(1000, dim)), dtype=np.float32))

    np.testing.assert_array_almost_equal(
        hypervolume._cum_hypervolume_origin(points, vectors),
        xla_pareto.jax_cum_hypervolume_origin(points, vectors),
        decimal=2)


if __name__ == '__main__':
  absltest.main()
