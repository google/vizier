"""Tests for eagle strategy."""
import numpy as np
from vizier._src.algorithms.optimizers import eagle_strategy

from absl.testing import absltest


class VectorizedEagleStrategyTest(absltest.TestCase):

  def setUp(self):
    super(VectorizedEagleStrategyTest, self).setUp()
    self.config = eagle_strategy.EagleStrategyConfig(visibility=1, gravity=1)
    self.eagle = eagle_strategy.VectorizedEagleStrategy(
        config=self.config,
        n_features=2,
        pool_size=4,
        batch_size=2,
        low_bound=0.0,
        high_bound=1.0,
        seed=1)
    self.iterations = 2

  def test_compute_features_diffs_and_dists(self):
    self.eagle._features = np.array([[1, 2], [3, 4], [7, 7], [8, 8]])
    features_diffs, dists = self.eagle._compute_features_diffs_and_dists()
    expected_features_diffs = np.array([[[0, 0], [2, 2], [6, 5], [7, 6]],
                                        [[-2, -2], [0, 0], [4, 3], [5, 4]]])
    np.testing.assert_array_equal(
        features_diffs,
        expected_features_diffs,
        err_msg='feature differences mismatch')

    expected_dists = np.array([[0, 8, 61, 85], [8, 0, 25, 41]])
    np.testing.assert_array_almost_equal(
        dists, expected_dists, err_msg='feature distance mismatch')

  def test_compute_scaled_directions(self):
    self.eagle._metrics = np.array([2, 3, 4, 1])
    g = self.config.gravity
    ng = -self.config.negative_gravity
    expected_scaled_directions = np.array([
        [g, g, g, ng],
        [ng, g, g, ng],
    ])
    scaled_directions = self.eagle._compute_scaled_directions()
    np.testing.assert_array_equal(scaled_directions, expected_scaled_directions)

  def test_compute_scaled_directions_with_removed_flies(self):
    self.eagle._metrics = np.array([-np.inf, 3, -np.inf, 1])
    g = self.config.gravity
    ng = -self.config.negative_gravity
    # Note that -np.inf - (-np.inf) is not >=0.
    expected_scaled_directions = np.array([
        [ng, g, ng, g],
        [ng, g, ng, ng],
    ])
    scaled_directions = self.eagle._compute_scaled_directions()
    np.testing.assert_array_equal(scaled_directions, expected_scaled_directions)

  def test_compute_features_changes(self):
    self.eagle._metrics = np.array([-np.inf, 3, -np.inf, 1])
    features_diffs = np.array([[[0, 0], [2, 2], [6, 5], [7, 6]],
                               [[-2, -2], [0, 0], [4, 3], [5, 4]]])

    dists = np.array([[0, 8, 61, 85], [8, 0, 25, 41]])

    scaled_directions = np.array([
        [-1, 1, -1, 1],
        [-1, 1, -1, -1],
    ])
    features_changes = self.eagle._compute_features_changes(
        features_diffs, dists, scaled_directions)
    # scaled_pulls array:
    # [[-1.00000000e+000  4.24835426e-018 -3.46883002e-133  2.65977679e-185]
    # [-4.24835426e-018  1.00000000e+000 -5.16642063e-055 -9.32462145e-090]]
    c0 = np.array([2, 2]) * 4.24835426e-018 + np.array([7, 6]) * 2.65977679e-185
    c1 = np.array([5, 4]) * (-9.32462145e-090)
    expected_features_changes = np.vstack([c0, c1])
    self.assertEqual(features_changes.shape, (2, 2))
    np.testing.assert_array_almost_equal(expected_features_changes,
                                         features_changes)

  def test_create_features(self):
    self.assertEqual(self.eagle._create_features().shape, (2, 2))

  def test_create_perturbations(self):
    perturbations = self.eagle._create_perturbations()
    self.assertEqual(perturbations.shape, (2, 2))

  def test_update_pool_features_and_metrics(self):
    self.eagle._features = np.array([[1, 2], [3, 4], [7, 7], [8, 8]],
                                    dtype=np.float64)
    self.eagle._metrics = np.array([2, 3, 4, 1], dtype=np.float64)
    self.eagle._perturbations = np.array([1, 1, 1, 1], dtype=np.float64)

    self.eagle._last_suggested_features = np.array([[9, 9], [10, 10]],
                                                   dtype=np.float64)
    batch_metrics = np.array([5, 0.5], dtype=np.float64)

    self.eagle._update_pool_features_and_metrics(batch_metrics)
    np.testing.assert_array_equal(
        self.eagle._features,
        np.array([[9, 9], [3, 4], [7, 7], [8, 8]], dtype=np.float64),
        err_msg='Features are not equal.')

    np.testing.assert_array_equal(
        self.eagle._metrics,
        np.array([5, 3, 4, 1], dtype=np.float64),
        err_msg='Metrics are not equal.')

    pc = self.config.penalize_factor
    np.testing.assert_array_equal(
        self.eagle._perturbations,
        np.array([1, pc, 1, 1], dtype=np.float64),
        err_msg='Perturbations are not equal.')

  def test_trim_pool(self):
    pc = self.config.perturbation
    self.eagle._features = np.array([[1, 2], [3, 4], [7, 7], [8, 8]],
                                    dtype=np.float64)
    self.eagle._metrics = np.array([2, 3, 4, 1], dtype=np.float64)
    self.eagle._perturbations = np.array([pc, 0, 0, pc], dtype=np.float64)
    self.eagle._trim_pool()

    np.testing.assert_array_almost_equal(
        self.eagle._features[[0, 2, 3], :],
        np.array([[1, 2], [7, 7], [8, 8]], dtype=np.float64),
        err_msg='Features are not equal.')
    self.assertTrue(
        all(np.not_equal(self.eagle._features[1, :], np.array([3, 4]))),
        msg='Features are not equal.')

    np.testing.assert_array_equal(
        self.eagle._metrics,
        np.array([2, -np.inf, 4, 1], dtype=np.float64),
        err_msg='Metrics are not equal.')
    # The best firefly is never removed.
    np.testing.assert_array_equal(
        self.eagle._perturbations,
        np.array([pc, pc, 0, pc], dtype=np.float64),
        err_msg='Perturbations are not equal.')


if __name__ == '__main__':
  absltest.main()
