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

"""Tests for outputwarpers."""

import numpy as np
import scipy
from vizier._src.algorithms.designers.gp import output_warpers
from absl.testing import absltest
from absl.testing import parameterized

OutputWarper = output_warpers.OutputWarper


@absltest.skipThisClass('Base class')
class _OutputWarperTestCase(absltest.TestCase):

  @property
  def warper(self) -> OutputWarper:
    raise RuntimeError('Subclasses should override this method!')

  @property
  def always_maps_to_finite(self) -> bool:
    # Override it to True if the warper should map every value to a
    # finite value.
    return False

  def labels_with_outliers(self):
    return np.array([[1.], [1.], [5.], [-1e80], [np.nan], [-np.inf]])

  def test_always_maps_to_finite(self):
    if not self.always_maps_to_finite:
      self.skipTest('This class does not map every value to a finite value.')

    labels = np.array([[1.], [1.], [5.], [-1e80], [np.nan], [-np.inf]])
    labels_warped = self.warper.warp(labels)
    np.testing.assert_allclose(
        np.isfinite(labels_warped), True, err_msg=f'warped: {labels_warped}')

  def test_input_is_not_mutated(self):
    labels_input = np.array([[1.], [1.], [5.], [10.]])
    _ = self.warper.warp(labels_input)
    self.assertTrue(
        (labels_input.flatten() == np.array([[1.], [1.], [5.],
                                             [10.]]).flatten()).all())

  def test_shape_is_preserved(self):
    labels = self.labels_with_outliers()
    labels_warped = self.warper.warp(labels)
    self.assertEqual(labels_warped.shape, labels.shape)

  def test_preserve_rank_despite_outliers(self):
    labels = self.labels_with_outliers()
    finite_indices = np.isfinite(labels)
    labels_warped = self.warper.warp(labels)
    np.testing.assert_array_equal(
        scipy.stats.rankdata(labels[finite_indices]),
        scipy.stats.rankdata(labels_warped[finite_indices]),
        err_msg=(f'Unwarped: {labels}\n'
                 f'Warped: {labels_warped}'))

  def test_preserve_rank_if_no_outliers(self):
    labels = np.array([[1.0], [1.0], [5.0], [-1], [-4], [np.nan], [np.nan]])
    finite_indices = np.isfinite(labels)
    labels_warped = self.warper.warp(labels)
    np.testing.assert_array_equal(
        scipy.stats.rankdata(labels[finite_indices]),
        scipy.stats.rankdata(labels_warped[finite_indices]),
        err_msg=(f'Unwarped: {labels}\n'
                 f'Warped: {labels_warped}'))

  def test_finite_maps_to_finite(self):
    labels = self.labels_with_outliers()
    finite_indices = np.isfinite(labels)
    labels_warped = self.warper.warp(labels)
    np.testing.assert_allclose(
        np.isfinite(labels_warped[finite_indices]),
        True,
        err_msg=f'warped: {labels_warped}')


class DefaultOutputWarperTest(_OutputWarperTestCase, parameterized.TestCase):

  @property
  def warper(self) -> OutputWarper:
    return output_warpers.create_default_warper(infeasible_warp=True)

  @property
  def always_maps_to_finite(self) -> bool:
    return True

  @parameterized.parameters([
      dict(labels=np.zeros(shape=(5, 1))),
      dict(labels=np.ones(shape=(5, 1))),
      dict(labels=100 * np.ones(shape=(5, 1))),
      dict(labels=-100 * np.ones(shape=(5, 1))),
  ])
  def test_all_identical_values_map_to_zero(self, labels):
    np.testing.assert_array_equal(self.warper.warp(labels), 0.0)

  @parameterized.named_parameters([
      dict(
          testcase_name='case1',
          unwarped=np.array([[1.], [1.], [5.], [-1e80], [np.nan], [-np.inf]]),
          expected=np.array([[-0.14118114], [-0.14118114], [0.5], [-0.5], [-2.],
                             [-2.]]),
      ),
  ])
  def test_known_arrays(self, unwarped: np.ndarray, expected: np.ndarray):
    actual = self.warper.warp(unwarped)
    np.testing.assert_allclose(actual, expected, err_msg=f'actual: {actual}')

  def test_default_warper_empty_warpers(self):
    with self.assertRaises(ValueError):
      output_warpers.create_default_warper(
          half_rank_warp=False, log_warp=False, infeasible_warp=False)


class ZScoreLabelsTest(_OutputWarperTestCase):

  @property
  def warper(self) -> OutputWarper:
    return output_warpers.ZScoreLabels()

  def test_preserve_rank_despite_outliers(self):
    # TODO: Fix this test, or add an explanation why this can be skipped.
    pass

  def test_known_arrays(self):
    # TODO: Add a couple of parameterized test cases.
    self.skipTest('No test cases provided')


class NormalizeLabelsTest(_OutputWarperTestCase):

  @property
  def warper(self) -> OutputWarper:
    return output_warpers.NormalizeLabels()

  def test_known_arrays(self):
    # TODO: Add a couple of parameterized test cases.
    self.skipTest('No test cases provided')


class DetectOutliersTest(_OutputWarperTestCase):

  @property
  def warper(self) -> OutputWarper:
    return output_warpers.create_warp_outliers_warper()

  # TODO: Add extra test coverage for the warp_outliers_warper.

  @property
  def always_maps_to_finite(self) -> bool:
    return True


class TransformToGaussianTest(_OutputWarperTestCase):

  @property
  def warper(self) -> OutputWarper:
    return output_warpers.TransformToGaussian()

  def test_finite_maps_to_finite(self):
    # TODO: Fix this test, or add an explanation why this can be skipped.
    pass

  def test_preserve_rank_if_no_outliers(self):
    # TODO: Fix this test, or add an explanation why this can be skipped.
    pass

  def test_preserve_rank_despite_outliers(self):
    # TODO: Fix this test, or add an explanation why this can be skipped.
    pass


class HalfRankComponentTest(_OutputWarperTestCase, parameterized.TestCase):

  @property
  def warper(self) -> output_warpers.HalfRankComponent:
    return output_warpers.HalfRankComponent()

  @parameterized.named_parameters([
      dict(
          testcase_name='case1',
          unwarped=np.array([[np.nan, 1, 2, 3, 4, 5, 6, 2, 10, 12, -np.inf]]).T,
          expected=np.array([
              [np.nan],
              [-2.7145447657886415],
              [0.3722561569665319],
              [2.322289907556879],
              [4.0],
              [5.0],
              [6.0],
              [0.3722561569665319],
              [10.0],
              [12.0],
              [np.nan],
          ]),
      ),
      dict(
          testcase_name='case2',
          unwarped=np.array([[np.nan, -4, -3, -2, 1.1, 1.2, 1.3, 1.4, 1.5]]).T,
          expected=np.array([
              [np.nan],
              [0.7984888240158797],
              [0.9467291870388195],
              [1.0380072549079085],
              [1.1139555940074284],
              [1.2],
              [1.3],
              [1.4],
              [1.5],
          ]),
      ),
      dict(
          testcase_name='case3',
          unwarped=np.array(
              [[np.nan, 1, 2, 3, 4, 4, 6, 7, 10, 11, 12]], dtype=np.float64
          ).T,
          expected=np.array([
              [np.nan],
              [-2.3573836671676096],
              [0.7453945664588675],
              [2.655910679724611],
              [4.2455644597926385],
              [4.2455644597926385],
              [6.0],
              [7.0],
              [10.0],
              [11.0],
              [12.0],
          ]),
      ),
  ])
  def test_known_arrays(self, unwarped: np.ndarray, expected: np.ndarray):
    actual = self.warper.warp(unwarped)
    np.testing.assert_allclose(
        actual, expected, err_msg=f'actual: {actual.tolist()}'
    )


class LogWarperComponentTest(_OutputWarperTestCase):

  def setUp(self):
    super().setUp()
    self.labels_arr = np.array(
        [[-100.0], [-200.0], [1.0], [2.0], [3.0], [10.0], [15.0]]
    )

  @property
  def warper(self) -> OutputWarper:
    return output_warpers.LogWarperComponent()

  def test_preserve_rank_despite_outliers(self):
    # TODO: Fix this test, or add an explanation why this can be skipped.
    pass

  def test_known_arrays(self):
    # TODO: Add a couple of parameterized test cases.
    self.skipTest('No test cases provided')

  def test_unwarp_shape(self):
    warper = self.warper
    _ = warper.warp(self.labels_arr)
    np.testing.assert_equal(
        warper.unwarp(self.labels_arr).shape, self.labels_arr.shape
    )

  def test_warp_shape(self):
    warper = self.warper
    _ = warper.warp(self.labels_arr)
    np.testing.assert_equal(
        warper.unwarp(self.labels_arr).shape, self.labels_arr.shape
    )

  def test_unwarp_values(self):
    warper = self.warper
    labels_arr_warped = warper.warp(self.labels_arr)
    expected = np.array([
        [-90.7415054],
        [-137.82329233],
        [-38.55794983],
        [-38.01309563],
        [-37.46762612],
        [-33.63193395],
        [-30.87322547],
    ])
    np.testing.assert_array_almost_equal(
        warper.unwarp(labels_arr_warped / 2), expected
    )

  def test_bijective_at_exact_points(self):
    warper = self.warper
    labels_arr_warped = warper.warp(self.labels_arr)
    np.testing.assert_array_almost_equal(
        self.labels_arr, warper.unwarp(labels_arr_warped)
    )

  def test_unwarp_preserve_rank_interpolate(self):
    """Tests rank preservation among points interpolated between the training labels.

    In details, the interplated array is labels_test_warped which includes
    mid-points between every two consective elements in the warped sorted array.
    We test wether the rank of warped labels augmented with the interpolated
    array in the warped domain is equal to the rank of unwarped labels and the
    unwarped interpolated array.
    """
    warper = self.warper
    labels_arr_warped = warper.warp(self.labels_arr)
    labels_test_warped = (
        np.sort(labels_arr_warped, axis=0)[0:-1]
        + np.sort(labels_arr_warped, axis=0)[1:]
    ) / 2
    labels_test = warper.unwarp(labels_test_warped)
    labels_all_warped = np.vstack((labels_arr_warped, labels_test_warped))
    labels_all = np.vstack((self.labels_arr, labels_test))
    np.testing.assert_array_almost_equal(
        np.argsort(labels_all_warped, axis=0),
        np.argsort(labels_all, axis=0),
    )

  def test_unwarp_preserve_rank_extrapolate(self):
    """Tests rank preservation among points extrapolated out of the range of the training labels."""

    warper = self.warper
    labels_arr_warped = warper.warp(self.labels_arr)
    labels_test_warped = np.array(
        [[np.min(labels_arr_warped) - 10.0], [np.max(labels_arr_warped) + 10.0]]
    )
    labels_test = warper.unwarp(labels_test_warped)
    labels_all_warped = np.vstack((labels_arr_warped, labels_test_warped))
    labels_all = np.vstack((self.labels_arr, labels_test))
    np.testing.assert_array_almost_equal(
        np.argsort(labels_all_warped, axis=0),
        np.argsort(labels_all, axis=0),
    )


class InfeasibleWarperTest(parameterized.TestCase):

  @property
  def always_maps_to_finite(self) -> bool:
    return True

  def test_warper_removes_nans(self):
    warper_infeasible = output_warpers.InfeasibleWarperComponent()
    labels = np.array([[-200.], [np.nan], [-1000.], [np.nan], [1.], [2.], [3.]])

    labels_warped_infeasible = warper_infeasible.warp(labels)
    self.assertEqual(np.isnan(labels_warped_infeasible).sum(), 0)

  def test_known_arrays(self):
    # TODO: Add a couple of parameterized test cases.
    self.skipTest('No test cases provided')


class OutputWarperPipelineTest(absltest.TestCase):
  """Tests the default outpur warper edge cases."""

  def test_all_nonfinite_labels(self):
    warper = output_warpers.OutputWarperPipeline()
    labels_infeaible = np.array([[-np.inf], [np.nan], [np.nan], [-np.inf]])
    self.assertTrue(
        (
            warper.warp(labels_infeaible)
            == -1 * np.ones(shape=labels_infeaible.shape).flatten()
        ).all()
    )


if __name__ == '__main__':
  absltest.main()
