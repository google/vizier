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

"""Tests for output_warpers, including NaN and infeasible trial handling."""

from jax import numpy as jnp
import numpy as np
import scipy
from tensorflow_probability.substrates import jax as tfp
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.gp import output_warpers
from vizier.pyvizier import converters
from vizier.pyvizier.converters import padding
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
    return np.array([[1.0], [1.0], [5.0], [-1e80], [np.nan], [-np.inf]])

  def test_always_maps_to_finite(self):
    if not self.always_maps_to_finite:
      self.skipTest('This class does not map every value to a finite value.')

    labels = np.array([[1.0], [1.0], [5.0], [-1e80], [np.nan], [-np.inf]])
    labels_warped = self.warper.warp(labels)
    np.testing.assert_allclose(
        np.isfinite(labels_warped), True, err_msg=f'warped: {labels_warped}'
    )

  def test_input_is_not_mutated(self):
    labels_input = np.array([[1.0], [1.0], [5.0], [10.0]])
    _ = self.warper.warp(labels_input)
    self.assertTrue(
        (
            labels_input.flatten()
            == np.array([[1.0], [1.0], [5.0], [10.0]]).flatten()
        ).all()
    )

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
        err_msg=f'Unwarped: {labels}\nWarped: {labels_warped}',
    )

  def test_preserve_rank_if_no_outliers(self):
    labels = np.array([[1.0], [1.0], [5.0], [-1], [-4], [np.nan], [np.nan]])
    finite_indices = np.isfinite(labels)
    labels_warped = self.warper.warp(labels)
    np.testing.assert_array_equal(
        scipy.stats.rankdata(labels[finite_indices]),
        scipy.stats.rankdata(labels_warped[finite_indices]),
        err_msg=f'Unwarped: {labels}\nWarped: {labels_warped}',
    )

  def test_finite_maps_to_finite(self):
    labels = self.labels_with_outliers()
    finite_indices = np.isfinite(labels)
    labels_warped = self.warper.warp(labels)
    np.testing.assert_allclose(
        np.isfinite(labels_warped[finite_indices]),
        True,
        err_msg=f'warped: {labels_warped}',
    )


class DefaultOutputWarperTest(_OutputWarperTestCase, parameterized.TestCase):

  @property
  def warper(self) -> OutputWarper:
    return output_warpers.create_default_warper()

  @property
  def always_maps_to_finite(self) -> bool:
    return True

  def test_unwarp_duplicate_labels(self):
    warper = self.warper
    _ = warper.warp(np.array([[1.0], [1.0], [5.0], [-1e80]]))
    labels = np.array([[1.0], [15.0], [10.0], [1.0]])
    np.testing.assert_array_equal(
        scipy.stats.rankdata(warper.unwarp(labels).flatten(), method='dense'),
        scipy.stats.rankdata(labels.flatten(), method='dense'),
    )

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
          unwarped=np.array(
              [[1.0], [1.0], [5.0], [-1e80], [np.nan], [-np.inf]]
          ),
          expected=np.array([
              [0.61848423],
              [0.61848423],
              [1.25966537],
              [0.25966537],
              [-1.24033463],
              [-1.24033463],
          ]),
      ),
      dict(
          testcase_name='case_all_NaNs',
          unwarped=np.array([[np.nan], [np.nan]]),
          expected=np.array([
              [-1.0],
              [-1.0],
          ]),
      ),
  ])
  def test_known_arrays(self, unwarped: np.ndarray, expected: np.ndarray):
    actual = self.warper.warp(unwarped)
    np.testing.assert_allclose(actual, expected, err_msg=f'actual: {actual}')

  def test_default_warper_empty_warpers(self):
    with self.assertRaises(ValueError):
      output_warpers.create_default_warper(
          half_rank_warp=False, log_warp=False, infeasible_warp=False
      )

  def test_unwarp(self):
    warper = self.warper
    labels_arr = np.array(
        [[-100.0], [-200.0], [1.0], [2.0], [3.0], [10.0], [15.0]]
    )
    np.testing.assert_array_almost_equal(
        warper.unwarp(warper.warp(labels_arr)), labels_arr
    )


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

  def setUp(self):
    super().setUp()
    self.labels_arr = np.asarray([10.0, 15.0, 20.0])[:, np.newaxis]

  @property
  def warper(self) -> OutputWarper:
    return output_warpers.NormalizeLabels()

  def labels_with_outliers(self):
    # Uses a less extreme outlier (-1e10 instead of -1e80) because linear
    # warping from [-1e80, 5.0] to [0.0, 1.0] maps 1.0, 1.0, 5.0, -1e80 to
    # 1.0, 1.0, 1.0, 0.0 due to numerical precision issues and fails to preserve
    # rank.
    return np.array([[1.0], [1.0], [5.0], [-1e10], [np.nan], [-np.inf]])

  def test_known_arrays(self):
    actual = self.warper.warp(self.labels_arr)
    expected = np.asarray([0.0, 0.5, 1.0])[:, np.newaxis]
    np.testing.assert_allclose(
        actual, expected, err_msg=f'actual: {actual.tolist()}'
    )


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

  def setUp(self):
    super().setUp()
    self.labels_arr = np.asarray(
        [10.0, -100.0, -200.0, 4.0, 0.0, 1.0, 2.0, -200.0, 3.0, 10.0, 15.0]
    )[:, np.newaxis]

  @property
  def warper(self) -> output_warpers.HalfRankComponent:
    return output_warpers.HalfRankComponent()

  @parameterized.named_parameters([
      dict(
          testcase_name='case1',
          unwarped=np.array([[
              np.nan,
              1,
              4,
              2,
              10,
              12,
              -np.inf,
              2,
              3,
              5,
              6,
          ]]).T,
          expected=np.array([
              [np.nan],
              [-2.7145447657886415],
              [4.0],
              [0.3722561569665319],
              [10.0],
              [12.0],
              [np.nan],
              [0.3722561569665319],
              [2.322289907556879],
              [5.0],
              [6.0],
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

  def test_unwarp_shape(self):
    warper = self.warper
    _ = warper.warp(self.labels_arr)
    np.testing.assert_equal(
        warper.unwarp(self.labels_arr).shape, self.labels_arr.shape
    )

  def test_bijective_at_exact_points(self):
    warper = self.warper
    labels_arr_warped = warper.warp(self.labels_arr)
    np.testing.assert_array_almost_equal(
        self.labels_arr, warper.unwarp(labels_arr_warped)
    )

  def test_unwarp_preserve_rank_interpolate(self):
    """Tests rank preservation among points interpolated between the training labels."""
    warper = self.warper
    _ = warper.warp(self.labels_arr)
    labels_test_warped = np.array([
        [-4.05487106],
        [-9.20688355],
        [-0.80017519],
        [2.0],
        [3.0],
        [10.0],
        [15.0],
        [1.0],
        [11.0],
        [-2.0],
    ])
    labels_test = warper.unwarp(labels_test_warped)
    np.testing.assert_array_almost_equal(
        np.argsort(labels_test, axis=0), np.argsort(labels_test_warped, axis=0)
    )

  def test_unwarp_preserve_rank_extrapolate(self):
    """Tests rank preservation among points extrapolated beyond the training labels."""
    warper = self.warper
    _ = warper.warp(self.labels_arr)
    labels_test_warped = np.array([
        [-4.05487106],
        [-9.20688355],
        [-0.80017519],
        [2.0],
        [3.0],
        [10.0],
        [15.0],
        [-10.0],
        [-20.0],
        [30.0],
        [50.0],
    ])
    labels_test = warper.unwarp(labels_test_warped)
    np.testing.assert_array_almost_equal(
        np.argsort(labels_test, axis=0), np.argsort(labels_test_warped, axis=0)
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
    labels = np.array(
        [[-200.0], [np.nan], [-1000.0], [np.nan], [1.0], [2.0], [3.0]]
    )

    labels_warped_infeasible = warper_infeasible.warp(labels)
    self.assertEqual(np.isnan(labels_warped_infeasible).sum(), 0)

  def test_all_nans(self):
    warper_infeasible = output_warpers.InfeasibleWarperComponent()
    labels = np.array([[np.nan], [np.nan], [np.nan], [np.nan]])
    labels_warped_infeasible = warper_infeasible.warp(labels)
    expected = np.array([[0.0], [0.0], [0.0], [0.0]])
    np.testing.assert_equal(labels_warped_infeasible, expected)

    unwarped = warper_infeasible.unwarp(labels_warped_infeasible)
    np.testing.assert_equal(unwarped, labels)

  def test_known_arrays(self):
    # TODO: Add a couple of parameterized test cases.
    self.skipTest('No test cases provided')


class BijectorWarperTest(absltest.TestCase):

  def test_trivial(self):
    warper = output_warpers.BijectorWarper(
        lambda arr: tfp.bijectors.Shift(arr.mean())
    )

    original = np.array([
        0.0,
        1.0,
        2.0,
    ])
    warped = original + 1
    np.testing.assert_allclose(warped, warper.warp(original))
    np.testing.assert_allclose(original, warper.unwarp(warped))


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


class LinearOutputWarperTest(parameterized.TestCase):
  """Tests for LinearOutputWarperTest."""

  @parameterized.parameters(
      {'low': -2.0, 'high': 2.0, 'dtype': 'numpy'},
      {'low': -5.0, 'high': 7.0, 'dtype': 'numpy'},
      {'low': 0.0, 'high': 1.0, 'dtype': 'jax'},
      {'low': 8.0, 'high': 10.0, 'dtype': 'jax'},
      {'low': -10.0, 'high': -2.0, 'dtype': 'jax'},
  )
  def test_warp_unwarp_and_range(self, low, high, dtype):
    num_samples = 50
    num_metrics = 3
    y = np.random.randn(num_samples, num_metrics)
    if dtype == 'jax':
      y = jnp.asarray(y, dtype=np.float64)
    output_warper = output_warpers.LinearOutputWarper.from_obs(
        y_obs=y, low_bound=low, high_bound=high
    )
    np.testing.assert_allclose(
        output_warper.unwarp(output_warper.warp(y)),
        y,
        atol=1e-05,
    )
    np.testing.assert_allclose(
        np.min(output_warper.warp(y), axis=0),
        np.array([low] * num_metrics),
        atol=1e-05,
    )
    np.testing.assert_allclose(
        np.max(output_warper.warp(y), axis=0),
        np.array([high] * num_metrics),
        atol=1e-05,
    )

  @parameterized.parameters(
      {'dtype': 'numpy', 'label_value': 0.0},
      {'dtype': 'jax', 'label_value': 123.0},
      {'dtype': 'jax', 'label_value': -54321.0},
  )
  def test_warp_unwarp_constant_labels(self, dtype: str, label_value: float):
    y = np.ones(shape=(10, 3)) * label_value
    if dtype == 'jax':
      y = jnp.asarray(y, dtype=np.float64)
    output_warper = output_warpers.LinearOutputWarper.from_obs(
        y_obs=y, low_bound=-2.0, high_bound=2.0
    )
    warped_y = output_warper.warp(y)
    np.testing.assert_allclose(
        warped_y,
        -2.0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        output_warper.unwarp(warped_y),
        y,
        atol=1e-5,
    )


class TrialOutputWarperTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.problem = vz.ProblemStatement()
    self.problem.search_space.root.add_float_param('x', 0.0, 10.0)
    self.problem.metric_information.append(
        vz.MetricInformation(name='y', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    )
    self.converter = converters.TrialToModelInputConverter.from_problem(
        self.problem
    )
    self.warper = output_warpers.TrialOutputWarper(self.converter)

  def _make_warper(
      self, goal: vz.ObjectiveMetricGoal
  ) -> output_warpers.TrialOutputWarper:
    """Returns a warper for a study with a single metric 'y' with `goal`."""
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('x', 0.0, 10.0)
    problem.metric_information.append(vz.MetricInformation(name='y', goal=goal))
    return output_warpers.TrialOutputWarper(
        converters.TrialToModelInputConverter.from_problem(problem)
    )

  def _finite_trials(self) -> list[vz.Trial]:
    """Returns three feasible trials with finite metric values."""
    trials = []
    for i, value in enumerate([10.0, 20.0, 30.0]):
      trial = vz.Trial(parameters={'x': float(i)})
      trial.complete(vz.Measurement(metrics={'y': value}))
      trials.append(trial)
    return trials

  def test_feasible_with_nan_dropped(self):
    # Trial 1: Feasible, y=10.0
    t1 = vz.Trial(parameters={'x': 1.0})
    t1.complete(vz.Measurement(metrics={'y': 10.0}))

    # Trial 2: Feasible, y=NaN (should be dropped)
    t2 = vz.Trial(parameters={'x': 2.0})
    t2.complete(vz.Measurement(metrics={'y': np.nan}))

    # Trial 3: Feasible, no measurement (should be dropped)
    t3 = vz.Trial(parameters={'x': 3.0})

    # Trial 4: Feasible, y=20.0
    t4 = vz.Trial(parameters={'x': 4.0})
    t4.complete(vz.Measurement(metrics={'y': 20.0}))

    model_data = self.warper.warp_trials([t1, t2, t3, t4])
    self.assertLen(self.warper._output_warpers, 1)
    self.assertIn('y', self.warper._output_warpers)
    # Features and labels should only have 2 rows (t1 and t4).
    self.assertEqual(model_data.features.continuous.unpad().shape[0], 2)
    self.assertEqual(model_data.labels.unpad().shape[0], 2)
    self.assertTrue(np.all(np.isfinite(model_data.labels.unpad())))

  def test_infeasible_trial_warped_to_bad_value(self):
    # Trial 1: Feasible, y=10.0
    t1 = vz.Trial(parameters={'x': 1.0})
    t1.complete(vz.Measurement(metrics={'y': 10.0}))

    # Trial 2: Feasible, y=20.0
    t2 = vz.Trial(parameters={'x': 2.0})
    t2.complete(vz.Measurement(metrics={'y': 20.0}))

    # Trial 3: Infeasible trial (no measurement)
    t3 = vz.Trial(parameters={'x': 3.0})
    t3.complete(vz.Measurement(), infeasibility_reason='evaluation failed')

    # Trial 4: Infeasible trial with NaN measurement
    t4 = vz.Trial(parameters={'x': 4.0})
    t4.complete(
        vz.Measurement(metrics={'y': np.nan}), infeasibility_reason='crashed'
    )

    model_data = self.warper.warp_trials([t1, t2, t3, t4])
    # All 4 trials should be retained.
    self.assertEqual(model_data.features.continuous.unpad().shape[0], 4)
    labels = np.asarray(model_data.labels.unpad())
    self.assertEqual(labels.shape[0], 4)
    self.assertTrue(np.all(np.isfinite(labels)))
    # Infeasible trials should be warped to worse values than feasible trials.
    self.assertLess(labels[2, 0], labels[0, 0])
    self.assertLess(labels[3, 0], labels[0, 0])

  def test_zero_mean_property_with_dropped_nans(self):
    # Two feasible trials: y=10.0, y=20.0
    t1 = vz.Trial(parameters={'x': 1.0})
    t1.complete(vz.Measurement(metrics={'y': 10.0}))
    t2 = vz.Trial(parameters={'x': 2.0})
    t2.complete(vz.Measurement(metrics={'y': 20.0}))

    # Baseline with only t1 and t2
    data_clean = self.warper.warp_trials([t1, t2])

    # With missing NaN trials added (should produce identical warped labels)
    t_nan = vz.Trial(parameters={'x': 5.0})
    t_nan.complete(vz.Measurement(metrics={'y': np.nan}))
    data_with_nans = self.warper.warp_trials([t1, t_nan, t2])

    np.testing.assert_allclose(
        data_clean.labels.unpad(), data_with_nans.labels.unpad()
    )

  def test_multimetric_partial_nan_dropped(self):
    multi_problem = vz.ProblemStatement()
    multi_problem.search_space.root.add_float_param('x', 0.0, 10.0)
    multi_problem.metric_information.append(
        vz.MetricInformation(name='y1', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    )
    multi_problem.metric_information.append(
        vz.MetricInformation(name='y2', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    )
    multi_converter = converters.TrialToModelInputConverter.from_problem(
        multi_problem
    )
    multi_warper = output_warpers.TrialOutputWarper(multi_converter)

    # Trial 1: Both metrics present
    t1 = vz.Trial(parameters={'x': 1.0})
    t1.complete(vz.Measurement(metrics={'y1': 10.0, 'y2': 20.0}))

    # Trial 2: y1 present, y2 is NaN (not infeasible -> should be dropped)
    t2 = vz.Trial(parameters={'x': 2.0})
    t2.complete(vz.Measurement(metrics={'y1': 15.0, 'y2': np.nan}))

    # Trial 3: Infeasible (y1 present, y2 is NaN -> should be kept and warped)
    t3 = vz.Trial(parameters={'x': 3.0})
    t3.complete(
        vz.Measurement(metrics={'y1': 5.0, 'y2': np.nan}),
        infeasibility_reason='partial crash',
    )

    model_data = multi_warper.warp_trials([t1, t2, t3])
    # t2 is dropped, so only t1 and t3 remain (2 trials)
    self.assertEqual(model_data.features.continuous.unpad().shape[0], 2)
    self.assertEqual(model_data.labels.unpad().shape[0], 2)
    self.assertTrue(np.all(np.isfinite(model_data.labels.unpad())))
    self.assertLen(multi_warper._output_warpers, 2)
    self.assertIn('y1', multi_warper._output_warpers)
    self.assertIn('y2', multi_warper._output_warpers)

  def test_all_missing_returns_empty(self):
    t_nan1 = vz.Trial(parameters={'x': 1.0})
    t_nan1.complete(vz.Measurement(metrics={'y': np.nan}))
    t_nan2 = vz.Trial(parameters={'x': 2.0})

    model_data = self.warper.warp_trials([t_nan1, t_nan2])
    self.assertEqual(model_data.features.continuous.unpad().shape[0], 0)
    self.assertEqual(model_data.labels.unpad().shape[0], 0)
    self.assertEmpty(self.warper._output_warpers)

  def test_empty_trials_with_padding_schedule(self):
    padding_converter = converters.TrialToModelInputConverter.from_problem(
        self.problem,
        padding_schedule=padding.PaddingSchedule(
            num_trials=padding.PaddingType.MULTIPLES_OF_10,  # pyrefly: ignore[unexpected-keyword]
        ),
    )
    padding_warper = output_warpers.TrialOutputWarper(padding_converter)
    model_data = padding_warper.warp_trials([])
    self.assertEqual(model_data.features.continuous.unpad().shape[0], 0)
    self.assertEqual(model_data.labels.unpad().shape[0], 0)
    # Padded shapes must be synchronized.
    self.assertEqual(
        model_data.features.continuous.shape[0],
        model_data.labels.shape[0],
    )
    self.assertEqual(model_data.features.continuous.shape[0], 0)
    self.assertEmpty(padding_warper._output_warpers)

  def test_unwarp_without_warp_raises_value_error(self):
    samples = np.array([[1.0, 2.0], [3.0, 4.0]])
    with self.assertRaises(ValueError):
      self.warper.unwarp(samples)

  def test_unwarp_invalid_ndim_raises_value_error(self):
    t = vz.Trial(parameters={'x': 1.0})
    t.complete(vz.Measurement(metrics={'y': 10.0}))
    self.warper.warp_trials([t])

    # 1D array
    with self.assertRaises(ValueError):
      self.warper.unwarp(np.array([1.0, 2.0]))

    # 4D array
    with self.assertRaises(ValueError):
      self.warper.unwarp(np.ones((2, 2, 2, 2)))

  def test_unwarp_single_metric_roundtrip(self):
    trials = []
    for i in range(10):
      t = vz.Trial(parameters={'x': float(i)})
      t.complete(vz.Measurement(metrics={'y': float(i) * 5.0}))
      trials.append(t)

    model_data = self.warper.warp_trials(trials)
    warped_labels = np.asarray(model_data.labels.unpad())  # (10, 1)

    # Fake samples of shape (num_samples=2, num_trials=10)
    samples_2d = np.tile(warped_labels.T, (2, 1))  # (2, 10)
    unwarped_2d = self.warper.unwarp(samples_2d)
    self.assertEqual(unwarped_2d.shape, (2, 10))

    # Also test 3D input of shape (num_samples=2, num_trials=10, num_metrics=1)
    samples_3d = np.expand_dims(samples_2d, axis=-1)
    unwarped_3d = self.warper.unwarp(samples_3d)
    # Squeezed to 2D for single metric
    self.assertEqual(unwarped_3d.shape, (2, 10))

  def test_unwarp_multi_metric(self):
    multi_problem = vz.ProblemStatement()
    multi_problem.search_space.root.add_float_param('x', 0.0, 10.0)
    multi_problem.metric_information.append(
        vz.MetricInformation(name='y1', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    )
    multi_problem.metric_information.append(
        vz.MetricInformation(name='y2', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    )
    multi_converter = converters.TrialToModelInputConverter.from_problem(
        multi_problem
    )
    multi_warper = output_warpers.TrialOutputWarper(multi_converter)

    trials = []
    for i in range(10):
      t = vz.Trial(parameters={'x': float(i)})
      t.complete(
          vz.Measurement(metrics={'y1': float(i), 'y2': float(i) * 1000.0})
      )
      trials.append(t)

    model_data = multi_warper.warp_trials(trials)
    warped_labels = np.asarray(model_data.labels.unpad())  # (10, 2)

    # Samples of shape (num_samples=3, num_trials=10, num_metrics=2)
    samples_3d = np.tile(
        np.expand_dims(warped_labels, axis=0), (3, 1, 1)
    )  # (3, 10, 2)
    unwarped_3d = multi_warper.unwarp(samples_3d)
    self.assertEqual(unwarped_3d.shape, (3, 10, 2))
    # y2 scale should be roughly 1000x of y1 scale
    self.assertGreater(unwarped_3d[0, -1, 1], unwarped_3d[0, -1, 0] * 100)

  # --- Goal-aware handling of non-finite metric values. ----------------------
  #
  # Every metric value falls into exactly one of five cases:
  #   A. Finite: used as-is.
  #   B. Infinite in the metric's GOOD direction (+inf for MAXIMIZE, -inf for
  #      MINIMIZE): rejected by the Vizier API with INVALID_ARGUMENT, so only
  #      stale stored data can carry one. The trial is dropped -- and crucially
  #      no exception escapes, since `_validate_labels` would raise on +inf and
  #      take the whole designer down.
  #   C. Infinite in the metric's BAD direction (-inf for MAXIMIZE, +inf for
  #      MINIMIZE): accepted by the API; it means "infinitely bad". The trial is
  #      KEPT and warped to a finite value worse than every finite value.
  #   D. NaN on an infeasible trial: treated exactly like case C.
  #   E. NaN on a feasible trial: missing data. The whole trial is dropped,
  #      because the GP requires complete rows.

  @parameterized.named_parameters(
      dict(
          testcase_name='maximize',
          goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          infinitely_bad=-np.inf,
      ),
      dict(
          testcase_name='minimize',
          goal=vz.ObjectiveMetricGoal.MINIMIZE,
          infinitely_bad=np.inf,
      ),
  )
  def test_infinitely_bad_metric_is_identical_to_infeasible_nan(
      self, goal: vz.ObjectiveMetricGoal, infinitely_bad: float
  ):
    """Cases C and D must be completely indistinguishable."""

    def make_trials(bad_value_spelling: str) -> list[vz.Trial]:
      trials = self._finite_trials()
      t_bad = vz.Trial(parameters={'x': 9.0})
      if bad_value_spelling == 'infinity':
        # Case C: FEASIBLE trial whose metric is infinitely bad.
        t_bad.complete(vz.Measurement(metrics={'y': infinitely_bad}))
      else:
        # Case D: INFEASIBLE trial whose metric is NaN.
        t_bad.complete(
            vz.Measurement(metrics={'y': np.nan}),
            infeasibility_reason='crashed',
        )
      trials.append(t_bad)
      return trials

    labels_infinity = np.asarray(
        self._make_warper(goal)
        .warp_trials(make_trials('infinity'))
        .labels.unpad()
    )
    labels_nan = np.asarray(
        self._make_warper(goal)
        .warp_trials(make_trials('infeasible_nan'))
        .labels.unpad()
    )

    # Both spellings must keep all four trials. Asserting the shape first means
    # the equality below cannot be satisfied trivially by dropping both.
    self.assertEqual(labels_infinity.shape, (4, 1))
    self.assertEqual(labels_nan.shape, (4, 1))
    np.testing.assert_array_equal(labels_infinity, labels_nan)
    # And the shared value really is a penalty, not just an arbitrary match.
    self.assertTrue(np.all(np.isfinite(labels_infinity)))
    self.assertLess(labels_infinity[3, 0], np.min(labels_infinity[:3, 0]))

  @parameterized.named_parameters(
      dict(
          testcase_name='maximize',
          goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          bad_infinity=-np.inf,
      ),
      dict(
          testcase_name='minimize',
          goal=vz.ObjectiveMetricGoal.MINIMIZE,
          bad_infinity=np.inf,
      ),
  )
  def test_infinitely_bad_metric_is_kept_and_penalized(
      self, goal: vz.ObjectiveMetricGoal, bad_infinity: float
  ):
    """Case C: infinity in the metric's BAD direction is kept and penalized."""
    trials = self._finite_trials()
    t_bad = vz.Trial(parameters={'x': 9.0})
    t_bad.complete(vz.Measurement(metrics={'y': bad_infinity}))
    trials.append(t_bad)

    model_data = self._make_warper(goal).warp_trials(trials)
    labels = np.asarray(model_data.labels.unpad())
    # Dropped by the previous pre-filter, and also dropped by any
    # implementation that assumes +inf is always infinitely good.
    self.assertEqual(model_data.features.continuous.unpad().shape[0], 4)
    self.assertEqual(labels.shape, (4, 1))
    self.assertTrue(np.all(np.isfinite(labels)))
    self.assertLess(labels[3, 0], np.min(labels[:3, 0]))
    if goal == vz.ObjectiveMetricGoal.MINIMIZE:
      # The converter flips the sign of MINIMIZE metrics, so labels follow the
      # maximization convention: the smallest raw value has the largest label.
      self.assertGreater(labels[0, 0], labels[2, 0])

  @parameterized.named_parameters(
      dict(
          testcase_name='maximize',
          goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          good_infinity=np.inf,
      ),
      dict(
          testcase_name='minimize',
          goal=vz.ObjectiveMetricGoal.MINIMIZE,
          good_infinity=-np.inf,
      ),
  )
  def test_infinitely_good_metric_is_dropped_without_raising(
      self, goal: vz.ObjectiveMetricGoal, good_infinity: float
  ):
    """Case B: infinity in the GOOD direction is dropped without raising in `_validate_labels`."""
    finite_trials = self._finite_trials()
    t_good = vz.Trial(parameters={'x': 9.0})
    t_good.complete(vz.Measurement(metrics={'y': good_infinity}))

    baseline = self._make_warper(goal).warp_trials(finite_trials)
    # Must not raise: +inf in label space reaching `_validate_labels` raises
    # ValueError and would take the whole designer down.
    model_data = self._make_warper(goal).warp_trials(finite_trials + [t_good])

    self.assertEqual(model_data.features.continuous.unpad().shape[0], 3)
    # The surviving trials are warped exactly as if the bad trial never existed.
    np.testing.assert_allclose(
        np.asarray(model_data.labels.unpad()),
        np.asarray(baseline.labels.unpad()),
    )

  def test_nan_on_feasible_trial_drops_whole_trial(self):
    """Case E: a NaN on a feasible trial is missing data, not infeasibility."""
    finite_trials = self._finite_trials()
    t_nan = vz.Trial(parameters={'x': 9.0})
    t_nan.complete(vz.Measurement(metrics={'y': np.nan}))

    baseline = self._make_warper(vz.ObjectiveMetricGoal.MAXIMIZE).warp_trials(
        finite_trials
    )
    model_data = self._make_warper(vz.ObjectiveMetricGoal.MAXIMIZE).warp_trials(
        finite_trials + [t_nan]
    )

    self.assertEqual(model_data.features.continuous.unpad().shape[0], 3)
    # Missing data must not perturb the warping of the remaining trials.
    np.testing.assert_allclose(
        np.asarray(model_data.labels.unpad()),
        np.asarray(baseline.labels.unpad()),
    )

  def test_nan_on_infeasible_trial_is_kept_and_penalized(self):
    """Case D: a NaN on an infeasible trial means 'infinitely bad'."""
    trials = self._finite_trials()
    t_infeasible = vz.Trial(parameters={'x': 9.0})
    t_infeasible.complete(
        vz.Measurement(metrics={'y': np.nan}), infeasibility_reason='crashed'
    )
    trials.append(t_infeasible)

    model_data = self._make_warper(vz.ObjectiveMetricGoal.MAXIMIZE).warp_trials(
        trials
    )
    labels = np.asarray(model_data.labels.unpad())
    self.assertEqual(model_data.features.continuous.unpad().shape[0], 4)
    self.assertEqual(labels.shape, (4, 1))
    self.assertTrue(np.all(np.isfinite(labels)))
    self.assertLess(labels[3, 0], np.min(labels[:3, 0]))

  def test_infeasible_trial_without_final_measurement_is_kept(self):
    """Case D with no measurement at all: `final_measurement` is None.

    `warp_trials` feeds such a trial to `converter.to_labels`, which maps a
    missing measurement to NaN rather than raising, so the trial survives the
    filter and is penalized like any other infeasible trial.
    """
    trials = self._finite_trials()
    # Never completed with a measurement, only marked infeasible.
    t_infeasible = vz.Trial(
        parameters={'x': 9.0},
        infeasibility_reason='never ran',  # pyrefly: ignore[unexpected-keyword]
    )
    trials.append(t_infeasible)
    self.assertTrue(t_infeasible.infeasible)
    self.assertIsNone(t_infeasible.final_measurement)

    model_data = self._make_warper(vz.ObjectiveMetricGoal.MAXIMIZE).warp_trials(
        trials
    )
    labels = np.asarray(model_data.labels.unpad())
    self.assertEqual(model_data.features.continuous.unpad().shape[0], 4)
    self.assertEqual(labels.shape, (4, 1))
    self.assertTrue(np.all(np.isfinite(labels)))
    self.assertLess(labels[3, 0], np.min(labels[:3, 0]))

  def test_infinitely_good_metric_on_infeasible_trial_is_dropped(self):
    """An infeasible trial can still carry a stale infinitely-good value.

    The `+inf` guard must be checked BEFORE the `trial.infeasible`
    short-circuit, otherwise such a trial is admitted to training and the
    `+inf` reaches `_validate_labels`, which raises.
    """
    finite_trials = self._finite_trials()
    t_bad = vz.Trial(parameters={'x': 9.0})
    t_bad.complete(
        vz.Measurement(metrics={'y': np.inf}), infeasibility_reason='crashed'
    )
    self.assertTrue(t_bad.infeasible)

    baseline = self._make_warper(vz.ObjectiveMetricGoal.MAXIMIZE).warp_trials(
        finite_trials
    )
    # Must not raise ValueError('Infinity metric value is not valid.').
    model_data = self._make_warper(vz.ObjectiveMetricGoal.MAXIMIZE).warp_trials(
        finite_trials + [t_bad]
    )

    self.assertEqual(model_data.features.continuous.unpad().shape[0], 3)
    np.testing.assert_allclose(
        np.asarray(model_data.labels.unpad()),
        np.asarray(baseline.labels.unpad()),
    )

  def test_filter_stays_aligned_under_padding(self):
    """Labels must stay zipped to the right trials when padding is active.

    The filter classifies trials by walking `to_labels(...).unpad()` alongside
    the candidate list. `unpad()` slices back to the pre-padding shape and
    padding only ever appends rows, so row i must still belong to candidate i
    even when the padding schedule rounds the trial count up.
    """
    padding_converter = converters.TrialToModelInputConverter.from_problem(
        self.problem,
        padding_schedule=padding.PaddingSchedule(
            num_trials=padding.PaddingType.MULTIPLES_OF_10,  # pyrefly: ignore[unexpected-keyword]
        ),
    )
    warper = output_warpers.TrialOutputWarper(padding_converter)

    # 12 trials -> padded up to 20, so unpad() must undo a 8-row pad.
    # x=3 is infinitely good (dropped) and x=7 is missing data (dropped).
    trials = []
    for i in range(12):
      trial = vz.Trial(parameters={'x': float(i)})
      if i == 3:
        trial.complete(vz.Measurement(metrics={'y': np.inf}))
      elif i == 7:
        trial.complete(vz.Measurement(metrics={'y': np.nan}))
      else:
        trial.complete(vz.Measurement(metrics={'y': float(i)}))
      trials.append(trial)

    model_data = warper.warp_trials(trials)
    features = np.asarray(model_data.features.continuous.unpad())
    self.assertEqual(features.shape[0], 10)
    self.assertEqual(model_data.labels.unpad().shape[0], 10)
    # Parameters are scaled to [0, 1] over the range [0.0, 10.0], so the
    # surviving feature column pins exactly WHICH trials were kept. If the
    # filter mis-attributed a label row to a neighbouring trial, the wrong x
    # would survive here.
    expected_x = np.array([i / 10.0 for i in range(12) if i not in (3, 7)])
    np.testing.assert_allclose(features[:, 0], expected_x, atol=1e-6)
    # When all trials are dropped, padded feature and label row counts match.
    empty_data = warper.warp_trials([trials[3], trials[7]])
    self.assertEqual(
        empty_data.features.continuous.padded_array.shape[0],
        empty_data.labels.padded_array.shape[0],
    )


if __name__ == '__main__':
  absltest.main()
