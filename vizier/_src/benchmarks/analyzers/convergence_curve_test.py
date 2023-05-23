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

"""Tests for convergence_curve."""

import numpy as np
from vizier import pyvizier
from vizier._src.benchmarks.analyzers import convergence_curve as convergence

from absl.testing import absltest
from absl.testing import parameterized


def _gen_trials(values):
  """Returns trials where trials[i] has empty metric name equal to values[i]."""
  trials = []
  for v in values:
    trial = pyvizier.Trial()
    trials.append(
        trial.complete(
            pyvizier.Measurement(metrics={'': pyvizier.Metric(value=v)})))
  return trials


class ConvergenceCurveTest(absltest.TestCase):

  def test_align_xs_merge_ys_on_different_lengths(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 2, 3]),
        ys=np.array([[2, 1, 1]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    c2 = convergence.ConvergenceCurve(
        xs=np.array([1]),
        ys=np.array([[3]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    aligned = convergence.ConvergenceCurve.align_xs([c1, c2])[0]

    np.testing.assert_array_equal(aligned.xs, [1, 2, 3])
    np.testing.assert_array_equal(aligned.ys,
                                  np.array([[2, 1, 1], [3, np.nan, np.nan]]))

  def test_align_xs_merge_ys_on_distinct_xvalues(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 3, 4]),
        ys=np.array([[2, 1, 1]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    c2 = convergence.ConvergenceCurve(
        xs=np.array([2]),
        ys=np.array([[3]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    aligned = convergence.ConvergenceCurve.align_xs([c1, c2])[0]

    np.testing.assert_array_equal(aligned.xs.shape, (3,))
    np.testing.assert_array_equal(aligned.ys,
                                  np.array([[2, 1.25, 1], [3, np.nan, np.nan]]))

  def test_align_xs_merge_ys_with_interpolation(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 2, 3, 4, 5]),
        ys=np.array([[2, 2, 1, 0.5, 0.5], [1, 1, 1, 1, 1]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    aligned = convergence.ConvergenceCurve.align_xs(
        [c1], interpolate_repeats=True
    )[0]

    np.testing.assert_array_equal(aligned.xs, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(
        aligned.ys, np.array([[2, 1.5, 1.0, 0.5, 0.5], [1, 1, 1, 1, 1]]))

  def test_extrapolate_ys_with_steps(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 2, 3, 4]),
        ys=np.array([[2, 1.5, 1, 0.5], [1, 1, 1, 1]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)

    extra_c1 = convergence.ConvergenceCurve.extrapolate_ys(c1, steps=2)

    np.testing.assert_array_equal(extra_c1.xs.shape, (6,))
    np.testing.assert_array_equal(
        extra_c1.ys,
        np.array([[2, 1.5, 1.0, 0.5, 0.0, -0.5], [1, 1, 1, 1, 1, 1]]))

  def test_align_xs_merge_ys_on_increasing_and_dicreasing_fails(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 3, 4]),
        ys=np.array([[2, 3, 3]]),
        trend=convergence.ConvergenceCurve.YTrend.INCREASING,
    )
    c2 = convergence.ConvergenceCurve(
        xs=np.array([2]),
        ys=np.array([[3]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    with self.assertRaisesRegex(ValueError, 'increasing'):
      # pylint: disable=[expression-not-assigned]
      convergence.ConvergenceCurve.align_xs([c1, c2])[0]

  def test_align_xs_keep_ys(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 3, 4]),
        ys=np.array([[2, 1, 1], [8, 3, 1]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
    c2 = convergence.ConvergenceCurve(
        xs=np.array([2]),
        ys=np.array([[3]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )

    aligned1, aligned2 = convergence.ConvergenceCurve.align_xs(
        [c1, c2], keep_curves_separate=True
    )

    np.testing.assert_array_equal(aligned1.xs, np.array([1.0, 2.5, 4.0]))
    np.testing.assert_array_equal(aligned2.xs, np.array([1.0, 2.5, 4.0]))

    np.testing.assert_array_equal(
        aligned1.ys, np.array([[2.0, 1.25, 1.0], [8.0, 4.25, 1.0]])
    )
    np.testing.assert_array_equal(
        aligned2.ys, np.array([[3.0, np.nan, np.nan]])
    )


class ConvergenceCurveConverterTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('maximize', pyvizier.ObjectiveMetricGoal.MAXIMIZE, [[2, 2, 3]]),
      ('minimize', pyvizier.ObjectiveMetricGoal.MINIMIZE, [[2, 1, 1]]))
  def test_convert_basic(self, goal, expected):
    trials = _gen_trials([2, 1, 3])
    generator = convergence.ConvergenceCurveConverter(
        pyvizier.MetricInformation(name='', goal=goal))
    curve = generator.convert(trials)
    np.testing.assert_array_equal(curve.xs, [1, 2, 3])
    np.testing.assert_array_equal(curve.ys, expected)

  @parameterized.named_parameters(
      ('maximize', pyvizier.ObjectiveMetricGoal.MAXIMIZE, [[2, 2, 3]]),
      ('minimize', pyvizier.ObjectiveMetricGoal.MINIMIZE, [[-2, -1, -1]]),
  )
  def test_convert_flip_signs(self, goal, expected):
    trials = _gen_trials([2, 1, 3])
    generator = convergence.ConvergenceCurveConverter(
        pyvizier.MetricInformation(name='', goal=goal), flip_signs_for_min=True
    )
    curve = generator.convert(trials)
    np.testing.assert_array_equal(curve.xs, [1, 2, 3])
    np.testing.assert_array_equal(curve.ys, expected)


class HyperConvergenceCurveConverterTest(parameterized.TestCase):

  def test_convert_basic(self):
    generator = convergence.HypervolumeCurveConverter([
        pyvizier.MetricInformation(
            name='max', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
        ),
        pyvizier.MetricInformation(
            name='min', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        ),
    ])
    pytrials = []
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 4.0, 'min': 2.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 3.0, 'min': -1.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 4.0, 'min': -2.0})
        )
    )

    curve = generator.convert(pytrials)
    np.testing.assert_array_equal(curve.xs, [1, 2, 3])
    np.testing.assert_array_almost_equal(curve.ys, [[0.0, 3.0, 8.0]], decimal=1)


class LogEfficiencyConvergenceComparatorTest(absltest.TestCase):

  def setUp(self):
    super(LogEfficiencyConvergenceComparatorTest, self).setUp()
    xs = np.array(range(0, 20))
    xs_t = xs.reshape(1, len(xs))
    self._baseline_curve = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(np.array([-0.9, -1.0, -1.1]).reshape(3, 1) * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)

    self._worse_curves = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(-0.5 * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    self._better_curves = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(np.array([-1.5, -1.8, -2.0]).reshape(3, 1) * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)

  def test_get_relative_efficiency_curve(self):
    baseline_length = len(self._baseline_curve.xs)
    rel_effiency = convergence.LogEfficiencyConvergenceCurveComparator(
        baseline_curve=self._baseline_curve, compared_curve=self._better_curves
    ).log_efficiency_curve()
    higher_quantile = convergence.LogEfficiencyConvergenceCurveComparator(
        baseline_curve=self._baseline_curve,
        compared_curve=self._better_curves,
        compared_quantile=0.9,
    ).log_efficiency_curve()

    self.assertEqual(rel_effiency.ys.shape, (1, baseline_length))
    self.assertEqual(higher_quantile.ys.shape, (1, baseline_length))
    # Better curves should have positive efficiency.
    self.assertTrue((rel_effiency.ys >= 0.0).all())
    # Higher quantile means better efficiency which means more positive scores.
    self.assertTrue((higher_quantile.ys >= rel_effiency.ys).all())

  def test_get_relative_efficiency_flat(self):
    flat_curve = convergence.ConvergenceCurve(
        xs=np.array(range(0, 20)),
        ys=np.array([4.0, 3.0, 2.0] + [1.5] * 17).reshape(1, 20),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    self_eff = convergence.LogEfficiencyConvergenceCurveComparator(
        baseline_curve=flat_curve, compared_curve=flat_curve
    ).log_efficiency_curve()
    # Relative efficiency of a curve on itself is close to 0.
    self.assertAlmostEqual(np.linalg.norm(self_eff.ys), 0.0, delta=0.1)

  def test_get_relative_efficiency_short_curve(self):
    baseline_length = len(self._baseline_curve.xs)
    short_length = round(baseline_length / 2)
    short_curve = convergence.ConvergenceCurve(
        xs=self._baseline_curve.xs[:short_length],
        ys=self._baseline_curve.ys[:, :short_length],
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    short_efficiency = convergence.LogEfficiencyConvergenceCurveComparator(
        baseline_curve=self._baseline_curve, compared_curve=short_curve
    ).log_efficiency_curve()
    self.assertEqual(short_efficiency.ys.shape, (1, baseline_length))
    self.assertEqual(float(short_efficiency.ys[:, -1]), -float('inf'))

  def test_get_efficiency_score(self):
    # Higher compared quantile should increase score. Higher baseline
    # quantile should decrease score.
    median_score = convergence.LogEfficiencyConvergenceCurveComparator(
        baseline_curve=self._baseline_curve, compared_curve=self._better_curves
    ).score()
    self.assertGreater(
        convergence.LogEfficiencyConvergenceCurveComparator(
            baseline_curve=self._baseline_curve,
            compared_curve=self._better_curves,
            compared_quantile=0.9,
        ).score(),
        median_score,
    )
    self.assertLess(
        convergence.LogEfficiencyConvergenceCurveComparator(
            baseline_curve=self._baseline_curve,
            compared_curve=self._better_curves,
            baseline_quantile=0.9,
        ).score(),
        median_score,
    )

  def test_effiency_score_symmetry(self):
    base_score = convergence.LogEfficiencyConvergenceCurveComparator(
        baseline_curve=self._baseline_curve, compared_curve=self._better_curves
    ).score()
    reversed_score = convergence.LogEfficiencyConvergenceCurveComparator(
        baseline_curve=self._better_curves, compared_curve=self._baseline_curve
    ).score()
    self.assertAlmostEqual(base_score, -reversed_score, delta=0.01)

  def test_efficiency_score_value(self):
    xs = self._baseline_curve.xs
    xs_t = xs.reshape(1, len(xs))

    worse_curves = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(-0.5 * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    better_curves = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(np.array([-1.5, -1.8, -2.0]).reshape(3, 1) * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    # Efficiency score for exponential curves can be approximated.

    self.assertGreater(
        convergence.LogEfficiencyConvergenceCurveComparator(
            baseline_curve=self._baseline_curve, compared_curve=better_curves
        ).score(),
        0.4,
    )
    self.assertLess(
        convergence.LogEfficiencyConvergenceCurveComparator(
            baseline_curve=self._baseline_curve, compared_curve=worse_curves
        ).score(),
        -0.4,
    )

  def test_comparator_failure(self):
    unknown_curve = convergence.ConvergenceCurve(
        xs=self._baseline_curve.xs, ys=self._baseline_curve.ys)
    with self.assertRaisesRegex(ValueError, 'increasing or decreasing'):
      convergence.LogEfficiencyConvergenceCurveComparator(
          baseline_curve=unknown_curve, compared_curve=self._baseline_curve
      )


class SimpleConvergenceComparatorTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'ys1': np.array([[11, 12, 20, 50, 100, 300]]),
          'ys2': np.array([[1, 2, 10]]),
          'cutoff': None,
          'res': 0.0,
      },
      {
          'ys1': np.array([[1, 2, 10]]),
          'ys2': np.array([[11, 12, 20, 50, 100, 300]]),
          'cutoff': None,
          'res': 1.0,
      },
      {
          'ys1': np.array([[1, 4, 8, 10]]),
          'ys2': np.array([[2, 5, 6, 8]]),
          'cutoff': None,
          'res': 0.5,
      },
      {
          'ys1': np.array([[1, 4, 8, 10, 12]]),
          'ys2': np.array([[2, 5, 6, 8, 10]]),
          'cutoff': 1,
          'res': 0.25,
      },
      {
          'ys1': np.array([[1, 4, 8, 10, 12]]),
          'ys2': np.array([[2, 5, 6, 8, 10]]),
          'cutoff': 3,
          'res': 0.0,
      },
      {
          'ys1': np.array([[2, 5, 6, 8, 10]]),
          'ys2': np.array([[1, 4, 8, 10, 12]]),
          'cutoff': 3,
          'res': 1.0,
      },
  )
  def test_score_one_curve_above_other(self, ys1, ys2, res, cutoff):
    xs1 = np.arange(ys1.shape[1])
    xs2 = np.arange(ys2.shape[1])
    curve1 = convergence.ConvergenceCurve(
        xs=xs1, ys=ys1, trend=convergence.ConvergenceCurve.YTrend.INCREASING
    )
    curve2 = convergence.ConvergenceCurve(
        xs=xs2, ys=ys2, trend=convergence.ConvergenceCurve.YTrend.INCREASING
    )
    comparator = convergence.SimpleConvergenceCurveComparator(
        curve1, curve2, xs_cutoff=cutoff
    )
    self.assertEqual(comparator.score(), res)


class PercentageBetterConvergenceComparatorTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'ys1': np.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
          'ys2': np.array([[-1, 20, 30, 70]]),
          'res': 0.5,
      },
      {
          'ys1': np.array([[1, 2, 3, 4]]),
          'ys2': np.array([[10, 20, 30, 70]]),
          'res': 1.0,
      },
      {
          'ys1': np.array([[10, 20, 30, 70]]),
          'ys2': np.array([[1, 2, 3, 4]]),
          'res': -1.0,
      },
  )
  def test_score(self, ys1, ys2, res):
    xs1 = np.arange(ys1.shape[1])
    xs2 = np.arange(ys2.shape[1])
    curve1 = convergence.ConvergenceCurve(
        xs=xs1, ys=ys1, trend=convergence.ConvergenceCurve.YTrend.INCREASING
    )
    curve2 = convergence.ConvergenceCurve(
        xs=xs2, ys=ys2, trend=convergence.ConvergenceCurve.YTrend.INCREASING
    )
    comparator = convergence.PercentageBetterConvergenceCurveComparator(
        curve1, curve2
    )
    self.assertEqual(comparator.score(), res)


if __name__ == '__main__':
  absltest.main()
