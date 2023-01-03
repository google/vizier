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

  def test_align_xs_on_different_lengths(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 2, 3]),
        ys=np.array([[2, 1, 1]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    c2 = convergence.ConvergenceCurve(
        xs=np.array([1]),
        ys=np.array([[3]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    aligned = convergence.ConvergenceCurve.align_xs([c1, c2])

    np.testing.assert_array_equal(aligned.xs, [1, 2, 3])
    np.testing.assert_array_equal(aligned.ys,
                                  np.array([[2, 1, 1], [3, np.nan, np.nan]]))

  def test_align_xs_on_distinct_xvalues(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 3, 4]),
        ys=np.array([[2, 1, 1]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    c2 = convergence.ConvergenceCurve(
        xs=np.array([2]),
        ys=np.array([[3]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    aligned = convergence.ConvergenceCurve.align_xs([c1, c2])

    np.testing.assert_array_equal(aligned.xs.shape, (3,))
    np.testing.assert_array_equal(aligned.ys,
                                  np.array([[2, 1.25, 1], [3, np.nan, np.nan]]))

  def test_align_xs_with_interpolation(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 2, 3, 4, 5]),
        ys=np.array([[2, 2, 1, 0.5, 0.5], [1, 1, 1, 1, 1]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    aligned = convergence.ConvergenceCurve.align_xs([c1],
                                                    interpolate_repeats=True)

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

  def test_align_xs_on_increasing_and_dicreasing_fails(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 3, 4]),
        ys=np.array([[2, 1, 1]]),
        trend=convergence.ConvergenceCurve.YTrend.INCREASING)
    c2 = convergence.ConvergenceCurve(
        xs=np.array([2]),
        ys=np.array([[3]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    with self.assertRaisesRegex(ValueError, 'increasing'):
      convergence.ConvergenceCurve.align_xs([c1, c2])


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


class ConvergenceComparatorTest(absltest.TestCase):

  def setUp(self):
    super(ConvergenceComparatorTest, self).setUp()
    xs = np.array(range(0, 20))
    xs_t = xs.reshape(1, len(xs))
    self._baseline_curve = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(np.array([-0.9, -1.0, -1.1]).reshape(3, 1) * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    self._baseline = convergence.ConvergenceCurveComparator(
        self._baseline_curve)

    self._worse_curves = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(-0.5 * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    self._better_curves = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(np.array([-1.5, -1.8, -2.0]).reshape(3, 1) * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)

  def testGetRelativeEfficiencyCurve(self):
    baseline_length = len(self._baseline_curve.xs)
    rel_effiency = self._baseline.log_efficiency_curve(self._better_curves)
    higher_quantile = self._baseline.log_efficiency_curve(
        self._better_curves, compared_quantile=0.9)

    self.assertEqual(rel_effiency.ys.shape, (1, baseline_length))
    self.assertEqual(higher_quantile.ys.shape, (1, baseline_length))
    # Better curves should have positive efficiency.
    self.assertTrue((rel_effiency.ys >= 0.0).all())
    # Higher quantile means better efficiency which means more positive scores.
    self.assertTrue((higher_quantile.ys >= rel_effiency.ys).all())

  def testGetRelativeEfficiencyFlat(self):
    flat_curve = convergence.ConvergenceCurve(
        xs=np.array(range(0, 20)),
        ys=np.array([4.0, 3.0, 2.0] + [1.5] * 17).reshape(1, 20),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    comparator = convergence.ConvergenceCurveComparator(flat_curve)
    self_eff = comparator.log_efficiency_curve(flat_curve)
    # Relative efficiency of a curve on itself is close to 0.
    self.assertAlmostEqual(np.linalg.norm(self_eff.ys), 0.0, delta=0.1)

  def testGetRelativeEfficiencyShortCurve(self):
    baseline_length = len(self._baseline_curve.xs)
    short_length = round(baseline_length / 2)
    short_curve = convergence.ConvergenceCurve(
        xs=self._baseline_curve.xs[:short_length],
        ys=self._baseline_curve.ys[:, :short_length],
        trend=convergence.ConvergenceCurve.YTrend.DECREASING)
    short_efficiency = self._baseline.log_efficiency_curve(short_curve)
    self.assertEqual(short_efficiency.ys.shape, (1, baseline_length))
    self.assertEqual(float(short_efficiency.ys[:, -1]), -float('inf'))

  def testGetEfficiencyScore(self):
    # Higher compared quantile should increase score. Higher baseline
    # quantile should decrease score.
    median_score = self._baseline.get_log_efficiency_score(self._better_curves)
    self.assertGreater(
        self._baseline.get_log_efficiency_score(
            self._better_curves, compared_quantile=0.9), median_score)
    self.assertLess(
        self._baseline.get_log_efficiency_score(
            self._better_curves, baseline_quantile=0.9), median_score)

  def testEffiencyScoreSymmetry(self):
    base_score = self._baseline.get_log_efficiency_score(self._better_curves)
    reversed_score = convergence.ConvergenceCurveComparator(
        self._better_curves).get_log_efficiency_score(self._baseline_curve)
    self.assertAlmostEqual(base_score, -reversed_score, delta=0.01)

  def testEfficiencyScoreValue(self):
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
        self._baseline.get_log_efficiency_score(better_curves), 0.4)
    self.assertLess(self._baseline.get_log_efficiency_score(worse_curves), -0.4)

  def testComparatorFailure(self):
    unknown_curve = convergence.ConvergenceCurve(
        xs=self._baseline_curve.xs, ys=self._baseline_curve.ys)
    with self.assertRaisesRegex(ValueError, 'increasing or decreasing'):
      convergence.ConvergenceCurveComparator(unknown_curve)


if __name__ == '__main__':
  absltest.main()
