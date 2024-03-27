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
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
    c2 = convergence.ConvergenceCurve(
        xs=np.array([1]),
        ys=np.array([[3]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
    aligned = convergence.ConvergenceCurve.align_xs([c1, c2])[0]

    np.testing.assert_array_equal(aligned.xs, [1, 2, 3])
    np.testing.assert_array_equal(
        aligned.ys, np.array([[2, 1, 1], [3, np.nan, np.nan]])
    )

  def test_align_xs_merge_ys_on_distinct_xvalues(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 3, 4]),
        ys=np.array([[2, 1, 1]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
    c2 = convergence.ConvergenceCurve(
        xs=np.array([2]),
        ys=np.array([[3]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
    aligned = convergence.ConvergenceCurve.align_xs([c1, c2])[0]

    np.testing.assert_array_equal(aligned.xs.shape, (3,))
    np.testing.assert_array_equal(
        aligned.ys, np.array([[2, 1.25, 1], [3, np.nan, np.nan]])
    )

  def test_align_xs_merge_ys_with_interpolation(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 2, 3, 4, 5]),
        ys=np.array([[2, 2, 1, 0.5, 0.5], [1, 1, 1, 1, 1]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
    aligned = convergence.ConvergenceCurve.align_xs(
        [c1], interpolate_repeats=True
    )[0]

    np.testing.assert_array_equal(aligned.xs, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(
        aligned.ys, np.array([[2, 1.5, 1.0, 0.5, 0.5], [1, 1, 1, 1, 1]])
    )

  def test_extrapolate_ys_with_steps(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 2, 3, 4]),
        ys=np.array([[2, 1.5, 1, 0.5], [1, 1, 1, 1]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )

    extra_c1 = convergence.ConvergenceCurve.extrapolate_ys(c1, steps=2)

    np.testing.assert_array_equal(extra_c1.xs.shape, (6,))
    np.testing.assert_array_equal(
        extra_c1.ys,
        np.array([[2, 1.5, 1.0, 0.5, 0.0, -0.5], [1, 1, 1, 1, 1, 1]]),
    )

  def test_align_xs_merge_ys_on_increasing_and_dicreasing_fails(self):
    c1 = convergence.ConvergenceCurve(
        xs=np.array([1, 3, 4]),
        ys=np.array([[2, 3, 3]]),
        trend=convergence.ConvergenceCurve.YTrend.INCREASING,
    )
    c2 = convergence.ConvergenceCurve(
        xs=np.array([2]),
        ys=np.array([[3]]),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
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
      ('minimize', pyvizier.ObjectiveMetricGoal.MINIMIZE, [[2, 1, 1]]),
  )
  def test_convert_basic(self, goal, expected):
    trials = _gen_trials([2, 1, 3])
    generator = convergence.ConvergenceCurveConverter(
        pyvizier.MetricInformation(name='', goal=goal)
    )
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

  @parameterized.parameters(
      (
          pyvizier.ObjectiveMetricGoal.MAXIMIZE,
          2,
          [2, 1, 4, 5],
          [[2, 2, 5, 5]],
      ),
      (
          pyvizier.ObjectiveMetricGoal.MAXIMIZE,
          2,
          [4, 5, 2, 1],
          [[5, 5, 5, 5]],
      ),
      (
          pyvizier.ObjectiveMetricGoal.MAXIMIZE,
          2,
          [2, 1, 4],
          [[2, 2, 4]],
      ),
      (
          pyvizier.ObjectiveMetricGoal.MAXIMIZE,
          3,
          [2, 1, 4, 7, 9, 8, 10],
          [[4, 4, 4, 9, 9, 9, 10]],
      ),
      (
          pyvizier.ObjectiveMetricGoal.MINIMIZE,
          3,
          [3, 7, 2, 3, 0],
          [[2, 2, 2, 0, 0]],
      ),
  )
  def test_convert_with_batch_size(self, goal, batch_size, values, expected):
    trials = _gen_trials(values)
    generator = convergence.ConvergenceCurveConverter(
        pyvizier.MetricInformation(name='', goal=goal),
        flip_signs_for_min=False,
        batch_size=batch_size,
    )
    curve = generator.convert(trials)
    np.testing.assert_array_equal(curve.xs, list(range(1, len(trials) + 1)))
    np.testing.assert_array_equal(curve.ys, np.float_(expected))

  @parameterized.named_parameters(
      ('maximize', pyvizier.ObjectiveMetricGoal.MAXIMIZE, [[-np.inf, 2, 2]]),
      ('minimize', pyvizier.ObjectiveMetricGoal.MINIMIZE, [[-np.inf, -2, -1]]),
  )
  def test_convert_flip_signs_inf(self, goal, expected):
    sign = 1 if goal == pyvizier.ObjectiveMetricGoal.MINIMIZE else -1
    trials = _gen_trials([sign * np.inf, 2, 1])
    generator = convergence.ConvergenceCurveConverter(
        pyvizier.MetricInformation(name='', goal=goal), flip_signs_for_min=True
    )
    curve = generator.convert(trials)
    np.testing.assert_array_equal(curve.xs, [1, 2, 3])
    np.testing.assert_array_equal(curve.ys, expected)

  @parameterized.parameters(
      (pyvizier.ObjectiveMetricGoal.MAXIMIZE, [2, 1, 4, 5], 2),
      (pyvizier.ObjectiveMetricGoal.MAXIMIZE, [2, 1, 4], 1),
      (pyvizier.ObjectiveMetricGoal.MAXIMIZE, [2, 1, 4, 7, 9, 8, 10], 4),
      (pyvizier.ObjectiveMetricGoal.MINIMIZE, [3, 7, 2, 3, 4], 3),
  )
  def test_convert_with_state_updates(self, goal, values, split_idx):
    trials = _gen_trials(values)
    generator = convergence.ConvergenceCurveConverter(
        pyvizier.MetricInformation(name='', goal=goal),
        flip_signs_for_min=False,
    )
    curve = generator.convert(trials)

    generator = convergence.ConvergenceCurveConverter(
        pyvizier.MetricInformation(name='', goal=goal),
        flip_signs_for_min=False,
    )
    first_trials = trials[:split_idx]
    second_trials = trials[split_idx:]
    first_curve = generator.convert(first_trials)
    second_curve = generator.convert(second_trials)

    np.testing.assert_array_equal(
        curve.xs, np.concatenate((first_curve.xs, second_curve.xs), axis=-1)
    )
    np.testing.assert_array_equal(
        curve.ys, np.concatenate((first_curve.ys, second_curve.ys), axis=-1)
    )


class HypervolumeCurveConverterTest(parameterized.TestCase):

  def test_convert_with_origin_reference(self):
    generator = convergence.HypervolumeCurveConverter(
        [
            pyvizier.MetricInformation(
                name='max', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
            ),
            pyvizier.MetricInformation(
                name='min', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
            ),
        ],
        reference_value=np.array([0.0]),
    )
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
    np.testing.assert_array_almost_equal(
        curve.ys, [[0.0, 3.0, 8.0]], decimal=0.5
    )

  def test_convert_with_reference(self):
    generator = convergence.HypervolumeCurveConverter(
        [
            pyvizier.MetricInformation(
                name='max', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
            ),
            pyvizier.MetricInformation(
                name='min', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
            ),
        ],
        reference_value=np.array([3.0, 0.0]),
    )
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
    np.testing.assert_array_almost_equal(
        curve.ys, [[0.0, 0.0, 2.0]], decimal=0.5
    )

  def test_convert_with_none_reference(self):
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
            pyvizier.Measurement(metrics={'max': -np.inf, 'min': np.inf})
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
    np.testing.assert_array_almost_equal(
        curve.ys, [[0.0, 0.0, 1.0]], decimal=0.5
    )

  def test_convert_with_inf_none_reference(self):
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
            pyvizier.Measurement(metrics={'max': -np.inf, 'min': np.inf})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': -np.inf, 'min': 2.0})
        )
    )

    curve = generator.convert(pytrials)
    np.testing.assert_array_equal(curve.xs, [1, 2])
    np.testing.assert_array_equal(curve.ys, [[0.0, 0.0]])

  def test_convert_with_state(self):
    generator = convergence.HypervolumeCurveConverter(
        [
            pyvizier.MetricInformation(
                name='max', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
            ),
            pyvizier.MetricInformation(
                name='min', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
            ),
        ],
        reference_value=np.array([0.0]),
    )
    pytrials = []
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 4.0, 'min': 2.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 5.0, 'min': -1.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 4.0, 'min': -2.0})
        )
    )

    curve = generator.convert(pytrials)
    np.testing.assert_array_equal(curve.xs, [1, 2, 3])
    np.testing.assert_array_almost_equal(
        curve.ys, [[0.0, 5.0, 9.0]], decimal=0.5
    )

    pytrials = []
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 2.0, 'min': -0.1})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 5.0, 'min': -2.0})
        )
    )

    curve = generator.convert(pytrials)
    np.testing.assert_array_equal(curve.xs, [4, 5])
    np.testing.assert_array_almost_equal(curve.ys, [[9.0, 10.0]], decimal=0.5)

  def test_convert_factor_with_inf(self):
    generator = convergence.HypervolumeCurveConverter(
        [
            pyvizier.MetricInformation(
                name='max', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
            ),
            pyvizier.MetricInformation(
                name='min', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
            ),
        ],
        infer_origin_factor=1.0,
    )
    pytrials = []
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': -np.inf, 'min': np.inf})
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
    np.testing.assert_array_almost_equal(
        curve.ys, [[0.0, 1.0, 4.0]], decimal=0.5
    )


class MultiMetricCurveConverterTest(parameterized.TestCase):

  def test_convert_single_objective(self):
    generator = convergence.MultiMetricCurveConverter.from_metrics_config(
        metrics_config=pyvizier.MetricsConfig([
            pyvizier.MetricInformation(
                name='min', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
            ),
            pyvizier.MetricInformation(
                name='safe',
                goal=pyvizier.ObjectiveMetricGoal.MINIMIZE,
                safety_threshold=0.1,
            ),
        ]),
        flip_signs_for_min=True,
    )
    pytrials = []
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'min': 4.0, 'safe': -2.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'min': 3.0, 'safe': 1.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'min': 2.0, 'safe': -1.0})
        )
    )

    curve = generator.convert(pytrials)
    np.testing.assert_array_equal(curve.xs, [1, 2, 3])
    np.testing.assert_array_almost_equal(curve.ys, [[-4.0, -4.0, -2.0]])

  def test_convert_multiobjective(self):
    generator = convergence.MultiMetricCurveConverter.from_metrics_config(
        metrics_config=pyvizier.MetricsConfig([
            pyvizier.MetricInformation(
                name='max', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
            ),
            pyvizier.MetricInformation(
                name='min', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
            ),
            pyvizier.MetricInformation(
                name='safe',
                goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE,
                safety_threshold=-0.2,
            ),
        ]),
        reference_value=np.array([0.0]),
    )
    pytrials = []
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 4.0, 'min': -1.0, 'safe': 1.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(
                metrics={'max': 5.0, 'min': -1.0, 'safe': -1.0}
            )
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 4.0, 'min': -2.0, 'safe': 2.1})
        )
    )

    curve = generator.convert(pytrials)
    np.testing.assert_array_equal(curve.xs, [1, 2, 3])
    np.testing.assert_array_almost_equal(
        curve.ys, [[4.0, 4.0, 8.0]], decimal=0.5
    )


class RestartingCurveConverterTest(absltest.TestCase):

  def test_convert_with_restart(self):
    def converter_factory():
      return convergence.HypervolumeCurveConverter(
          [
              pyvizier.MetricInformation(
                  name='max', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
              ),
              pyvizier.MetricInformation(
                  name='min', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
              ),
          ],
      )

    restart_converter = convergence.RestartingCurveConverter(
        converter_factory, restart_min_trials=2, restart_rate=1.001
    )
    pytrials = []
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 0.0, 'min': 0.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 5.0, 'min': -1.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 4.0, 'min': -2.0})
        )
    )

    curve = restart_converter.convert(pytrials)
    np.testing.assert_array_equal(curve.xs, [1, 2, 3])
    np.testing.assert_array_almost_equal(
        curve.ys, [[0.0, 5.0, 9.0]], decimal=0.5
    )

    pytrials = []
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': -1.0, 'min': 1.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 5.0, 'min': -2.0})
        )
    )

    restart_converter.convert([pytrials[0]])
    curve = restart_converter.convert([pytrials[1]])
    np.testing.assert_array_equal(curve.xs, [5])
    np.testing.assert_array_almost_equal(curve.ys, [[18.0]], decimal=0.5)

  def test_convert_no_restart(self):
    def converter_factory():
      return convergence.HypervolumeCurveConverter(
          [
              pyvizier.MetricInformation(
                  name='max', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
              ),
              pyvizier.MetricInformation(
                  name='min', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
              ),
          ],
      )

    converter = convergence.RestartingCurveConverter(
        converter_factory, restart_min_trials=10
    )
    pytrials = []
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 0.0, 'min': 0.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 5.0, 'min': -1.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 4.0, 'min': -2.0})
        )
    )

    curve = converter.convert(pytrials)
    np.testing.assert_array_equal(curve.xs, [1, 2, 3])
    np.testing.assert_array_almost_equal(
        curve.ys, [[0.0, 5.0, 9.0]], decimal=0.5
    )

    pytrials = []
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': -1.0, 'min': 1.0})
        )
    )
    pytrials.append(
        pyvizier.Trial().complete(
            pyvizier.Measurement(metrics={'max': 5.0, 'min': -2.0})
        )
    )

    curve = converter.convert(pytrials)
    np.testing.assert_array_equal(curve.xs, [4, 5])
    np.testing.assert_array_almost_equal(curve.ys, [[9.0, 10.0]], decimal=0.5)


class WinRateConvergenceCurveComparatorAllComparisonsModeTest(
    absltest.TestCase
):
  """Test the WinRateConvergenceCurveComparator with "all comparisons" mode."""

  def setUp(self):
    super(WinRateConvergenceCurveComparatorAllComparisonsModeTest, self).setUp()
    xs = np.array(range(0, 20))
    xs_t = xs.reshape(1, len(xs))
    self._baseline_curve = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(np.array([-0.9, -1.0, -1.1]).reshape(3, 1) * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )

    self._worse_curves = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(-0.5 * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
    self._better_curves = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(np.array([-1.5, -1.8, -2.0]).reshape(3, 1) * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )

  def test_higher_quantile_curve(self):
    baseline_length = len(self._baseline_curve.xs)
    median_score = convergence.WinRateConvergenceCurveComparator(
        baseline_curve=self._baseline_curve, compared_curve=self._better_curves
    ).curve()
    reverse_median_score = convergence.WinRateConvergenceCurveComparator(
        baseline_curve=self._baseline_curve, compared_curve=self._worse_curves
    ).curve()
    higher_quantile_score = convergence.WinRateConvergenceCurveComparator(
        baseline_curve=self._baseline_curve,
        compared_curve=self._better_curves,
        compared_quantile=0.9,
    ).curve()

    self.assertEqual(median_score.ys.shape, (1, baseline_length))
    self.assertEqual(higher_quantile_score.ys.shape, (1, baseline_length))
    # Better curves should have positive efficiency.
    self.assertTrue((median_score.ys >= 0.0).all())
    self.assertTrue((reverse_median_score.ys <= 0.0).all())
    # Higher quantile means better efficiency which means more positive scores.
    self.assertTrue((higher_quantile_score.ys >= median_score.ys).all())

  def test_get_winrate_score(self):
    # Higher compared quantile should increase score. Higher baseline
    # quantile should decrease score.
    median_score = convergence.WinRateConvergenceCurveComparator(
        baseline_curve=self._baseline_curve, compared_curve=self._better_curves
    ).score()
    self.assertGreaterEqual(
        convergence.WinRateConvergenceCurveComparator(
            baseline_curve=self._baseline_curve,
            compared_curve=self._better_curves,
            compared_quantile=0.9,
        ).score(),
        median_score,
    )
    self.assertLessEqual(
        convergence.WinRateConvergenceCurveComparator(
            baseline_curve=self._baseline_curve,
            compared_curve=self._better_curves,
            baseline_quantile=0.9,
        ).score(),
        median_score,
    )


class LogEfficiencyConvergenceComparatorTest(absltest.TestCase):

  def setUp(self):
    super(LogEfficiencyConvergenceComparatorTest, self).setUp()
    xs = np.array(range(0, 20))
    xs_t = xs.reshape(1, len(xs))
    self._baseline_curve = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(np.array([-0.9, -1.0, -1.1]).reshape(3, 1) * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )

    self._worse_curves = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(-0.5 * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
    self._better_curves = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(np.array([-1.5, -1.8, -2.0]).reshape(3, 1) * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )

  def test_get_relative_efficiency_curve(self):
    baseline_length = len(self._baseline_curve.xs)
    rel_effiency = convergence.LogEfficiencyConvergenceCurveComparator(
        baseline_curve=self._baseline_curve, compared_curve=self._better_curves
    ).curve()
    higher_quantile = convergence.LogEfficiencyConvergenceCurveComparator(
        baseline_curve=self._baseline_curve,
        compared_curve=self._better_curves,
        compared_quantile=0.9,
    ).curve()

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
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
    self_eff = convergence.LogEfficiencyConvergenceCurveComparator(
        baseline_curve=flat_curve, compared_curve=flat_curve
    ).curve()
    # Relative efficiency of a curve on itself is close to 0.
    self.assertAlmostEqual(np.linalg.norm(self_eff.ys), 0.0, delta=0.1)

  def test_get_relative_efficiency_short_curve(self):
    baseline_length = len(self._baseline_curve.xs)
    short_length = round(baseline_length / 2)
    short_curve = convergence.ConvergenceCurve(
        xs=self._baseline_curve.xs[:short_length],
        ys=self._baseline_curve.ys[:, :short_length],
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
    max_score = 10.3
    short_efficiency = convergence.LogEfficiencyConvergenceCurveComparator(
        baseline_curve=self._baseline_curve,
        compared_curve=short_curve,
        max_score=max_score,
    ).curve()
    self.assertEqual(short_efficiency.ys.shape, (1, baseline_length))
    self.assertEqual(float(short_efficiency.ys[:, -1]), -max_score)

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
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
    better_curves = convergence.ConvergenceCurve(
        xs=xs,
        ys=np.exp(np.array([-1.5, -1.8, -2.0]).reshape(3, 1) * xs_t),
        trend=convergence.ConvergenceCurve.YTrend.DECREASING,
    )
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
        xs=self._baseline_curve.xs, ys=self._baseline_curve.ys
    )
    with self.assertRaisesRegex(ValueError, 'increasing or decreasing'):
      convergence.LogEfficiencyConvergenceCurveComparator(
          baseline_curve=unknown_curve, compared_curve=self._baseline_curve
      )


class WinRateConvergenceCurveComparatorQuantilesModeTest(
    parameterized.TestCase
):
  """Tests for WinRateConvergenceCurveComparator with quantiles mode."""

  @parameterized.parameters(
      {
          'ys1': np.array([[11, 12, 20]]),
          'ys2': np.array([[1, 2, 10]]),
          'res': -0.5,
      },
      {
          'ys1': np.array([[1, 2, 10]]),
          'ys2': np.array([[11, 12, 20]]),
          'res': 0.5,
      },
      {
          'ys1': np.array([[1, 4, 8, 10]]),
          'ys2': np.array([[2, 5, 6, 8]]),
          'res': 0.0,
      },
  )
  def test_score_one_curve_above_other(self, ys1, ys2, res):
    xs1 = np.arange(ys1.shape[1])
    xs2 = np.arange(ys2.shape[1])
    curve1 = convergence.ConvergenceCurve(
        xs=xs1, ys=ys1, trend=convergence.ConvergenceCurve.YTrend.INCREASING
    )
    curve2 = convergence.ConvergenceCurve(
        xs=xs2, ys=ys2, trend=convergence.ConvergenceCurve.YTrend.INCREASING
    )
    comparator = convergence.WinRateConvergenceCurveComparator(
        curve1, curve2, comparison_mode='quantiles'
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


class OptimalityGapGainComparatorTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'ys1': np.array([[1, 4, 8, 10, 12]]),
          'ys2': np.array([[2, 5, 6, 8, 10]]),
          'trend': convergence.ConvergenceCurve.YTrend.INCREASING,
          'res': (10 - 12) / 12,
      },
      {
          'ys1': np.array([[1, 4, 8, 10, 12]]),
          'ys2': np.array([[2, 5, 6, 8, 1000]]),
          'trend': convergence.ConvergenceCurve.YTrend.INCREASING,
          'res': 1.0,
      },
      {
          'ys1': np.array([[1, 4, 8, 10, 1000]]),
          'ys2': np.array([[2, 5, 6, 8, 10]]),
          'trend': convergence.ConvergenceCurve.YTrend.INCREASING,
          'res': -0.5,
      },
      {
          'ys1': np.array([[11, 5, 3, 1]]),
          'ys2': np.array([[130, 4, 2, 0.5]]),
          'trend': convergence.ConvergenceCurve.YTrend.DECREASING,
          'res': (-0.5 - (-1)) / 1,
      },
  )
  def test_score(self, ys1, ys2, trend, res):
    xs1 = np.arange(ys1.shape[1])
    xs2 = np.arange(ys2.shape[1])
    curve1 = convergence.ConvergenceCurve(xs=xs1, ys=ys1, trend=trend)
    curve2 = convergence.ConvergenceCurve(xs=xs2, ys=ys2, trend=trend)
    comparator = convergence.OptimalityGapGainComparator(curve1, curve2)
    self.assertAlmostEqual(comparator.score(), res, delta=0.0001)


class OptimalityGapWinRateComparatorTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'ys1': np.array([[1, 4, 8, 10, 12]]),
          'ys2': np.array([[2, 5, 6, 8, 10]]),
          'trend': convergence.ConvergenceCurve.YTrend.INCREASING,
          'res': 0.0,
      },
      {
          'ys1': np.array([[1, 4, 8, 10, 12]]),
          'ys2': np.array([[2, 5, 6, 8, 1000]]),
          'trend': convergence.ConvergenceCurve.YTrend.INCREASING,
          'res': 1.0,
      },
      {
          'ys1': np.array([[1, 4, 8, 10, 1000]]),
          'ys2': np.array([[2, 5, 6, 8, 10]]),
          'trend': convergence.ConvergenceCurve.YTrend.INCREASING,
          'res': 0.0,
      },
      {
          'ys1': np.array([[11, 5, 3, 1]]),
          'ys2': np.array([[130, 4, 2, 0.5]]),
          'trend': convergence.ConvergenceCurve.YTrend.DECREASING,
          'res': 1.0,
      },
  )
  def test_score(self, ys1, ys2, trend, res):
    xs1 = np.arange(ys1.shape[1])
    xs2 = np.arange(ys2.shape[1])
    curve1 = convergence.ConvergenceCurve(xs=xs1, ys=ys1, trend=trend)
    curve2 = convergence.ConvergenceCurve(xs=xs2, ys=ys2, trend=trend)
    comparator = convergence.OptimalityGapWinRateComparator(curve1, curve2)
    self.assertEqual(comparator.score(), res)


if __name__ == '__main__':
  absltest.main()
