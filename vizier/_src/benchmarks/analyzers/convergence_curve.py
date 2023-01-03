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

"""Converters and comparators for convergence curves."""

import dataclasses
import enum
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
from vizier import pyvizier


@dataclasses.dataclass
class ConvergenceCurve:
  """Represents a batch of convergence curves on the same task."""
  xs: np.ndarray  # [T] array. All curves share the x axis.
  ys: np.ndarray  # [N x T] array where N is the batch size.
  ylabel: str = ''  # Optional for plotting.
  xlabel: str = ''  # Optional for plotting.
  # Indicates ys should be increasing in t or decreasing.

  @enum.unique
  class YTrend(enum.Enum):
    """Trend of ys across t."""
    UNKNOWN = 'unknown'
    INCREASING = 'increasing'
    DECREASING = 'decreasing'

  trend: YTrend = YTrend.UNKNOWN

  @property
  def batch_size(self) -> int:
    return self.ys.shape[0]

  @classmethod
  def align_xs(cls,
               curves: Sequence['ConvergenceCurve'],
               *,
               resolution: Optional[int] = None,
               interpolate_repeats=False) -> 'ConvergenceCurve':
    """Align curves to the same xs using linear interpolation.

    If xs are greater than xp[-1], then default is np.nan.

    Args:
      curves:
      resolution: Number of points to interpolate to. Leave it None to use the
        maximal resolution.
      interpolate_repeats: Interpolate repeated values in the curve via
      linear interpolation (except for the last repeated segment).

    Returns:
      ConvergenceCurve whose batch size is equal to the sum of the batch size
      of `curves`.
    """
    if not curves:
      raise ValueError('Empty sequence of curves.')
    if len(set([c.trend for c in curves])) > 1:
      raise ValueError('All curves must be increasing or decreasing.')
    minx = np.min([np.min(c.xs) for c in curves])
    maxx = np.max([np.max(c.xs) for c in curves])
    resolution = resolution or np.max([np.size(c.xs) for c in curves])
    xs = np.linspace(minx, maxx, resolution)

    all_ys = []
    for curve in curves:
      for ys in curve.ys:
        if interpolate_repeats:
          _, indices = np.unique(ys, return_index=True)
          # Sorting indices from increasing order due to sign differences.
          indices = sorted(indices)
          if len(ys) - 1 not in indices:
            indices.append(len(ys) - 1)
        else:
          # Use the whole curve.
          indices = range(len(ys))

        all_ys.append(
            np.interp(
                xs,
                np.array([curve.xs[i] for i in indices]),
                np.array([ys[i] for i in indices]),
                right=np.nan))

    # Take all non-empty ylabels.
    ylabels = list(set([c.ylabel for c in curves if c.ylabel]))
    if not ylabels:
      ylabel = ''
    elif len(ylabels) > 1:
      print('Curves have different ylabels: %s.', ylabels)
      ylabel = ''
    else:
      ylabel = ylabels[0]

    return cls(xs=xs, ys=np.stack(all_ys), ylabel=ylabel, trend=curves[0].trend)

  @classmethod
  def extrapolate_ys(cls,
                     curve: 'ConvergenceCurve',
                     steps: int = 0) -> 'ConvergenceCurve':
    """Extrapolates the future ys using a variant of linear extrapolation.

    Args:
      curve:
      steps: Number of steps to perform extrapolation.

    Returns:
      ConvergenceCurve whose xs, ys are extrapolated. Batch size remains the
      same but xs, ys now have length T + steps.
    """
    if curve.trend not in (cls.YTrend.INCREASING, cls.YTrend.DECREASING):
      raise ValueError('Curve must be increasing or decreasing.')

    all_extra_ys = []
    for ys in curve.ys:
      # Use average slope in the last half of the curve as slope extrapolate.
      ys_later_half = ys[int(len(ys) / 2):]
      slope = np.mean([
          ys_later_half[idx] - ys_later_half[idx - 1]
          for idx in range(1, len(ys_later_half))
      ])

      extra_ys = np.append(
          ys, [ys[-1] + slope * (1 + step) for step in range(steps)])
      all_extra_ys.append(extra_ys)

    return cls(
        xs=np.append(curve.xs,
                     [curve.xs[-1] + 1 + step for step in range(steps)]),
        ys=np.stack(all_extra_ys),
        ylabel=curve.ylabel,
        trend=curve.trend)


class ConvergenceCurveConverter:
  """Converter for Trial sequence to ConvergenceCurve."""

  def __init__(self,
               metric_information: pyvizier.MetricInformation,
               *,
               flip_signs: bool = False,
               cost_fn: Callable[[pyvizier.Trial], Union[float,
                                                         int]] = lambda _: 1,
               measurements_type: str = 'final'):
    """Init.

    Args:
      metric_information: Information of relevant metric.
      flip_signs: If True, flips the signs of metric values to always
      maximize. Useful when desiring all increasing curves.
      cost_fn: Cost of each Trial (to determine xs in ConvergenceCurve).
      measurements_type: ['final', 'intermediate', 'all']
    """
    self.metric_information = metric_information
    self.flip_signs = flip_signs
    self.cost_fn = cost_fn
    self.measurements_type = measurements_type

  def convert(self, trials: Sequence[pyvizier.Trial]) -> ConvergenceCurve:
    """Returns ConvergenceCurve of batch size 1."""
    yvals = [np.nan]
    xvals = [0]
    comparator = self.comparator
    for trial in trials:
      candidates = [np.nan]
      if self.measurements_type in ('final', 'all'):
        if trial.final_measurement and (self.metric_information.name
                                        in trial.final_measurement.metrics):
          candidates.append(trial.final_measurement.metrics[
              self.metric_information.name].value)
      if self.measurements_type in ('intermediate', 'all'):
        for measurement in trial.measurements:
          if self.metric_information.name in measurement.metrics:
            candidates.append(
                measurement.metrics[self.metric_information.name].value)

      yvalue = comparator(candidates)
      xvals.append(xvals[-1] + self.cost_fn(trial))
      yvals.append(comparator([yvalue, yvals[-1]]))

    yvals = np.asarray(yvals[1:])
    trend = ConvergenceCurve.YTrend.DECREASING
    if (self.metric_information.goal == pyvizier.ObjectiveMetricGoal.MAXIMIZE
       ) or (self.metric_information.goal
             == pyvizier.ObjectiveMetricGoal.MINIMIZE and self.flip_signs):
      trend = ConvergenceCurve.YTrend.INCREASING
    return ConvergenceCurve(
        xs=np.asarray(xvals[1:]),
        ys=np.asarray(yvals).reshape([1, -1]) * (-1 if self.flip_signs else 1),
        trend=trend,
        ylabel=self.metric_information.name)

  @property
  def comparator(self):
    """Comparator used for creating the convergence curve."""
    return np.nanmax if (
        self.metric_information.goal
        == pyvizier.ObjectiveMetricGoal.MAXIMIZE) else np.nanmin


class ConvergenceCurveComparator:
  """Comparator methods for ConvergenceCurves.

  Methods in this class generally return comparison metrics for a compared curve
  against a baseline curve. Only works for curves with INCREASING or DECREASING
  trend.

  Example usage:
    baseline_curve = ConvergenceCurve(...)
    comparator = ConvergenceCurveComparator(baseline_curve)
    comparator.log_efficiency_curve(compared_curve)
  """

  def __init__(self, baseline_curve: ConvergenceCurve):
    """Initialize class with baseline curve.

    Args:
      baseline_curve: A baseline ConvergenceCurve to compare against.

    Raises:
      ValueError: If baseline curve is not INCREASING or DECREASING.
    """
    if baseline_curve.trend not in (ConvergenceCurve.YTrend.INCREASING,
                                    ConvergenceCurve.YTrend.DECREASING):
      raise ValueError(f'Curve trend {baseline_curve.trend} must be either'
                       'increasing or decreasing.')
    self._baseline_curve = baseline_curve
    self._sign = 1.0 if (self._baseline_curve.trend
                         == ConvergenceCurve.YTrend.INCREASING) else -1.0

  def log_efficiency_curve(self,
                           compared_curve: ConvergenceCurve,
                           baseline_quantile: float = 0.5,
                           compared_quantile: float = 0.5) -> ConvergenceCurve:
    """Builds the log sample efficiency curve.

    The compared curve should approximately use exp(-relative efficiency)% less
    Trials than the baseline curve. Note that a positive relative effiency
    demonstrates that the compared curve is better than the baseline. Also,
    the relative efficiency CURVES are not fully symmetric due to differences
    in drops in objective values.

    Args:
      compared_curve: Compared convergence curve.
      baseline_quantile: Quantile in [0, 1] of the batched baseline curve to use
        for efficiency comparison. The higher the quantile, the better the
        quality of the baseline batch.
      compared_quantile: Quantile in [0, 1] of the batched compared curve to use
        for efficiency comparison. The higher the quantile, the better the
        quality of the baseline batch.

    Returns:
      ConvergenceCurves with ys (batch size 1) as the relative efficiency curve.

      The xs of curve is equal to the xs of baseline curve but there
      may be indices equal to positive 'inf' which indicate that no point
      in the compared curve outperforms the baseline at that index.

    Raises:
      ValueError: If the trends do mismatch.
      ValueError: If baseline_quantile or compared_quantile are not in [0, 1].
    """
    if self._baseline_curve.trend != compared_curve.trend:
      raise ValueError(
          f'Baseline curve trend {self._baseline_curve.trend}'
          f' must match compared curve trend {compared_curve.trend}')
    baseline_quantile = np.nanquantile(
        self._sign * self._baseline_curve.ys, baseline_quantile, axis=0)
    # This may not be [1,2,...] due to repeats.
    baseline_index_curve = build_convergence_curve(baseline_quantile,
                                                   baseline_quantile)

    other_index_curve = build_convergence_curve(
        baseline_quantile,
        np.nanquantile(
            self._sign * compared_curve.ys, compared_quantile, axis=0))

    ys = np.log(1 + np.asarray(baseline_index_curve)) - np.log(
        1 + np.asarray(other_index_curve))
    return ConvergenceCurve(
        xs=self._baseline_curve.xs, ys=ys.reshape(1, len(ys)))

  def get_log_efficiency_score(self,
                               compared_curve: ConvergenceCurve,
                               baseline_quantile=0.5,
                               compared_quantile=0.5,
                               max_score=5) -> float:
    """Gets a finalized log efficiency score.

    The compared curve should approximately use exp(-score)% Trials compared to
    the baseline curve. Note that a high positive score demonstrates that the
    compared curve uses less Trials and is better than the baseline.

    Args:
      compared_curve: Compared convergence curve.
      baseline_quantile: Quantile in [0, 1] of the batched baseline curve to use
        for efficiency comparison. The higher the quantile, the better the
        quality of the baseline batch.
      compared_quantile: Quantile in [0, 1] of the batched compared curve to use
        for efficiency comparison. The higher the quantile, the better the
        quality of the baseline batch.
      max_score: Maximum log efficiency score.

    Returns:
      Sample efficiency score. This score is symmetric and always finite when
      baseline_quantile <= compare_quantile (recommended setting).
    """
    baseline_curve = ConvergenceCurve.align_xs([self._baseline_curve],
                                               interpolate_repeats=True)
    compared_curve = ConvergenceCurve.align_xs([compared_curve],
                                               interpolate_repeats=True)
    # Combined curve (as the baseline) becomes the y-values at which
    # Trial efficiency is evaluated.
    combined_curve = ConvergenceCurve.align_xs([baseline_curve, compared_curve])
    combined_curve.ys = np.nanmedian(combined_curve.ys, axis=0, keepdims=True)
    comparator = ConvergenceCurveComparator(baseline_curve=combined_curve)

    # Look ahead for exp(max_score)*T steps, as score is in the log space.
    extend_steps = int(np.exp(max_score) * len(self._baseline_curve.xs))
    extended_baseline = ConvergenceCurve.extrapolate_ys(self._baseline_curve,
                                                        extend_steps)
    extended_compared = ConvergenceCurve.extrapolate_ys(compared_curve,
                                                        extend_steps)

    efficiency_baseline = comparator.log_efficiency_curve(
        extended_baseline, compared_quantile=baseline_quantile)
    efficiency_compared = comparator.log_efficiency_curve(
        extended_compared, compared_quantile=compared_quantile)

    # Clip log efficiency and return median log efficiency in last half.
    diff = np.clip(
        efficiency_compared.ys, a_min=-max_score, a_max=max_score) - np.clip(
            efficiency_baseline.ys, a_min=-max_score, a_max=max_score)
    return np.median(diff[int(len(diff) / 2):])


def build_convergence_curve(baseline_curve: Sequence[float],
                            compared_curve: Sequence[float]) -> List[float]:
  """Builds a relative convergence curve (see returns for definition).

  Args:
    baseline_curve: Baseline maximization convergence curve.
    compared_curve: Compared maximization convergence curve.

  Returns:
    A list of numbers where i-th (zero-index) element is the smallest "j" such
    that baseline_curve[i] <= compared_curve[j]
  """
  convergence_curve = []
  t1 = 0
  for t0 in range(len(baseline_curve)):
    while t1 < len(compared_curve) and compared_curve[t1] < baseline_curve[t0]:
      t1 += 1
    convergence_curve.append(float('inf') if t1 >= len(compared_curve) else t1)
  return convergence_curve
