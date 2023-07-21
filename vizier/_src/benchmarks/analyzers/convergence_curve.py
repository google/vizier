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

import abc
import bisect
import enum
import logging
from typing import Callable, List, Optional, Sequence

import attr
import numpy as np
from vizier import pyvizier
from vizier.pyvizier import converters
from vizier.pyvizier import multimetric
from vizier.pyvizier.multimetric import xla_pareto


@attr.s(auto_attribs=True)
class ConvergenceCurve:
  """Represents multiple convergence curves on the same task."""
  xs: np.ndarray  # [T] array. All curves share the x axis.
  ys: np.ndarray  # [N x T] array where N is the number of curves.
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

  def __attrs_post_init__(self):
    if len(self.ys.shape) != 2:
      raise ValueError(
          'The shape of ys should be (num_curves, n_steps); but ys shape is'
          f' {self.ys.shape}'
      )
    if len(self.xs) != self.ys.shape[1]:
      raise ValueError(
          f'Shape mismatch for time dim: {len(self.xs)} vs {self.ys.shape}'
      )
    # Allow for small numerical imprecisions.
    if self.trend == ConvergenceCurve.YTrend.INCREASING:
      if not np.all(np.nan_to_num(np.diff(self.ys, axis=-1)) >= -1e-8):
        raise ValueError(f'Increasing trend not found: {self.ys}')

    if self.trend == ConvergenceCurve.YTrend.DECREASING:
      if not np.all(np.nan_to_num(np.diff(self.ys, axis=-1)) <= 1e-8):
        raise ValueError(f'Decreasing trend not found: {self.ys}')

  @property
  def num_curves(self) -> int:
    return self.ys.shape[0]

  @classmethod
  def _interpolate_curves(
      cls,
      curves: Sequence['ConvergenceCurve'],
      *,
      resolution: Optional[int] = None,
      interpolate_repeats: bool = False,
  ) -> tuple[np.ndarray, list[np.ndarray]]:
    """Interpolate curves to have the same xs using linear interpolation.

    If xs are greater than xp[-1], then default is np.nan.

    Args:
      curves:
      resolution: Number of points to interpolate to. Leave it None to use the
        maximal resolution.
      interpolate_repeats: Interpolate repeated values (xs and ys) in the curve
        via linear interpolation (except for the last repeated segment).

    Returns:
      A tuple of the interpolated xs array, and a list of the interpolated ys
      arrays.
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
      curve_ys = []
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
        curve_ys.append(
            np.interp(
                xs,
                np.array([curve.xs[i] for i in indices]),
                np.array([ys[i] for i in indices]),
                right=np.nan,
            )
        )
      all_ys.append(np.stack(curve_ys, axis=0))
    return xs, all_ys

  @classmethod
  def _align_xs_combine_ys(
      cls,
      curves: Sequence['ConvergenceCurve'],
      *,
      resolution: Optional[int] = None,
      interpolate_repeats: bool = False,
  ) -> list['ConvergenceCurve']:
    """Align curves (same xs) using linear interpolation and combine all ys."""
    xs, all_ys = cls._interpolate_curves(
        curves, resolution=resolution, interpolate_repeats=interpolate_repeats
    )
    # Take all non-empty ylabels.
    ylabels = list(set([c.ylabel for c in curves if c.ylabel]))
    if not ylabels:
      ylabel = ''
    elif len(ylabels) > 1:
      logging.info(
          'Curves have different ylabels: %s. None of them is selected.',
          ylabels,
      )
      ylabel = ''
    else:
      ylabel = ylabels[0]
    return [
        cls(xs=xs, ys=np.vstack(all_ys), ylabel=ylabel, trend=curves[0].trend)
    ]

  @classmethod
  def _align_xs_keep_ys(
      cls,
      curves: Sequence['ConvergenceCurve'],
      *,
      resolution: Optional[int] = None,
      interpolate_repeats: bool = False,
  ) -> list['ConvergenceCurve']:
    """Align curves (same xs) using linear interpolation and keep ys."""
    xs, all_ys = cls._interpolate_curves(
        curves, resolution=resolution, interpolate_repeats=interpolate_repeats
    )
    return [
        cls(xs=xs, ys=ys, ylabel=curves[i].ylabel, trend=curves[0].trend)
        for i, ys in enumerate(all_ys)
    ]

  @classmethod
  def align_xs(
      cls,
      curves: Sequence['ConvergenceCurve'],
      *,
      resolution: Optional[int] = None,
      interpolate_repeats: bool = False,
      keep_curves_separate: bool = False,
  ) -> list['ConvergenceCurve']:
    """Align curves (same xs) using linear interpolation."""
    if keep_curves_separate:
      return cls._align_xs_keep_ys(
          curves, resolution=resolution, interpolate_repeats=interpolate_repeats
      )
    else:
      return cls._align_xs_combine_ys(
          curves, resolution=resolution, interpolate_repeats=interpolate_repeats
      )

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


@attr.define
class ConvergenceCurveConverter:
  """Converter for Trial sequence to ConvergenceCurve.

  Attributes:
      metric_information: Information of relevant metric.
      flip_signs_for_min: If True, flips the signs of metric values to always
        maximize. Useful when desiring all increasing curves.
      cost_fn: Cost of each Trial (to determine xs in ConvergenceCurve).
      measurements_type: ['final', 'intermediate', 'all']
      batch_size: Number of trials in each batch. In each batch, the order of
        trials is ignored and the best trial is used.
  """

  metric_information: pyvizier.MetricInformation
  flip_signs_for_min: bool = attr.field(default=False, kw_only=True)
  cost_fn: Callable[[pyvizier.Trial], float] = attr.field(
      default=lambda _: 1, kw_only=True
  )
  measurements_type: str = attr.field(default='final', kw_only=True)
  batch_size: int = attr.field(default=1, kw_only=True)

  def convert(self, trials: Sequence[pyvizier.Trial]) -> ConvergenceCurve:
    """Returns ConvergenceCurve with a single curve."""
    yvals = [np.nan]
    xvals = [0]
    candidates = []

    for i in range(0, len(trials), self.batch_size):
      batch = trials[i : i + self.batch_size]
      for trial in batch:
        if self.measurements_type in ('final', 'all'):
          if trial.final_measurement and (
              self.metric_information.name in trial.final_measurement.metrics
          ):
            candidates.append(
                trial.final_measurement.metrics[
                    self.metric_information.name
                ].value
            )
        if self.measurements_type in ('intermediate', 'all'):
          for measurement in trial.measurements:
            if self.metric_information.name in measurement.metrics:
              candidates.append(
                  measurement.metrics[self.metric_information.name].value
              )
        xvals.append(xvals[-1] + self.cost_fn(trial))

      new_yval = self.comparator([self.comparator(candidates), yvals[-1]])
      yvals.extend([new_yval] * len(batch))
      candidates = []

    yvals = np.asarray(yvals[1:])
    if self.metric_information.goal == pyvizier.ObjectiveMetricGoal.MAXIMIZE:
      trend = ConvergenceCurve.YTrend.INCREASING
      flipped = False
    elif self.flip_signs_for_min:
      trend = ConvergenceCurve.YTrend.INCREASING
      flipped = True
    else:
      trend = ConvergenceCurve.YTrend.DECREASING
      flipped = False
    return ConvergenceCurve(
        xs=np.asarray(xvals[1:]),
        ys=np.asarray(yvals).reshape([1, -1]) * (-1 if flipped else 1),
        trend=trend,
        ylabel=self.metric_information.name,
    )

  @property
  def comparator(self):
    """Comparator used for creating the convergence curve."""
    return np.nanmax if (
        self.metric_information.goal
        == pyvizier.ObjectiveMetricGoal.MAXIMIZE) else np.nanmin


class HypervolumeCurveConverter:
  """Converts Trials to cumulative hypervolume curve for multiobjective."""

  def __init__(
      self,
      metric_informations: Sequence[pyvizier.MetricInformation],
      *,
      reference_value: Optional[np.ndarray] = None,
      num_vectors: int = 10000,
  ):
    """Init.

    Args:
      metric_informations:
      reference_value: Reference point value from which hypervolume is computed,
        with shape that is broadcastable with (dim,). If None, this computes the
        minimum of each objective as the reference point.
      num_vectors: Number of vectors from which hypervolume is computed.
    """
    if len(metric_informations) < 2:
      raise ValueError(
          'Should not use hypervolume curve with less than'
          f'two metrics {metric_informations}'
      )
    self._metric_informations = metric_informations
    self._num_vectors = num_vectors

    def create_metric_converter(mc):
      return converters.DefaultModelOutputConverter(
          mc,
          flip_sign_for_minimization_metrics=True,
          raise_errors_for_missing_metrics=False,
      )

    self._converter = converters.DefaultTrialConverter(
        parameter_converters=[],
        metric_converters=[
            create_metric_converter(mc) for mc in metric_informations
        ],
    )
    self._origin_value = reference_value

  def convert(self, trials: Sequence[pyvizier.Trial]) -> ConvergenceCurve:
    """Returns ConvergenceCurve with a curve of shape 1 x len(trials)."""
    # Returns a len(trials) x number of metrics np.ndarray.
    metrics = self._converter.to_labels_array(trials)
    if self._origin_value is None:
      origin = np.nanmin(metrics, axis=0)
    else:
      if len(self._origin_value) == 1:
        origin = np.broadcast_to(self._origin_value, (metrics.shape[1],))
      else:
        if self._origin_value.shape != (metrics.shape[1],):
          raise ValueError(
              f'Metric shapes {self._origin_value.shape} do not '
              f'match: {(metrics.shape[1],)}'
          )
        origin = self._origin_value

    front = multimetric.ParetoFrontier(
        points=metrics,
        origin=origin,
        num_vectors=self._num_vectors,
        cum_hypervolume_base=xla_pareto.jax_cum_hypervolume_origin,
    )
    hv_curve = front.hypervolume(is_cumulative=True)
    return ConvergenceCurve(
        xs=np.asarray(range(1, len(hv_curve) + 1)),
        ys=np.asarray(hv_curve).reshape([1, -1]),
        trend=ConvergenceCurve.YTrend.INCREASING,
        ylabel='hypervolume',
    )


@attr.define
class ConvergenceComparatorBase(abc.ABC):
  """Base class for convergence curve compartors.

  Attributes:
    baseline_curve: The baseline ConvergenceCurve.
    compared_curve: The compared ConvergenceCurve.
    baseline_quantile: Quantile in [0, 1] of the batched baseline curve to use
      for efficiency comparison. The higher the quantile, the better the quality
      of the baseline batch.
    compared_quantile: Quantile in [0, 1] of the batched compared curve to use
      for efficiency comparison. The higher the quantile, the better the quality
      of the baseline batch.
  """

  _baseline_curve: ConvergenceCurve = attr.field()
  _compared_curve: ConvergenceCurve = attr.field()
  _baseline_quantile: float = attr.field(
      default=0.5,
      validator=[attr.validators.le(1), attr.validators.ge(0)],
      kw_only=True,
  )
  _compared_quantile: float = attr.field(
      default=0.5,
      validator=[attr.validators.le(1), attr.validators.ge(0)],
      kw_only=True,
  )
  _sign: float = attr.field(init=False)

  def __attrs_post_init__(self):
    """Validates the curves and determines the sign.

    Raises:
      ValueError: If baseline curve is not INCREASING or DECREASING.
      ValueError: If the trends mismatch.
      ValueError: If baseline_quantile or compared_quantile are not in [0, 1].
    """
    if self._baseline_curve.trend not in (
        ConvergenceCurve.YTrend.INCREASING,
        ConvergenceCurve.YTrend.DECREASING,
    ):
      raise ValueError(
          f'Curve trend {self._baseline_curve.trend} must be either'
          'increasing or decreasing.'
      )
    if self._baseline_curve.trend != self._compared_curve.trend:
      raise ValueError(
          f'Baseline curve trend {self._baseline_curve.trend}'
          f' must match compared curve trend {self._compared_curve.trend}'
      )
    self._sign = (
        1.0
        if (self._baseline_curve.trend == ConvergenceCurve.YTrend.INCREASING)
        else -1.0
    )

  def _standardize_curves(
      self,
      xs_cutoff: Optional[float] = None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Standardize convergence curves.

    Note: This is an helper function that the class implementing this interface
    can choose to use or not.

    1. Align xs and keeping each ys.
    2. Apply quantiles and impute NaN.
    3. Remove values where xs < xs_cutoff.

    Args:
      xs_cutoff: The xs value before which values are ignored.

    Returns:
      The standardize baseline and compared curves.
    """
    # Align the curves while keeping each ys.
    align_baseline_curve, align_compared_curve = ConvergenceCurve.align_xs(
        [self._baseline_curve, self._compared_curve],
        interpolate_repeats=False,
        keep_curves_separate=True,
    )
    # Apply batch quantiels.
    baseline_ys = np.nanquantile(
        self._sign * align_baseline_curve.ys,
        self._baseline_quantile,
        axis=0,
    )
    compared_ys = np.nanquantile(
        self._sign * align_compared_curve.ys,
        self._compared_quantile,
        axis=0,
    )
    # Impute NaN values as -inf. This happens due to `align_xs` assigning
    # np.nan for xs that are outside the original convergence curve.
    baseline_ys = np.nan_to_num(baseline_ys, nan=-np.inf)
    compared_ys = np.nan_to_num(compared_ys, nan=-np.inf)

    # Remove burn cutoff values.
    if xs_cutoff is not None:
      baseline_cutoff_ind = np.where(align_baseline_curve.xs >= xs_cutoff)[0]
      compared_cutoff_ind = np.where(align_compared_curve.xs >= xs_cutoff)[0]
      if np.size(baseline_cutoff_ind) == 0 or np.size(compared_cutoff_ind) == 0:
        raise ValueError('fThe given burn_cutoff {xs_cutoff} value is too high')
      else:
        baseline_ys = baseline_ys[baseline_cutoff_ind[0] :]
        compared_ys = compared_ys[compared_cutoff_ind[0] :]

    return baseline_ys, compared_ys

  @abc.abstractmethod
  def score(self) -> float:
    ...


@attr.define
class LogEfficiencyConvergenceCurveComparator(ConvergenceComparatorBase):
  """Comparator methods for ConvergenceCurves.

  Methods in this class generally return comparison metrics for a compared curve
  against a baseline curve. Only works for curves with INCREASING or DECREASING
  trend.

  Example usage:
    baseline_curve = ConvergenceCurve(...)
    comparator = LogEfficiencyConvergenceCurveComparator(baseline_curve)
    comparator.log_efficiency_curve(compared_curve)
  """
  max_score: float = attr.field(
      default=1.0, validator=[attr.validators.ge(0)], kw_only=True
  )
  summary_function: Callable[[np.ndarray], float] = attr.field(
      default=np.median
  )

  def log_efficiency_curve(self) -> ConvergenceCurve:
    """Builds the log sample efficiency curve.

    The compared curve should approximately use exp(-relative efficiency)% less
    Trials than the baseline curve. Note that a positive relative effiency
    demonstrates that the compared curve is better than the baseline. Also,
    the relative efficiency CURVES are not fully symmetric due to differences
    in drops in objective values.

    Returns:
      ConvergenceCurves with ys (batch size 1) as the relative efficiency curve.

      The xs of curve is equal to the xs of baseline curve but there
      may be indices equal to positive 'inf' which indicate that no point
      in the compared curve outperforms the baseline at that index.

    Raises:
      ValueError: If the trends do mismatch.
      ValueError: If baseline_quantile or compared_quantile are not in [0, 1].
    """

    baseline_quantile = np.nanquantile(
        self._sign * self._baseline_curve.ys, self._baseline_quantile, axis=0
    )
    # This may not be [1,2,...] due to repeats.
    baseline_index_curve = build_convergence_curve(baseline_quantile,
                                                   baseline_quantile)

    other_index_curve = build_convergence_curve(
        baseline_quantile,
        np.nanquantile(
            self._sign * self._compared_curve.ys,
            self._compared_quantile,
            axis=0,
        ),
    )

    ys = np.log(1 + np.asarray(baseline_index_curve)) - np.log(
        1 + np.asarray(other_index_curve))
    return ConvergenceCurve(
        xs=self._baseline_curve.xs, ys=ys.reshape(1, len(ys)))

  def score(self) -> float:
    """Gets a finalized log efficiency score.

    The compared curve should approximately use exp(-score)% Trials compared to
    the baseline curve. Note that a high positive score demonstrates that the
    compared curve uses less Trials and is better than the baseline.

    Returns:
      Sample efficiency score. This score is symmetric and always finite when
      baseline_quantile <= compare_quantile (recommended setting).
    """
    baseline_curve = ConvergenceCurve.align_xs(
        [self._baseline_curve], interpolate_repeats=True
    )[0]
    compared_curve = ConvergenceCurve.align_xs(
        [self._compared_curve], interpolate_repeats=True
    )[0]
    # Combined curve (as the baseline) becomes the y-values at which
    # Trial efficiency is evaluated.
    combined_curve = ConvergenceCurve.align_xs(
        [baseline_curve, compared_curve]
    )[0]
    combined_curve.ys = np.nanmedian(combined_curve.ys, axis=0, keepdims=True)

    # Look ahead for exp(max_score)*T steps, as score is in the log space.
    extend_steps = int(np.exp(self.max_score) * len(self._baseline_curve.xs))
    extended_baseline = ConvergenceCurve.extrapolate_ys(
        baseline_curve, extend_steps
    )
    extended_compared = ConvergenceCurve.extrapolate_ys(
        compared_curve, extend_steps
    )
    baseline_comparator = LogEfficiencyConvergenceCurveComparator(
        baseline_curve=combined_curve,
        compared_curve=extended_baseline,
        compared_quantile=self._baseline_quantile,
    )
    efficiency_baseline = baseline_comparator.log_efficiency_curve()
    compared_comparator = LogEfficiencyConvergenceCurveComparator(
        baseline_curve=combined_curve,
        compared_curve=extended_compared,
        compared_quantile=self._compared_quantile,
    )
    efficiency_compared = compared_comparator.log_efficiency_curve()

    # Clip log efficiency and return median log efficiency in last half.
    diff = np.clip(
        efficiency_compared.ys, a_min=-self.max_score, a_max=self.max_score
    ) - np.clip(
        efficiency_baseline.ys, a_min=-self.max_score, a_max=self.max_score
    )
    return self.summary_function(diff)


@attr.define
class SimpleConvergenceCurveComparator(ConvergenceComparatorBase):
  """Comparator method based on simple comparison.

  Attributes:
    burn_cutoff: The cutoff below which values not included in score.
  """

  _xs_cutoff: Optional[float] = None

  def score(self) -> float:
    """Computes the simple convergence score.

    The score is the percentage of indices (after the burn cutoff) for which the
    interpolated values of 'compared_curve' are better than the interpolated
    values of 'baseline_curve'. A large score value means 'compared' is better.

    Returns:
      The percentage better convergence score [0.0, 1.0].

    Raises:
      ValueError: If curve trends are not INCREASING or DECREASING, or not
      equal.
    """
    baseline_ys, compared_ys = self._standardize_curves(self._xs_cutoff)
    # Compute mean indices that compared is better than baseline.
    return np.mean(baseline_ys < compared_ys)


@attr.define
class PercentageBetterConvergenceCurveComparator(ConvergenceComparatorBase):
  """Comparator method based on percentage better.

  Attributes:
    burn_cutoff: The cutoff below which values not included in score.
  """

  _xs_cutoff: Optional[float] = None

  def _compute_directional_score(
      self, baseline: np.ndarray, compared: np.ndarray
  ) -> float:
    """Compute the percentage better score.

    The score is the average percentage steps that 'compared' is better than
    'baseline'.

    Note that: sum_i sum_j 1{c_j > b_i} = sum_j sum_i {b_i < c_j}. Therefore, we
    can either iterate over 'compared' and count the number of steps that
    'baseline' is worse OR we can iterate over 'baseline' and count the number
    of steps that 'compared' is better (which is the current implementation).

    Implementation
    --------------
    1. For each index of `baseline`:
      - Finds the smallest index of 'comared' that is better.
      - Compute the percentage of `compared` steps that are better.
    2. Average the percentages across all 'baseline' indices.

    The more dominante 'compared' over 'baseline' the closer the score is to
    1.0.

    Args:
      baseline: The baseline convergence curve.
      compared: The compared convergence curve.

    Returns:
      The average number of steps that compared is better than baseline (score
      is in [0, 1]).
    """
    convergence_curve = build_convergence_curve(list(baseline), list(compared))
    pct_baseline_compared = [
        (len(compared) - i) / len(compared) if i != float('inf') else 0
        for i in convergence_curve
    ]
    return np.mean(pct_baseline_compared)

  def score(self) -> float:
    """Computes the percentage better score.

    The score has two components:
    (a) sub-score quantifying how much 'compared' domniates 'baseline'.
    (b) sub-score quantifying how much 'baseline' domniates 'compared'.

    The final score is (a) - (b).

    Returns:
      The normalized percentage better convergence score [-1.0, 1.0].

    Raises:
      ValueError: If curve trends are not INCREASING or DECREASING, or not
      equal.
    """
    baseline_ys, compared_ys = self._standardize_curves(self._xs_cutoff)
    baseline_compared_score = self._compute_directional_score(
        baseline_ys, compared_ys
    )
    compared_baseline_score = self._compute_directional_score(
        compared_ys, baseline_ys
    )
    return baseline_compared_score - compared_baseline_score


def build_convergence_curve(
    baseline_curve: Sequence[float], compared_curve: Sequence[float]
) -> List[float]:
  """Builds a relative convergence curve (see returns for definition).

  Finds the smallest index j for each element i in 'baseline_curve'
  such that baseline_curve[i] <= compared_curve[j]. The function uses the
  'bisect_left' function to efficiently perform binary search under the
  assumption that 'baseline_curve' and 'compared_curve' are sorted in
  non-decreasing order.

  Args:
    baseline_curve: Baseline maximization convergence curve.
    compared_curve: Compared maximization convergence curve.

  Returns:
    A list of numbers where i-th (zero-index) element is the smallest "j" such
    that baseline_curve[i] <= compared_curve[j]
  """
  convergence_curve = []
  for value in baseline_curve:
    j = bisect.bisect_left(compared_curve, value)
    convergence_curve.append(j if j != len(compared_curve) else float('inf'))
  return convergence_curve
