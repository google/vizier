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

"""Tools for visualization with PlotElements."""


import json
from typing import Optional, Sequence, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vizier import pyvizier as vz
from vizier._src.benchmarks.analyzers import state_analyzer
from vizier.utils import json_utils


def plot_median_convergence(
    ax: mpl.axes.Axes,
    curves: 'np.ndarray',
    *,
    percentiles: Sequence[Tuple[int, int]] = ((40, 60),),
    alphas: Sequence[float] = (0.2,),
    xs: Optional['np.ndarray'] = None,
    **kwargs,
):
  """Aggregates multiple convergence curves into a plot with confidence bounds.

  Example usage:
    ```python
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    plot_median_convergence(ax,
                            [[1,1,2,3,4], [1,1,1,2,nan]],
                            percentiles=((40, 60), (30, 70)),
                            alphas=(0.4, 0.2),
                            xs=np.arange(1,6),
                            color='r')
    ```

  Args:
    ax: matplotlib axis to plot on.
    curves: Expected to have shape (Number of studies, points), where rows are
      convergence curves from repeated studies from the same algorithm and
      settings. May contain NaNs, which will be excluded from plotting.
    percentiles: Each pair defines (lower_percentile, upper_percentile).
    alphas: Must have the same length as percentiles. Defines the color strength
      for the confidence bounds. Make it decrease in the distance between lower
      and upper percentiles. (See example above).
    xs: x values for the plot. If not provided, uses np.arange(curves.shape[1]).
      Must have the shape (curves.shape[1], 0)
    **kwargs: Forwared to ax.plot().
  """
  if xs is None:
    xs = np.arange(curves.shape[1])

  line = ax.plot(xs, np.nanmedian(curves, axis=0), **kwargs)
  for (lower, upper), alpha in zip(percentiles, alphas):
    ax.fill_between(
        xs,
        np.nanpercentile(curves, lower, axis=0),
        np.nanpercentile(curves, upper, axis=0),
        alpha=alpha,
        color=line[0].get_color(),
    )


def plot_mean_convergence(
    ax: mpl.axes.Axes,
    curves: 'np.ndarray',
    *,
    alpha: float = 0.2,
    xs: Optional['np.ndarray'] = None,
    **kwargs,
):
  """Aggregates multiple convergence curves into a plot with standard error bounds.

  Example usage:
    ```python
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    plot_mean_convergence(ax,
                          [[1,1,2,3,4], [1,1,1,2,nan]],
                          alpha=0.3,
                          xs=np.arange(1,6),
                          color='r')
    ```

  Args:
    ax: matplotlib axis to plot on.
    curves: Expected to have shape (Number of studies, points), where rows are
      convergence curves from repeated studies from the same algorithm and
      settings. May contain NaNs, which will be excluded from plotting.
    alpha: Defines the color strength for the standard error bounds.
    xs: x values for the plot. If not provided, uses np.arange(curves.shape[1]).
      Must have the shape (curves.shape[1], 0)
    **kwargs: Forwared to ax.plot().
  """
  if xs is None:
    xs = np.arange(curves.shape[1])
  curves_mean = np.nanmean(curves, axis=0)
  curves_std_error = np.nanstd(curves, axis=0) / np.sqrt(curves.shape[0])

  line = ax.plot(xs, curves_mean, **kwargs)
  ax.fill_between(
      xs,
      curves_mean + 1.5 * curves_std_error,
      curves_mean - 1.5 * curves_std_error,
      alpha=alpha,
      color=line[0].get_color(),
  )


def plot_from_records(
    records: Sequence[state_analyzer.BenchmarkRecord],
    metrics: Optional[Sequence[str]] = None,
    *,
    fig_title: str = 'All Plot Elements',
    title_maxlen: int = 50,
    col_figsize: float = 6.0,
    row_figsize: float = 6.0,
    **kwargs,
):
  """Generates a grid of algorithm comparison plots.

  Generates one plot for each Experimenter x Metrics in records. Note that
  each row = Experimenter and each column = Metrics.

  Args:
    records: All BenchmarkRecords used for plotting.
    metrics: Keys in the plot_elements dict in BenchmarkRecord used for plot. If
      not supplied, all keys are plotted.
    fig_title: Title of the entire grid plot.
    title_maxlen: Maximum length of title of each Experimenter.
    col_figsize: Size of the column of each subfigure.
    row_figsize: Size of the row of each subfigure.
    **kwargs: Additional keyword args forwarded to pyplot.

  Raises:
    ValueError: When plot type is not supported.
  """

  def _metadata_to_str(metadata: vz.Metadata) -> str:
    visual_dict = {}
    for _, key, value in metadata.all_items():
      try:
        loaded = json.loads(value, cls=json_utils.NumpyDecoder)
        assert isinstance(loaded, dict)
        visual_dict = visual_dict | {k: v for k, v in loaded.items() if v}
      except Exception as e:  # pylint: disable=broad-except
        del e
        visual_dict[key] = value
    return str(visual_dict)

  records_list = [
      (rec.algorithm, _metadata_to_str(rec.experimenter_metadata), rec)
      for rec in records
  ]
  df = pd.DataFrame(
      records_list, columns=['algorithm', 'experimenter', 'record']
  )

  algorithms = df.algorithm.unique()
  colors = {
      algorithm: plt.get_cmap('tab10')(i)
      for i, algorithm in enumerate(algorithms)
  }
  total_rows = len(df.groupby('experimenter'))
  if metrics is None:
    metrics = set()
    for record in df.record:
      metrics = metrics.union(set(record.plot_elements.keys()))
    print(f'All inferred metrics {metrics}')

  fig, axes = plt.subplots(
      total_rows,
      len(metrics),
      figsize=(col_figsize * len(metrics), row_figsize * total_rows),
      squeeze=False,
  )
  fig.suptitle(fig_title, fontsize=16)

  fig_idx = 0
  for experimenter_key, group_by_experimenter in df.groupby('experimenter'):
    for metric_idx, metric in enumerate(metrics):
      ax = axes[fig_idx, metric_idx]
      subplot_title = (
          str(experimenter_key)[:title_maxlen] if experimenter_key else metric
      )
      ax.set_title(subplot_title)
      ax.set_ylabel(metric)
      for algorithm_name, group in group_by_experimenter.groupby('algorithm'):
        if not group.size:
          continue
        if len(group) != 1:
          print(
              f'Found more records than expected in {algorithm_name} for'
              f' {group}'
          )
        elems = group.record.iloc[0].plot_elements
        if metric not in elems:
          print(f'metric {metric} not found in {group.record.iloc[0]}')
          continue

        elem_for_metric = elems[metric]
        plot_type = elem_for_metric.plot_type
        if plot_type == 'error-bar':
          plot_median_convergence(
              ax,
              elem_for_metric.curve.ys,
              xs=elem_for_metric.curve.xs,
              label=f'{algorithm_name}',
              color=colors[algorithm_name],
              percentiles=(elem_for_metric.percentile_error_bar,),
              **kwargs,
          )
        elif plot_type == 'scatter':
          plot = elem_for_metric.plot_array
          ax.scatter(
              plot[:, 0],
              plot[:, 1],
              label=f'{algorithm_name}',
              color=colors[algorithm_name],
              **kwargs,
          )
        elif plot_type == 'histogram':
          plot = elem_for_metric.plot_array
          linewidth = (
              len(algorithms)
              + 1
              - float(list(algorithms).index(algorithm_name))
          )
          ax.hist(
              plot,
              histtype='step',
              density=True,
              fill=False,
              linewidth=linewidth,
              label=f'{algorithm_name}',
              color=colors[algorithm_name],
              **kwargs,
          )
        else:
          raise ValueError(f'{plot_type} plot not yet supported!')
        ax.set_xlabel(elem_for_metric.xlabel)
        ax.set_yscale(elem_for_metric.yscale)
        ax.yaxis.set_major_locator(mpl.ticker.LinearLocator(20))
        ax.yaxis.set_minor_locator(mpl.ticker.LinearLocator(100))
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
      ax.legend()
    fig_idx += 1
