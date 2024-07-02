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

"""Analyzers for BenchmarkStates for fast comparisons and statistics."""

import collections
import json
from typing import Callable, Dict, Optional, Sequence, Tuple

from absl import logging
import attrs
import numpy as np
import pandas as pd
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.benchmarks.analyzers import convergence_curve
from vizier.benchmarks import experimenters


RECORD_OBJECTIVE_KEY = 'objective'


@attrs.define(init=True, kw_only=True)
class PlotElement:
  """PlotElement with relevant information for a subplot."""

  curve: Optional[convergence_curve.ConvergenceCurve] = attrs.field(
      default=None,
      validator=attrs.validators.optional(
          attrs.validators.instance_of(convergence_curve.ConvergenceCurve)
      ),
  )
  plot_array: Optional[np.ndarray] = attrs.field(
      default=None,
      validator=attrs.validators.optional(
          attrs.validators.instance_of(np.ndarray)
      ),
  )
  # Error-bar uses curve, whereas histogram/scatter uses plot_array.
  plot_type: str = attrs.field(
      default='error-bar',
      validator=attrs.validators.in_(['error-bar', 'histogram', 'scatter']),
  )
  xlabel: str = attrs.field(
      default='Num Trials', validator=attrs.validators.instance_of(str)
  )
  yscale: str = attrs.field(
      default='linear',
      validator=attrs.validators.in_(['linear', 'symlog', 'logit']),
  )
  # Lower and upper percentiles to display for error bar.
  percentile_error_bar: Tuple[int, int] = attrs.field(
      default=(25, 75),
      validator=attrs.validators.deep_iterable(
          member_validator=attrs.validators.instance_of(int),
          iterable_validator=attrs.validators.instance_of(tuple),
      ),
  )


# Stores all relevant information and plots for a specific BenchmarkState.
@attrs.define(init=True, kw_only=True)
class BenchmarkRecord:
  algorithm: str = attrs.field(
      default='',
      validator=attrs.validators.instance_of(str),
  )
  experimenter_metadata: vz.Metadata = attrs.field(
      factory=vz.Metadata, validator=attrs.validators.instance_of(vz.Metadata)
  )
  plot_elements: Dict[str, PlotElement] = attrs.field(factory=dict)


class BenchmarkStateAnalyzer:
  """Analyzer for BenchmarkStates."""

  @classmethod
  def to_curve(
      cls,
      states: list[benchmarks.BenchmarkState],
      flip_signs_for_min: bool = False,
      reference_value: Optional[np.ndarray] = None,
  ) -> convergence_curve.ConvergenceCurve:
    """Generates a ConvergenceCurve from a batch of BenchmarkStates.

    Each state in batch should represent the same study (different repeat).

    Args:
      states: List of BenchmarkStates.
      flip_signs_for_min: If true, flip signs of curve when it is MINIMIZE
        metric.
      reference_value: Reference value for multiobjective hypervolume curve.

    Returns:
      Convergence curve with batch size equal to length of states.

    Raises:
      ValueError: When problem statements are not the same or is multiobjective.
    """
    if not states:
      raise ValueError('Empty States.')

    problem_statement = states[0].experimenter.problem_statement()

    curves = []
    for state in states:
      if problem_statement != state.experimenter.problem_statement():
        raise ValueError(
            f'States must have same problem {problem_statement}'
            f' and {state.experimenter.problem_statement()}'
        )
      state_trials = state.algorithm.supporter.GetTrials()

      if problem_statement.is_single_objective:
        kwargs = {'flip_signs_for_min': flip_signs_for_min}
      else:
        kwargs = {'reference_value': reference_value}

      converter = (
          convergence_curve.MultiMetricCurveConverter.from_metrics_config(
              problem_statement.metric_information,
              **kwargs,
          )
      )
      curve = converter.convert(state_trials)
      curves.append(curve)
    return convergence_curve.ConvergenceCurve.align_xs(curves)[0]

  @classmethod
  def to_record(
      cls,
      algorithm: str,
      experimenter_factory: experimenters.SerializableExperimenterFactory,
      states: list[benchmarks.BenchmarkState],
      flip_signs_for_min: bool = False,
  ) -> BenchmarkRecord:
    """Generates a BenchmarkRecord from a batch of BenchmarkStates.

    Each state in batch should represent the same study (different repeat).

    Args:
      algorithm: Algorithm name.
      experimenter_factory: Factory used for running BenchmarkState.
      states: List of BenchmarkStates.
      flip_signs_for_min: If true, flip signs of curve when it is MINIMIZE
        metric.

    Returns:
      BenchmarkRecord.
    """
    plot_elements = {}
    plot_elements[RECORD_OBJECTIVE_KEY] = PlotElement(
        curve=cls.to_curve(states, flip_signs_for_min=flip_signs_for_min),
        yscale='symlog',
    )
    return BenchmarkRecord(
        algorithm=algorithm,
        experimenter_metadata=experimenter_factory.dump(),
        plot_elements=plot_elements,
    )


def summarize_element(element: PlotElement) -> np.ndarray:
  if element.plot_type == 'error-bar':
    assert element.curve is not None
    return np.median(np.array(element.curve.ys[:, -1]))
  elif element.plot_type == 'histogram':
    return np.median(element.plot_array)
  else:
    raise NotImplementedError(f'Unsupported plot type {element.plot_type}')


def summarize_elements(
    plot_elements_list: Sequence[PlotElement],
) -> PlotElement:
  summary_vector = [
      summarize_element(element) for element in plot_elements_list
  ]
  return PlotElement(plot_array=np.array(summary_vector), plot_type='histogram')


class BenchmarkRecordAnalyzer:
  """Analyzer for a sequence of Benchmark Records."""

  @classmethod
  def add_comparison_metrics(
      cls,
      records: Sequence[BenchmarkRecord],
      baseline_algo: str,
      *,
      compare_metric: str = RECORD_OBJECTIVE_KEY,
      comparator_factory: convergence_curve.ConvergenceComparatorFactory = convergence_curve.LogEfficiencyConvergenceCurveComparator,
  ) -> list[BenchmarkRecord]:
    """Adds comparison scores as metrics via PlotElements to BenchmarkRecord.

    Comparisons are done for compare_metric with respect to the baseline_algo.

    Args:
      records: Sequence of BenchmarkRecords
      baseline_algo: Baseline algorithm to be compared against.
      compare_metric: Metric of comparison.
      comparator_factory: Comparator used for scoring.

    Returns:
      List of BenchmarkRecords with comparison scores added as metrics.

    Raises:
      ValueError: When baseline_algo cannot be found in records or the metric
      of comparison does not correspond to a curve.
    """
    records_list = [
        (rec.algorithm, json.dumps(dict(rec.experimenter_metadata)), rec)
        for rec in records
    ]
    df = pd.DataFrame(
        records_list, columns=['algorithm', 'experimenter', 'record']
    )

    analyzed_records = []
    for experimenter_key, experimenter_group in df.groupby('experimenter'):
      # Checks and stores the mapping from algorithm to plot_elements
      algo_to_elements_dict = {}
      for algorithm_name, group in experimenter_group.groupby('algorithm'):
        if not group.size:
          continue
        if len(group) != 1:
          output_str = (
              f'Condense {len(group)} records for {algorithm_name} with exptr '
              f' {experimenter_key} before applying comparisons for'
              f' {group}'
          )
          logging.error('%s', output_str)
          continue
        algo_to_elements_dict[algorithm_name] = group.record.iloc[
            0
        ].plot_elements

      # Finds the baseline algorithm and the comparison element.
      if compare_metric not in algo_to_elements_dict[baseline_algo]:
        raise ValueError(
            f'Compare metric {compare_metric} not in baseline {baseline_algo}'
        )
      baseline_element = algo_to_elements_dict[baseline_algo][compare_metric]
      if baseline_element.plot_type != 'error-bar':
        raise ValueError(
            f'No comparison can be done for {compare_metric} since'
            f' plot type is {baseline_element.plot_type}'
        )

      # Attempts to apply comparison and add comparison metrics.
      for algorithm_name, elems_dict in algo_to_elements_dict.items():
        compared_element = elems_dict[compare_metric]
        comparator = comparator_factory(
            baseline_curve=baseline_element.curve,
            compared_curve=compared_element.curve,
        )
        try:
          dict_key = (
              compare_metric + f':{comparator.name}_curve:' + baseline_algo
          )
          elems_dict[dict_key] = PlotElement(
              curve=comparator.curve(),
              plot_type='error-bar',
          )
        except Exception as e:  # pylint: disable=broad-exception-caught
          output_str = (
              f'Skip comparing curve for algo {algorithm_name} with'
              f' {compared_element} \n due to error {e}'
          )
          logging.error('%s', output_str)
        try:
          dict_key = compare_metric + f':{comparator.name}:' + baseline_algo
          elems_dict[dict_key] = PlotElement(
              plot_array=np.asarray([comparator.score()]),
              plot_type='histogram',
          )
        except Exception as e:  # pylint: disable=broad-exception-caught
          output_str = (
              f'Skip comparing score for algo {algorithm_name} with '
              f'  {compared_element} \n due to error {e}'
          )
          logging.error('%s', output_str)
        analyzed_records.append(
            BenchmarkRecord(
                algorithm=algorithm_name,
                experimenter_metadata=vz.Metadata(json.loads(experimenter_key)),  # pytype: disable=wrong-arg-types  # pandas-drop-duplicates-overloads
                plot_elements=elems_dict,
            )
        )

    return analyzed_records

  @classmethod
  def summarize(
      cls,
      records: Sequence[BenchmarkRecord],
      record_to_reduced_keys: Callable[[BenchmarkRecord], tuple[str, str]],
      *,
      summarize_elements_fn: Callable[
          [Sequence[PlotElement]], PlotElement
      ] = summarize_elements,
  ) -> Sequence[BenchmarkRecord]:
    """Summarizes PlotElements to BenchmarkRecord.

    Comparisons are done for compare_metric with respect to the baseline_algo.

    Args:
      records: Sequence of BenchmarkRecords
      record_to_reduced_keys: Takes a BenchmarkRecord and returns the removed
        key as string and the reduced metadata as a json string. For example, if
        metadata = {'experimenter': Sphere, 'dim': 4} and we want to reduce
        across experimenter, then the removed key would be Sphere and the
        reduced metadata would be '{"dim": 4}'.
      summarize_elements_fn: PlotElement summarization function.

    Returns:
      List of summarized BenchmarkRecords.
    """

    records_list = [
        (rec.algorithm, *record_to_reduced_keys(rec), rec) for rec in records
    ]
    df = pd.DataFrame(
        records_list,
        columns=['algorithm', 'removed_key', 'reduced_metadata', 'record'],
    )

    summarized_records = []
    for algorithm_name, group_by_algo in df.groupby('algorithm'):
      if not group_by_algo.size:
        continue
      for reduced_metadata, group_by_reduced_metadata in group_by_algo.groupby(
          'reduced_metadata'
      ):
        if not group_by_reduced_metadata.size:
          continue
        summarized_records.append(
            BenchmarkRecord(
                algorithm=str(algorithm_name),
                experimenter_metadata=vz.Metadata(json.loads(reduced_metadata)),
                plot_elements=cls._summarize_elements_df(
                    group_by_reduced_metadata,
                    summarize_elements_fn=summarize_elements_fn,
                ),
            )
        )
    return summarized_records

  @classmethod
  def _summarize_elements_df(
      cls,
      elements_df: pd.DataFrame,
      summarize_elements_fn: Callable[
          [Sequence[PlotElement]], PlotElement
      ] = summarize_elements,
  ) -> dict[str, PlotElement]:
    """Summarizes PlotElements via removed_key in a dataframe."""
    key_to_elements_list = collections.defaultdict(list)
    for removed_key, group in elements_df.groupby('removed_key'):
      if len(group) > 1:
        raise ValueError(f'More than 1 group after analysis for {removed_key}')
      for key, element in group.record.iloc[0].plot_elements.items():
        key_to_elements_list[key].append(element)

    # Summarize metadata and plot elements
    summarized_plot_elements = {}
    for key, elements_list in key_to_elements_list.items():
      summarized_plot_elements[key] = summarize_elements_fn(elements_list)
    return summarized_plot_elements
