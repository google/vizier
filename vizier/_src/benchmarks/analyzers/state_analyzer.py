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

"""Analyzers for BenchmarkStates for fast comparisons and statistics."""

from typing import Dict, List, Optional, Tuple
import attrs
import numpy as np
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.benchmarks.analyzers import convergence_curve
from vizier.benchmarks import experimenters


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
      states: List[benchmarks.BenchmarkState],
      flip_signs_for_min: bool = False,
  ) -> convergence_curve.ConvergenceCurve:
    """Generates a ConvergenceCurve from a batch of BenchmarkStates.

    Each state in batch should represent the same study (different repeat).

    Args:
      states: List of BenchmarkStates.
      flip_signs_for_min: If true, flip signs of curve when it is MINIMIZE
        metric.

    Returns:
      Convergence curve with batch size equal to length of states.

    Raises:
      ValueError: When problem statements are not the same or is multiobjective.
    """
    if not states:
      raise ValueError('Empty States.')

    problem_statement = states[0].experimenter.problem_statement()
    if not problem_statement.is_single_objective:
      raise ValueError('Multiobjective Conversion not supported yet.')

    converter = convergence_curve.ConvergenceCurveConverter(
        problem_statement.metric_information.item(),
        flip_signs_for_min=flip_signs_for_min,
    )
    curves = []
    for state in states:
      if problem_statement != state.experimenter.problem_statement():
        raise ValueError(
            f'States must have same problem {problem_statement}'
            f' and {state.experimenter.problem_statement()}'
        )

      state_trials = state.algorithm.supporter.GetTrials()
      curve = converter.convert(state_trials)
      curves.append(curve)
    return convergence_curve.ConvergenceCurve.align_xs(curves)[0]

  @classmethod
  def to_record(
      cls,
      algorithm: str,
      experimenter_factory: experimenters.SerializableExperimenterFactory,
      states: List[benchmarks.BenchmarkState],
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
    objective_key = 'objective'
    plot_elements[objective_key] = PlotElement(
        curve=cls.to_curve(states, flip_signs_for_min=flip_signs_for_min)
    )
    return BenchmarkRecord(
        algorithm=algorithm,
        experimenter_metadata=experimenter_factory.dump(),
        plot_elements=plot_elements,
    )
