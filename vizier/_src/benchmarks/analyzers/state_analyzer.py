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

from typing import List

from vizier._src.benchmarks.analyzers import convergence_curve
from vizier._src.benchmarks.runners import benchmark_state


class BenchmarkStateAnalyzer:
  """Analyzer for BenchmarkStates."""

  @classmethod
  def to_curve(
      cls,
      states: List[benchmark_state.BenchmarkState],
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
    return convergence_curve.ConvergenceCurve.align_xs(curves)
