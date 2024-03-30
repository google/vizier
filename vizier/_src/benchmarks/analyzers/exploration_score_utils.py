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

"""Utilities for computing exploration scores."""

from typing import Iterable, Optional

import numpy as np
import scipy
from vizier import pyvizier as vz

BenchmarkResults = dict[str, dict[str, dict[int, vz.ProblemAndTrials]]]


def compute_parameter_entropy(
    parameter_config: vz.ParameterConfig,
    parameter_values: Iterable[Optional[vz.ParameterValue]],
) -> float:
  """Computes the entropy of parameter values.

  Args:
    parameter_config: The parameter config.
    parameter_values: Values of a parameter.

  WARNING: Entropy estimation accuracy depends on the sample size, so to compare
  the entropies of two `parameter_values`, make sure they have the same size.

  Returns:
    The entropy of parameter values.
  """
  values = [pv.value for pv in parameter_values if pv is not None]
  if not values:
    return 0.0
  if parameter_config.type in [
      vz.ParameterType.CATEGORICAL,
      vz.ParameterType.DISCRETE,
  ] and hasattr(parameter_config, 'feasible_values'):
    if any([value not in parameter_config.feasible_values for value in values]):
      raise ValueError(
          f'Parameter values: {parameter_values} contain out-of-bound values.'
          f' Feasible values: {parameter_config.feasible_values}'
      )
    _, counts = np.unique(values, return_counts=True)
  elif hasattr(parameter_config, 'bounds'):
    min_val = parameter_config.bounds[0]
    max_val = parameter_config.bounds[1]
    if any([value < min_val or value > max_val for value in values]):
      raise ValueError(
          f'Parameter values: {parameter_values} contain out-of-bound values.'
          f' Bound: [{min_val}, {max_val}]'
      )
    if parameter_config.type == vz.ParameterType.INTEGER:
      _, counts = np.unique(values, return_counts=True)
    else:
      # Sets the number of fixed-width bins as c * sample_size ** (1.0 / 3.0).
      # The cubic-root dependency on the sample size appears in several common
      # bin-size selection strategies, e.g.
      # https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
      # The multiplier `c` is chosen such that for a small sample size, say 100,
      # we still get a reasonable number of bins, say 30.
      alpha = 1.0 / 3.0
      c = 30.0 / (100**alpha)
      n_sample = len(values)
      # We also ensure that the number of bins is at most the sample size.
      num_bins = np.min((int(c * n_sample**alpha), n_sample))
      counts, _ = np.histogram(
          values,
          bins=np.linspace(
              min_val,
              max_val,
              num=num_bins + 1,
              dtype=np.float32,
          ),
      )
  else:
    raise ValueError(
        'Invalid parameter config: either `feasible_values` or'
        '`bounds` is expected to be set, but both are unset. '
        f'Parameter config: {parameter_config}'
    )
  return float(scipy.stats.entropy(counts))


def compute_average_marginal_parameter_entropy(
    results: BenchmarkResults,
) -> float:
  """Computes the average marginal parameter entropy across results.

  Computes the marginal entropy of every parameter in every study, and then
  returns the average marginal entropy over all parameters and all studies.

  Args:
    results: Benchmark results.

  Returns:
    Average marginal parameter entropy.
  """
  marginal_param_entropies = []
  for _, spec_gen_results in results.items():
    for _, spec_results in spec_gen_results.items():
      for _, study in spec_results.items():
        for param_config in study.problem.search_space.parameters:
          param_values = [
              trial.parameters.get(param_config.name) for trial in study.trials
          ]
          marginal_param_entropies.append(
              compute_parameter_entropy(
                  parameter_config=param_config, parameter_values=param_values
              )
          )

  return np.mean(marginal_param_entropies)
