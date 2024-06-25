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

"""Multi-objective synthetic functions created by Deb et al.

Forked from
https://github.com/pytorch/botorch/blob/main/botorch/test_functions/multi_objective.py.
"""

from typing import Callable, Sequence
import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter
from vizier.pyvizier import converters


# TODO: Subclass MultiObjectiveNumpyExperimenter.
class DHExperimenter(experimenter.Experimenter):
  """Multiobjective problem with two objectives (f0, f1) and d-dimensions.

  The functions are defined as:
    f0(x) = x0
    f1(x) = h(x) + g(x) * s(x) for DH1 and DH2
    f1(x) = h(x) * (g(x) + s(x)) for DH3 and DH4

  Reference:
    K. Deb and H. Gupta. Searching for Robust Pareto-Optimal Solutions in
    Multi-objective Optimization. Evolutionary Multi-Criterion Optimization,
    Springer-Berlin, pp. 150-164, 2005.
  """

  def __init__(
      self,
      h_fn: Callable[[np.ndarray], float],
      g_fn: Callable[[np.ndarray], float],
      s_fn: Callable[[np.ndarray], float],
      f1_fn: Callable[[float, float, float], float],  # Uses h, g, s.
      bounds: Sequence[tuple[float, float]],
  ):
    self._h_fn = h_fn
    self._g_fn = g_fn
    self._s_fn = s_fn
    self._f1_fn = f1_fn
    self._bounds = bounds

    self._converter = converters.TrialToArrayConverter.from_study_config(
        study_config=self.problem_statement(),
        scale=False,
        flip_sign_for_minimization_metrics=False,
    )

  def problem_statement(self) -> vz.ProblemStatement:
    problem = vz.ProblemStatement()
    problem.metric_information.append(
        vz.MetricInformation(name="f0", goal=vz.ObjectiveMetricGoal.MINIMIZE)
    )
    problem.metric_information.append(
        vz.MetricInformation(name="f1", goal=vz.ObjectiveMetricGoal.MINIMIZE)
    )
    for i, (min_val, max_val) in enumerate(self._bounds):
      problem.search_space.root.add_float_param(f"x{i}", min_val, max_val)

    return problem

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    features = self._converter.to_features(suggestions)
    for i, suggestion in enumerate(suggestions):
      feat = features[i]
      f0 = feat[0]
      f1 = self._f1_fn(self._h_fn(feat), self._g_fn(feat), self._s_fn(feat))
      suggestion.complete(vz.Measurement(metrics={"f0": f0, "f1": f1}))

  @classmethod
  def DH1(cls, num_dimensions: int) -> "DHExperimenter":
    if num_dimensions < 2:
      raise ValueError(f"num_dimensions must be >= 2, got {num_dimensions}.")
    h_fn = lambda x: 1 - x[0] ** 2
    g_fn = lambda x: np.sum(
        10 + x[1:] ** 2 - 10 * np.cos(4 * np.pi * x[1:]),
    )
    s_fn = lambda x: 1 / (0.2 + x[0]) + x[0] ** 2
    f1_fn = lambda h, g, s: h + g * s
    bounds = [(0, 1)] + [(-1, 1) for _ in range(num_dimensions - 1)]
    return cls(h_fn, g_fn, s_fn, f1_fn, bounds)

  @classmethod
  def DH2(cls, num_dimensions: int) -> "DHExperimenter":
    """Same as DH1 but s_fn has a different (10.0) constant."""
    if num_dimensions < 2:
      raise ValueError(f"num_dimensions must be >= 2, got {num_dimensions}.")
    h_fn = lambda x: 1 - x[0] ** 2
    g_fn = lambda x: np.sum(
        10 + x[1:] ** 2 - 10 * np.cos(4 * np.pi * x[1:]),
    )
    s_fn = lambda x: 1 / (0.2 + x[0]) + 10.0 * x[0] ** 2
    f1_fn = lambda h, g, s: h + g * s
    bounds = [(0, 1)] + [(-1, 1) for _ in range(num_dimensions - 1)]
    return cls(h_fn, g_fn, s_fn, f1_fn, bounds)

  @classmethod
  def DH3(cls, num_dimensions: int) -> "DHExperimenter":
    if num_dimensions < 3:
      raise ValueError(f"num_dimensions must be >= 3, got {num_dimensions}.")
    exp_arg1 = lambda x: -(((x[1] - 0.35) / 0.25) ** 2)
    exp_arg2 = lambda x: -(((x[1] - 0.85) / 0.03) ** 2)
    h_fn = lambda x: 2 - 0.8 * np.exp(exp_arg1(x)) - np.exp(exp_arg2(x))

    g_fn = lambda x: 50 * np.sum(x[2:] ** 2)
    s_fn = lambda x: 1 - np.sqrt(x[0])
    f1_fn = lambda h, g, s: h * (g + s)
    bounds = [(0, 1), (0, 1)] + [(-1, 1) for _ in range(num_dimensions - 2)]
    return cls(h_fn, g_fn, s_fn, f1_fn, bounds)

  @classmethod
  def DH4(cls, num_dimensions: int) -> "DHExperimenter":
    """Similar to DH3 but different h_fn."""
    if num_dimensions < 3:
      raise ValueError(f"num_dimensions must be >= 3, got {num_dimensions}.")
    exp_arg1 = lambda x: -(((np.sum(x[:2]) - 0.35) / 0.25) ** 2)
    exp_arg2 = lambda x: -(((np.sum(x[:2]) - 0.85) / 0.03) ** 2)
    h_fn = lambda x: 2 - x[0] - 0.8 * np.exp(exp_arg1(x)) - np.exp(exp_arg2(x))

    g_fn = lambda x: 50 * np.sum(x[2:] ** 2)
    s_fn = lambda x: 1 - np.sqrt(x[0])
    f1_fn = lambda h, g, s: h * (g + s)
    bounds = [(0, 1), (0, 1)] + [(-1, 1) for _ in range(num_dimensions - 2)]
    return cls(h_fn, g_fn, s_fn, f1_fn, bounds)
