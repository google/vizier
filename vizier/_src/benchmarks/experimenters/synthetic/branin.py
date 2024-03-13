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

"""Branin function."""
from typing import Sequence

import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter
from vizier._src.benchmarks.experimenters import numpy_experimenter


def _branin(x: np.ndarray) -> float:
  """Branin function.

  This function can accept batch shapes, although it is typed to return floats
  to conform to NumpyExperimenter API.

  Args:
    x: Shape (B*, 2) array.

  Returns:
    Shape (B*) array.
  """
  a = 1
  b = 5.1 / (4 * np.pi**2)
  c = 5 / np.pi
  r = 6
  s = 10
  t = 1 / (8 * np.pi)
  x1 = x[..., 0]
  x2 = x[..., 1]

  y = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
  return y


class Branin2DExperimenter(experimenter.Experimenter):
  """2D minimization function. See https://www.sfu.ca/~ssurjano/branin.html."""

  def __init__(self):
    self._impl = numpy_experimenter.NumpyExperimenter(
        _branin, self.problem_statement()
    )

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    self._impl.evaluate(suggestions)

  def problem_statement(self) -> vz.ProblemStatement:
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param("x1", -5, 10)
    problem.search_space.root.add_float_param("x2", 0, 15)
    problem.metric_information.append(
        vz.MetricInformation(name="value", goal=vz.ObjectiveMetricGoal.MINIMIZE)
    )
    return problem
