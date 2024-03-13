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

"""HartMann 6-Dimensional function."""

from typing import Sequence
import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter
from vizier._src.benchmarks.experimenters import numpy_experimenter


def _hartmann6d(x: np.ndarray) -> float:
  """Hartmann function.

  Args:
    x: Shape (6,) array

  Returns:
    Shape (,) array.
  """
  # For math notations:
  # pylint: disable=invalid-name
  alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
  A = np.asarray([
      [10, 3, 17, 3.5, 1.7, 8],
      [0.05, 10, 17, 0.1, 8, 14],
      [3, 3.5, 1.7, 10, 17, 8],
      [17, 8, 0.05, 10, 0.1, 14],
  ])
  P = 1e-4 * np.asarray([
      [1312, 1696, 5569, 124, 8283, 5886],
      [2329, 4135, 8307, 3736, 1004, 9991],
      [2348, 1451, 3522, 2883, 3047, 6650],
      [4047, 8828, 8732, 5743, 1091, 381],
  ])
  y = -alpha @ np.exp(-np.sum(A * (x - P) ** 2, axis=1))
  return y


class Hartmann6DExperimenter(experimenter.Experimenter):
  """6D minimization function. See https://www.sfu.ca/~ssurjano/hart6.html."""

  def __init__(self):
    self._impl = numpy_experimenter.NumpyExperimenter(
        _hartmann6d, self.problem_statement()
    )

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    self._impl.evaluate(suggestions)

  def problem_statement(self) -> vz.ProblemStatement:
    problem = vz.ProblemStatement()
    for i in range(1, 7):
      problem.search_space.root.add_float_param(f"x{i}", 0, 1)
    problem.metric_information.append(
        vz.MetricInformation(name="value", goal=vz.ObjectiveMetricGoal.MINIMIZE)
    )
    return problem


def _hartmann3d(x: np.ndarray) -> float:
  """Hartmann function.

  Args:
    x: Shape (6,) array

  Returns:
    Shape (,) array.
  """
  # For math notations:
  # pylint: disable=invalid-name
  alpha = np.asarray([1.0, 1.2, 3.0, 3.2])
  A = np.asarray([
      [3, 10, 30],
      [0.1, 10, 35],
      [3, 10, 30],
      [0.1, 10, 35],
  ])
  P = 1e-4 * np.asarray([
      [3689, 1170, 2673],
      [4699, 4387, 7470],
      [1091, 8732, 5547],
      [381, 5743, 8828],
  ])
  y = -alpha @ np.exp(-np.sum(A * (x - P) ** 2, axis=1))
  return y


class Hartmann3DExperimenter(experimenter.Experimenter):
  """3D minimization function. See https://www.sfu.ca/~ssurjano/hart3.html."""

  def __init__(self):
    self._impl = numpy_experimenter.NumpyExperimenter(
        _hartmann3d, self.problem_statement()
    )

  def evaluate(self, suggestions: Sequence[vz.Trial]) -> None:
    self._impl.evaluate(suggestions)

  def problem_statement(self) -> vz.ProblemStatement:
    problem = vz.ProblemStatement()
    for i in range(1, 4):
      problem.search_space.root.add_float_param(f"x{i}", 0, 1)
    problem.metric_information.append(
        vz.MetricInformation(name="value", goal=vz.ObjectiveMetricGoal.MINIMIZE)
    )
    return problem
