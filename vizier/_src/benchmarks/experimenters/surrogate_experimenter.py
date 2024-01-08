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

"""Uses a predictor as a surrogate model for evaluations."""
import copy
from typing import Sequence

import jax
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter


class PredictorExperimenter(experimenter.Experimenter):
  """Uses predictor as a surrogate for objective value."""

  def __init__(
      self,
      predictor: vza.Predictor,
      problem_statement: vz.ProblemStatement,
      seed: int = 0,
  ):
    """Init.

    Args:
      predictor: Surrogate model for extrapolating unseen trials.
      problem_statement: Original problem statement for suggestions.
      seed: RNG seed for predictor.
    """
    self._predictor = predictor
    self._problem_statement = problem_statement
    self._rng = jax.random.PRNGKey(seed)
    self._objective_name = self._problem_statement.single_objective_metric_name

  def evaluate(self, suggestions: Sequence[vz.Trial]):
    prediction = self._predictor.predict(suggestions, self._rng)
    for i, suggestion in enumerate(suggestions):
      evaluation = prediction.mean[i]
      suggestion.complete(
          vz.Measurement(metrics={self._objective_name: evaluation})
      )

  def problem_statement(self) -> vz.ProblemStatement:
    return copy.deepcopy(self._problem_statement)

  def __repr__(self) -> str:
    return (
        f'PredictorExperimenter on problem {self._problem_statement} with'
        f' {self._predictor}'
    )
