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

"""Scalarizing Designer for MOO via reducing to a scalarized objective."""
import copy
from typing import Optional, Protocol, Sequence

import attr
import jax
from jax import numpy as jnp
from jax.typing import ArrayLike
from vizier import algorithms as vza
from vizier import pyvizier as vz


# TODO Separate into another file/folder.
class Scalarization(Protocol):
  """Reduces an array of objectives to a single float.

  Assumes all objectives are for MAXIMIZATION.
  """

  def __call__(self, objectives: ArrayLike) -> jax.Array:
    pass


@attr.define(init=True)
class HyperVolumeScalarization(Scalarization):
  """HyperVolume Scalarization."""

  weights: ArrayLike = attr.ib()

  reference_point: Optional[ArrayLike] = attr.ib(default=None, kw_only=True)

  def __attrs_post_init__(self):
    if any(self.weights <= 0):
      raise ValueError(f'Non-positive weights {self.weights}')

  def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
      self,
      objectives: ArrayLike,
  ) -> jax.Array:
    # Uses scalarizations in https://arxiv.org/abs/2006.04655 for
    # non-convex multiobjective optimization.
    if self.reference_point is not None:
      return jnp.min((objectives - self.reference_point) / self.weights)
    else:
      return jnp.min(objectives / self.weights)


class ScalarizingDesigner(vza.Designer):
  """Multiobjective Scalarizing Designer.

  This designer applies multiobjective optimization by reducing to
  single objective optimization via scalarization.
  """

  def __init__(
      self,
      problem_statement: vz.ProblemStatement,
      designer_factory: vza.DesignerFactory[vza.Designer],
      scalarization: Scalarization,
      *,
      seed: Optional[int] = None,
  ):
    """Init.

    Args:
      problem_statement: Must be a mulitobjective search space.
      designer_factory: Factory to create the single-objective designer.
      scalarization: Scalarization to be applied to objective metrics.
      seed: Any valid seed for factory.
    """
    self._scalarization = scalarization
    self._objectives = problem_statement.metric_information.of_type(
        vz.MetricType.OBJECTIVE
    )
    if len(self._objectives) <= 1:
      raise ValueError(f'Problem should be multi-objective {self._objectives}')

    # Create a single-objective designer.
    self._scalarized_metric_name = 'scalarized'
    single_objective_metric = problem_statement.metric_information.exclude_type(
        vz.MetricType.OBJECTIVE
    )
    single_objective_metric.append(
        vz.MetricInformation(
            name=self._scalarized_metric_name,
            goal=vz.ObjectiveMetricGoal.MAXIMIZE,
        )
    )
    self._problem_statement = copy.deepcopy(problem_statement)
    self._problem_statement.metric_information = single_objective_metric
    self._designer = designer_factory(self._problem_statement, seed=seed)

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    for trial in completed.trials:
      objectives = [
          trial.final_measurement.metrics.get_value(
              config.name, default=jnp.nan
          )
          for config in self._objectives
      ]
      # Simply append the scalarized value.
      trial.final_measurement.metrics[self._scalarized_metric_name] = (
          self._scalarization(jnp.array(objectives))
      )

    self._designer.update(completed, all_active)

  def suggest(
      self, count: Optional[int] = None
  ) -> Sequence[vz.TrialSuggestion]:
    """Make new suggestions.

    Args:
      count: Makes best effort to generate this many suggestions. If None,
        suggests as many as the algorithm wants.

    Returns:
      New suggestions.
    """
    return self._designer.suggest(count)
