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

"""Scalarizing Designer for MOO via reducing to a scalarized objective."""
import copy
import random
from typing import Optional, Sequence

from jax import numpy as jnp
from jax import random as jax_random
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import scalarization
from vizier._src.algorithms.ensemble import ensemble_designer


class ScalarizingDesigner(vza.Designer):
  """Multiobjective Scalarizing Designer.

  This designer applies multiobjective optimization by reducing to
  single objective optimization via scalarization.
  """

  def __init__(
      self,
      problem_statement: vz.ProblemStatement,
      designer_factory: vza.DesignerFactory[vza.Designer],
      scalarizer: scalarization.Scalarization,
      *,
      seed: Optional[int] = None,
  ):
    """Init.

    Args:
      problem_statement: Must be a mulitobjective search space.
      designer_factory: Factory to create the single-objective designer.
      scalarizer: Scalarization to be applied to objective metrics.
      seed: Any valid seed for factory.
    """
    self._scalarizer = scalarizer
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
          trial.final_measurement_or_die.metrics.get_value(
              config.name, default=jnp.nan
          )
          for config in self._objectives
      ]
      # Simply append the scalarized value.
      trial.final_measurement_or_die.metrics[self._scalarized_metric_name] = (
          self._scalarizer(jnp.array(objectives))
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


def create_gaussian_scalarizing_designer(
    problem_statement: vz.ProblemStatement,
    designer_factory: vza.DesignerFactory[vza.Designer],
    scalarization_factory: scalarization.ScalarizationFromWeights,
    num_ensemble: int,
    *,
    seed: Optional[int] = None,
) -> vza.Designer:
  """Factory for an Ensemble of Gaussian weighted scalarized Designers."""
  objectives = problem_statement.metric_information.of_type(
      vz.MetricType.OBJECTIVE
  )
  if len(objectives) <= 1:
    raise ValueError(
        'Problem should be multi-objective for applying '
        f'scalarization ensembling {objectives}'
    )

  key = jax_random.PRNGKey(seed or random.getrandbits(32))
  weights = abs(
      jax_random.normal(key=key, shape=(num_ensemble, len(objectives)))
  )
  weights /= jnp.linalg.norm(weights, axis=1)[..., jnp.newaxis]
  ensemble_dict = {}
  for weight in weights:
    ensemble_dict[f'scalarized_weight: {weight}'] = ScalarizingDesigner(
        problem_statement=problem_statement,
        designer_factory=designer_factory,
        scalarizer=scalarization_factory(weight),
        seed=seed,
    )
  return ensemble_designer.EnsembleDesigner(ensemble_dict)
