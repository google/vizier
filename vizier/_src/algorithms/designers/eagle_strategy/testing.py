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

"""Testing utils for Eagle designer."""
from typing import Optional

import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy_utils
from vizier.benchmarks import experimenters

EagleStrategyDesiger = eagle_strategy.EagleStrategyDesigner
FireflyAlgorithmConfig = eagle_strategy.FireflyAlgorithmConfig
EagleStrategyUtils = eagle_strategy_utils.EagleStrategyUtils
FireflyPool = eagle_strategy_utils.FireflyPool
Firefly = eagle_strategy_utils.Firefly


def create_fake_trial(
    parent_fly_id: int,
    x_value: float,
    obj_value: Optional[float],
) -> vz.Trial:
  """Create a fake completed trial ('obj_value' = None means infeasible trial)."""
  trial = vz.Trial()
  measurement = vz.Measurement(
      metrics={
          eagle_strategy_utils.OBJECTIVE_NAME: vz.Metric(
              value=obj_value or float('inf')
          )
      }
  )
  trial.parameters['x'] = x_value
  trial.complete(
      measurement,
      inplace=True,
      infeasibility_reason='infeasible' if obj_value is None else None,
  )
  trial.metadata.ns('eagle')['parent_fly_id'] = str(parent_fly_id)
  return trial


def create_fake_problem_statement() -> vz.ProblemStatement:
  """Create a fake problem statement."""
  problem = vz.ProblemStatement()
  problem.search_space.root.add_float_param('x', 0.0, 10.0)
  problem.metric_information.append(
      vz.MetricInformation(
          name=eagle_strategy_utils.OBJECTIVE_NAME,
          goal=vz.ObjectiveMetricGoal.MAXIMIZE,
      )
  )
  return problem


def create_fake_fly(
    parent_fly_id: int,
    x_value: float,
    obj_value: Optional[float],
) -> Firefly:
  """Create a fake firefly with a fake completed trial."""
  trial = create_fake_trial(parent_fly_id, x_value, obj_value)
  return Firefly(id_=parent_fly_id, perturbation=1.0, generation=1, trial=trial)


def create_fake_empty_firefly_pool(capacity: int = 10) -> FireflyPool:
  """Create a fake empty Firefly pool."""
  problem = create_fake_problem_statement()
  # By default incorporating infeasible trials is disabled; setting it manually.
  config = FireflyAlgorithmConfig(infeasible_force_factor=0.1)
  rng = np.random.default_rng(0)
  utils = EagleStrategyUtils(problem_statement=problem, config=config, rng=rng)
  return FireflyPool(utils, capacity)


def create_fake_populated_firefly_pool(
    *,
    capacity: int,
    x_values: Optional[list[float]] = None,
    obj_values: Optional[list[Optional[float]]] = None,
    parent_fly_ids: Optional[list[int]] = None,
) -> FireflyPool:
  """Create a fake populated Firefly pool with a given capacity."""
  firefly_pool = create_fake_empty_firefly_pool(capacity=capacity)
  rng = np.random.default_rng(0)
  if not x_values:
    x_values = [float(x) for x in rng.uniform(low=0, high=10, size=(5,))]
  if not obj_values:
    obj_values = [
        float(o) for o in rng.uniform(low=-1.5, high=1.5, size=(len(x_values),))
    ]
  if not parent_fly_ids:
    parent_fly_ids = list(range(len(obj_values)))

  if not len(obj_values) == len(x_values) == len(parent_fly_ids):
    raise ValueError('Length of obj_values, ')

  for parent_fly_id, x_value, obj_value in zip(
      parent_fly_ids, x_values, obj_values
  ):
    firefly = create_fake_fly(
        parent_fly_id=parent_fly_id, x_value=x_value, obj_value=obj_value
    )
    # pylint: disable=protected-access
    firefly_pool._pool[parent_fly_id] = firefly

  # pylint: disable=protected-access
  firefly_pool._max_fly_id = capacity
  return firefly_pool


def create_fake_empty_eagle_designer() -> EagleStrategyDesiger:
  """Create a fake empty eagle designer."""
  problem = create_fake_problem_statement()
  return EagleStrategyDesiger(problem_statement=problem)


def create_fake_populated_eagle_designer(
    *,
    x_values: Optional[list[float]] = None,
    obj_values: Optional[list[Optional[float]]] = None,
    parent_fly_ids: Optional[list[int]] = None,
) -> EagleStrategyDesiger:
  """Create a fake populated eagle designer."""
  problem = create_fake_problem_statement()
  eagle_designer = EagleStrategyDesiger(problem_statement=problem)
  # pylint: disable=protected-access
  pool_capacity = eagle_designer._firefly_pool._capacity
  # Override the eagle designer's firefly pool with a populated firefly pool.
  eagle_designer._firefly_pool = create_fake_populated_firefly_pool(
      capacity=pool_capacity,
      x_values=x_values,
      obj_values=obj_values,
      parent_fly_ids=parent_fly_ids,
  )
  return eagle_designer


def create_continuous_exptr(func, dim=6):
  problem = experimenters.bbob.DefaultBBOBProblemStatement(dim)
  rng = np.random.default_rng(0)
  shift = rng.uniform(low=-2.0, high=2.0, size=(dim,))
  return experimenters.ShiftingExperimenter(
      exptr=experimenters.NumpyExperimenter(func, problem), shift=shift
  )


def create_continuous_log_scale_exptr(func, dim=6):
  problem = experimenters.bbob.DefaultBBOBProblemStatement(
      dim,
      scale_type=vz.ScaleType.LOG,
      min_value=1e1,
      max_value=1e4,
  )
  rng = np.random.default_rng(0)
  shift = rng.uniform(low=-1e2, high=0.0, size=(dim,))
  return experimenters.ShiftingExperimenter(
      exptr=experimenters.NumpyExperimenter(func, problem), shift=shift
  )


def create_categorical_exptr(num_params: int = 9, num_feasible_values: int = 5):
  return experimenters.L1CategorialExperimenter(
      num_categories=[num_feasible_values] * num_params, verbose=True
  )
