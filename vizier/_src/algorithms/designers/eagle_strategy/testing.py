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

"""Testing utils for Eagle designer."""
from typing import List, Optional

import numpy as np
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy_utils
from vizier._src.benchmarks.experimenters import l1_categorical_experimenter
from vizier._src.benchmarks.experimenters import shifting_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob

EagleStrategyDesiger = eagle_strategy.EagleStrategyDesigner
FireflyAlgorithmConfig = eagle_strategy.FireflyAlgorithmConfig
EagleStrategyUtils = eagle_strategy_utils.EagleStrategyUtils
FireflyPool = eagle_strategy_utils.FireflyPool
Firefly = eagle_strategy_utils.Firefly
L1CategorialExperimenter = l1_categorical_experimenter.L1CategorialExperimenter


def create_fake_trial(
    parent_fly_id: int,
    x_value: float,
    obj_value: float,
) -> vz.Trial:
  """Create a fake completed trial."""
  trial = vz.Trial()
  measurement = vz.Measurement(
      metrics={eagle_strategy_utils.OBJECTIVE_NAME: vz.Metric(value=obj_value)})
  trial.parameters['x'] = x_value
  trial.complete(measurement, inplace=True)
  trial.metadata.ns('eagle')['parent_fly_id'] = str(parent_fly_id)
  return trial


def create_fake_problem_statement() -> vz.ProblemStatement:
  """Create a fake problem statement."""
  problem = vz.ProblemStatement()
  problem.search_space.root.add_float_param('x', 0.0, 10.0)
  problem.metric_information.append(
      vz.MetricInformation(
          name=eagle_strategy_utils.OBJECTIVE_NAME,
          goal=vz.ObjectiveMetricGoal.MAXIMIZE))
  return problem


def create_fake_fly(
    parent_fly_id: int,
    x_value: float,
    obj_value: float,
) -> Firefly:
  """Create a fake firefly with a fake completed trial."""
  trial = create_fake_trial(parent_fly_id, x_value, obj_value)
  return Firefly(id_=parent_fly_id, perturbation=1.0, generation=1, trial=trial)


def create_fake_empty_firefly_pool(capacity: int = 10) -> FireflyPool:
  """Create a fake empty Firefly pool."""
  problem = create_fake_problem_statement()
  config = FireflyAlgorithmConfig()
  rng = np.random.default_rng(0)
  utils = EagleStrategyUtils(problem_statement=problem, config=config, rng=rng)
  return FireflyPool(utils, capacity)


def create_fake_populated_firefly_pool(
    *,
    capacity: int,
    x_values: Optional[List[float]] = None,
    obj_values: Optional[List[float]] = None,
) -> FireflyPool:
  """Create a fake populated Firefly pool with a given capacity."""
  firefly_pool = create_fake_empty_firefly_pool(capacity=capacity)
  rng = np.random.default_rng(0)
  if not x_values:
    x_values = [float(x) for x in rng.uniform(low=0, high=10, size=(5,))]
  if not obj_values:
    obj_values = [float(o) for o in rng.uniform(low=-1.5, high=1.5, size=(5,))]
  for parent_fly_id, (obj_val, x_val) in enumerate(zip(obj_values, x_values)):
    # pylint: disable=protected-access
    firefly_pool._pool[parent_fly_id] = create_fake_fly(
        parent_fly_id=parent_fly_id, x_value=x_val, obj_value=obj_val)
  # pylint: disable=protected-access
  firefly_pool._max_fly_id = capacity
  return firefly_pool


def create_fake_empty_eagle_designer() -> EagleStrategyDesiger:
  """Create a fake empty eagle designer."""
  problem = create_fake_problem_statement()
  return EagleStrategyDesiger(problem_statement=problem)


def create_fake_populated_eagle_designer(
    *,
    x_values: Optional[List[float]] = None,
    obj_values: Optional[List[float]] = None) -> EagleStrategyDesiger:
  """Create a fake populated eagle designer."""
  problem = create_fake_problem_statement()
  eagle_designer = EagleStrategyDesiger(problem_statement=problem)
  # pylint: disable=protected-access
  pool_capacity = eagle_designer._firefly_pool.capacity
  # Override the eagle designer's firefly pool with a populated firefly pool.
  eagle_designer._firefly_pool = create_fake_populated_firefly_pool(
      x_values=x_values, obj_values=obj_values, capacity=pool_capacity)
  return eagle_designer


def create_continuous_exptr(func, dim=6):
  problem = bbob.DefaultBBOBProblemStatement(dim)
  rng = np.random.default_rng(0)
  shift = rng.uniform(low=-2.0, high=2.0, size=(dim,))
  return shifting_experimenter.ShiftingExperimenter(
      exptr=benchmarks.NumpyExperimenter(func, problem), shift=shift)


def create_categorical_exptr():
  num_categories = [5, 8, 10, 2, 15, 10, 5, 8, 12]
  return L1CategorialExperimenter(num_categories=num_categories, verbose=True)
