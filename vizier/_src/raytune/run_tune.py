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

"""Running Ray Tuners: https://docs.ray.io/en/latest/ray-air/tuner.html."""

from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from ray import air
from ray import data
from ray import tune
from ray.air import session
from vizier import pyvizier as vz
from vizier._src.raytune import converters
from vizier.benchmarks import experimenters


# See https://docs.ray.io/en/latest/data/dataset.html
def run_tune_distributed(
    run_tune_args_list: List[Tuple[Any]],
    run_tune: Callable[[Any], tune.result_grid.ResultGrid],
) -> List[tune.result_grid.ResultGrid]:
  """Distributes tuning via Ray datasets API for MapReduce purposes.

  NOTE: There are no datasets processed. However, all MapReduce operations
  are now done in the Datasets API under Ray.

  Args:
    run_tune_args_list: List of Tuples that are to be passed into run_tune.
    run_tune: Callable that accepts args from previous list.

  Returns:
    List of results.
  """
  ds = data.from_items([{'args_tuple': args} for args in run_tune_args_list])
  ds = ds.map(lambda x: {'result': run_tune(*x['args_tuple'])})
  return ds.take_all()


def run_tune_bbob(
    function_name: str,
    dimension: int,
    shift: Optional[np.ndarray] = None,
    tune_config: Optional[tune.TuneConfig] = None,
    run_config: Optional[air.RunConfig] = None,
) -> tune.result_grid.ResultGrid:
  """Runs Ray Tuners for BBOB problems.

  See https://docs.ray.io/en/latest/tune/key-concepts.html
  For more information on Tune and Run configs, see
  https://docs.ray.io/en/latest/ray-air/tuner.html

  Args:
    function_name: BBOB function name.
    dimension: Dimension of BBOB function.
    shift: Shift of BBOB function.
    tune_config: Ray Tune Config.
    run_config: Ray Run Config.

  Returns:
  """
  experimenter_factory = experimenters.BBOBExperimenterFactory(
      name=function_name, dim=dimension
  )
  if shift is not None:
    experimenter_factory = experimenters.SingleObjectiveExperimenterFactory(
        base_factory=experimenter_factory, shift=shift
    )
  return run_tune_from_factory(experimenter_factory, tune_config, run_config)


def run_tune_from_factory(
    experimenter_factory: experimenters.ExperimenterFactory,
    tune_config: Optional[tune.TuneConfig] = None,
    run_config: Optional[air.RunConfig] = None,
) -> tune.result_grid.ResultGrid:
  """Runs Ray Tuners from an Experimenter Factory.

  See https://docs.ray.io/en/latest/tune/key-concepts.html
  For more information on Tune and Run configs, see
  https://docs.ray.io/en/latest/ray-air/tuner.html

  Args:
    experimenter_factory: Experimenter Factory.
    tune_config: Ray Tune Config.
    run_config: Ray Run Config.

  Returns:
  """
  experimenter = experimenter_factory()
  problem = experimenter.problem_statement()

  param_space = converters.SearchSpaceConverter.to_dict(problem.search_space)
  objective = converters.ExperimenterConverter.to_callable(experimenter)

  metric_info = problem.metric_information.item()
  if tune_config is None:
    tune_config = tune.TuneConfig()
  tune_config.metric = metric_info.name
  if metric_info.goal == vz.ObjectiveMetricGoal.MINIMIZE:
    tune_config.mode = 'min'
  else:
    tune_config.mode = 'max'

  def objective_fn(config: vz.ParameterDict) -> None:
    # Config contains parameter names to values and is autopopulated for each
    # Trial. Evaluation is static for BBOB so we simply loop.
    for _ in range(tune_config.num_samples):
      result_dict = objective(config)
      session.report(result_dict)

  tuner = tune.Tuner(
      objective_fn,
      param_space=param_space,
      run_config=run_config,
      tune_config=tune_config,
  )
  return tuner.fit()
