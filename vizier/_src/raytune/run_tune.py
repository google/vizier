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

"""Running Ray Tuners: https://docs.ray.io/en/latest/ray-air/tuner.html."""
from typing import Any, Optional

import numpy as np
from ray import air
from ray import tune
from vizier._src.raytune import converters
from vizier.benchmarks import experimenters


def run_tune_bbob(
    function_name: str,
    dimension: int,
    shift: np.ndarray,
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
  bbob_factory = experimenters.BBOBExperimenterFactory(
      name=function_name, dim=dimension
  )
  experimenter_factory = experimenters.SingleObjectiveExperimenterFactory(
      base_factory=bbob_factory, shift=shift
  )
  experimenter = experimenter_factory()
  problem = experimenter.problem_statement()

  param_space = converters.SearchSpaceConverter.to_dict(problem.search_space)
  objective = converters.ExperimenterConverter.to_callable(experimenter)

  def objective_fn(config: dict[str, Any]):
    # Config contains parameter names to values.
    result_dict = objective(config)
    air.session.report(result_dict)

  tuner = tune.Tuner(
      objective_fn,
      param_space=param_space,
      run_config=run_config,
      tune_config=tune_config,
  )
  return tuner.fit()
