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

"""Converters for PyVizier with RayTune."""

from typing import Any, Dict, Callable, Union

from ray import tune
from ray.tune.search import sample
from vizier import pyvizier as vz
from vizier.benchmarks import experimenters


class SearchSpaceConverter:
  """Converts pyvizier.SearchSpace <-> RayTune Search Space."""

  @classmethod
  def to_dict(
      cls,
      search_space: vz.SearchSpace,
  ) -> Dict[str, Union[sample.Domain, sample.Sampler]]:
    """Converts PyVizier ProblemStatement to Ray search space."""
    param_space = {}

    for param in search_space.parameters:
      if param.type == vz.ParameterType.DOUBLE:
        lower, upper = param.bounds
        if param.scale_type == vz.ScaleType.LINEAR:
          param_space[param.name] = tune.uniform(lower, upper)
        elif param.scale_type == vz.ScaleType.LOG:
          param_space[param.name] = tune.loguniform(lower, upper)
        else:
          raise ValueError(f'DOUBLE scale {param.scale_type} not supported.')
      elif param.type == vz.ParameterType.INTEGER:
        lower, upper = param.bounds
        param_space[param.name] = tune.randint(lower, upper)
      else:
        feasible_values = param.feasible_values
        param_space[param.name] = tune.choice(feasible_values)
    return param_space

  @classmethod
  def to_vizier(
      cls,
      param_space: Dict[str, Any],
  ) -> vz.SearchSpace:
    """Converts from a Ray to a Vizier SearchSpace."""
    space = vz.SearchSpace()

    for param_name, param_config in param_space.items():
      # Find out if the parameter should be scaled.
      scale_type = None
      if isinstance(param_config, sample.Float):
        if isinstance(param_config.sampler, (sample.Grid, sample.Uniform)):
          scale_type = vz.ScaleType.LINEAR
        elif isinstance(param_config.sampler, sample.LogUniform):
          scale_type = vz.ScaleType.LOG
        elif isinstance(param_config.sampler, sample.Normal):
          raise ValueError(
              f'Normal sampler is not supported: {param_name}: {param_config}'
          )
        else:
          raise ValueError(
              f'Unknown sampler type encountered: {param_name}: {param_config}'
          )

      # Add the parameter to the search space.
      if isinstance(param_config, sample.Function):
        raise ValueError('Must use tune defined types. Functions not supported')
      elif isinstance(param_config, sample.Float):
        space.root.add_float_param(
            param_name,
            min_value=param_config.lower,
            max_value=param_config.upper,
            scale_type=scale_type,
        )
      elif isinstance(param_config, sample.Integer):
        space.root.add_int_param(
            param_name,
            min_value=param_config.lower,
            max_value=param_config.upper,
        )
      elif isinstance(param_config, sample.Categorical):
        if not all([isinstance(c, str) for c in param_config.categories]):
          raise ValueError('Only string values are supported for categories')
        space.root.add_categorical_param(
            param_name, feasible_values=list(map(str, param_config.categories))
        )
      else:
        raise ValueError(
            f'Unsupported config encountered: {param_name}: {param_config}'
        )
    return space


class ExperimenterConverter:
  """Converts Experimenters to Ray Trainables."""

  @classmethod
  def to_callable(
      cls,
      experimenter: experimenters.Experimenter,
  ) -> Callable[[vz.ParameterDict], Dict[str, float]]:
    """Returns a callable that takes Parameters and returns metric values."""

    def trainable(config: vz.ParameterDict) -> Dict[str, float]:
      trial = vz.Trial(parameters=config)
      experimenter.evaluate([trial])
      if trial.final_measurement is None:
        raise ValueError(f'No final_measurement on trial{trial.id}')
      return {k: v.value for k, v in trial.final_measurement.metrics.items()}

    return trainable
