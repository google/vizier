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

"""Randomization related utils."""

import logging
from typing import Dict, List, TypeVar

import numpy as np
from vizier import pyvizier as vz

_T = TypeVar('_T')


def sample_uniform(rng: np.random.Generator, min_value=0, max_value=1) -> float:
  """Samples unifrom value and udpate key."""
  return float(rng.uniform(low=min_value, high=max_value))


def sample_bernoulli(
    rng: np.random.Generator,
    prob1: float,
    value1: _T = 0,
    value2: _T = 1,
) -> _T:
  """Samples value1 with probability prob1."""
  return value1 if rng.binomial(1, p=prob1) else value2


def sample_integer(
    rng: np.random.Generator,
    min_value: float,
    max_value: float,
) -> int:
  """Samples a random integer."""
  val = sample_uniform(rng, min_value, max_value)
  return round(val)


def sample_categorical(rng: np.random.Generator, categories: List[str]) -> str:
  """Samples a random categorical value."""
  return rng.choice(categories)


def sample_discrete(rng: np.random.Generator,
                    feasible_points: List[float]) -> float:
  """Samples random discrete value.

  To sample a discrete value we sample uniformly a decimal value between the
  minimum and maximum feasible points and returns the closest feasible point.

  Args:
    rng:
    feasible_points:

  Returns:
    The sampled feasible point and a new key.
  """
  min_value = min(feasible_points)
  max_value = max(feasible_points)
  value = sample_uniform(rng, min_value, max_value)
  closest_element = get_closest_element(feasible_points, value)
  return closest_element


def get_closest_element(array: List[float], value: float) -> float:
  """Finds closest element in array to value."""
  gaps = [abs(x - value) for x in array]
  closest_idx = min(enumerate(gaps), key=lambda x: x[1])[0]
  return array[closest_idx]


def _sample_value(
    rng: np.random.Generator,
    param_config: vz.ParameterConfig,
) -> vz.ParameterValueTypes:
  """Samples random value based on the parameter type."""
  if param_config.type == vz.ParameterType.CATEGORICAL:
    return sample_categorical(rng, param_config.feasible_values)
  elif param_config.type == vz.ParameterType.DISCRETE:
    return sample_discrete(rng, param_config.feasible_values)
  else:
    min_value, max_value = param_config.bounds
    if param_config.type == vz.ParameterType.INTEGER:
      return sample_integer(rng, min_value, max_value)
    elif param_config.type == vz.ParameterType.DOUBLE:
      return sample_uniform(rng, min_value, max_value)
    else:
      logging.error('Invalid parameter config type: %s; deafults to DOUBLE.',
                    param_config.type)
      return sample_uniform(rng, min_value, max_value)


def sample_parameters(rng: np.random.Generator,
                      search_space: vz.SearchSpace) -> vz.ParameterDict:
  """Randomly samples parameter values from the search space."""
  sampled_parameters: Dict[str, vz.ParameterValue] = {}
  parameter_configs: List[vz.ParameterConfig] = search_space.parameters

  for param_config in parameter_configs:
    sample_param_value = _sample_value(rng, param_config)
    sampled_parameters[param_config.name] = vz.ParameterValue(
        sample_param_value)

  return vz.ParameterDict(sampled_parameters)


def shuffle_list(rng: np.random.Generator, items: List[_T]) -> List[_T]:
  """Shuffled a list of items (inplace)."""
  rng.shuffle(items)
  return items
