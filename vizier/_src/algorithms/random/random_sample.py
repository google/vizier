"""Randomization related utils."""
import logging
from typing import Dict, List, Tuple, TypeVar, Any
from jax import random
import numpy as np
from vizier import pyvizier as vz

_T = TypeVar('_T')
PRNGKey = Any


def sample_uniform(key: PRNGKey,
                   min_value=0,
                   max_value=1) -> Tuple[PRNGKey, float]:
  """Samples unifrom value and udpate key."""
  new_key, subkey = random.split(key)
  return new_key, float(
      random.uniform(subkey, minval=min_value, maxval=max_value))


def sample_bernoulli(
    key: PRNGKey,
    prob1: float,
    value1: _T = 0,
    value2: _T = 1,
) -> Tuple[PRNGKey, _T]:
  """Samples value1 with probability prob1."""
  new_key, subkey = random.split(key)
  if float(random.bernoulli(subkey, prob1)):
    return new_key, value1
  else:
    return new_key, value2


def sample_integer(
    key: PRNGKey,
    min_value: float,
    max_value: float,
) -> Tuple[PRNGKey, int]:
  """Samples a random integer."""
  new_key, val = sample_uniform(key, min_value, max_value)
  return new_key, round(val)


def sample_categorical(key: PRNGKey,
                       categories: List[str]) -> Tuple[PRNGKey, str]:
  """Samples a random categorical value."""
  new_key, subkey = random.split(key)
  ind = int(random.choice(subkey, np.array(range(len(categories)))))
  return new_key, categories[ind]


def sample_discrete(key: PRNGKey,
                    feasible_points: List[float]) -> Tuple[PRNGKey, float]:
  """Samples random discrete value.

  To sample a discrete value we sample uniformly a decimal value between the
  minimum and maximum feasible points and returns the closest feasible point.

  Args:
    key:
    feasible_points:

  Returns:
    The sampled feasible point and a new key.
  """
  min_value = min(feasible_points)
  max_value = max(feasible_points)
  new_key, value = sample_uniform(key, min_value, max_value)
  closest_element = get_closest_element(feasible_points, value)
  return new_key, closest_element


def get_closest_element(array: List[float], value: float) -> float:
  """Finds closest element in array to value."""
  gaps = [abs(x - value) for x in array]
  closest_idx = min(enumerate(gaps), key=lambda x: x[1])[0]
  return array[closest_idx]


def _sample_value(
    key: PRNGKey,
    param_config: vz.ParameterConfig,
) -> Tuple[PRNGKey, vz.ParameterValueTypes]:
  """Samples random value based on the parameter type."""
  if param_config.type == vz.ParameterType.CATEGORICAL:
    return sample_categorical(key, param_config.feasible_values)
  elif param_config.type == vz.ParameterType.DISCRETE:
    return sample_discrete(key, param_config.feasible_values)
  else:
    min_value, max_value = param_config.bounds
    if param_config.type == vz.ParameterType.INTEGER:
      return sample_integer(key, min_value, max_value)
    elif param_config.type == vz.ParameterType.DOUBLE:
      return sample_uniform(key, min_value, max_value)
    else:
      logging.error('Invalid parameter config type: %s; deafults to DOUBLE.',
                    param_config.type)
      return sample_uniform(key, min_value, max_value)


def sample_input_parameters(
    key: PRNGKey,
    search_space: vz.SearchSpace) -> Tuple[PRNGKey, vz.ParameterDict]:
  """Randomly samples input parameter values from the search space.

  Args:
    key: PRNGKey
    search_space: vz.SearchSpace describing the study search space.

  Returns:
    ParameterDict with the sampled parameter values.
  """
  sampled_parameters: Dict[str, vz.ParameterValue] = {}
  parameter_configs: List[vz.ParameterConfig] = search_space.parameters

  for param_config in parameter_configs:
    key, sample_param_value = _sample_value(key, param_config)
    sampled_parameters[param_config.name] = vz.ParameterValue(
        sample_param_value)

  return key, vz.ParameterDict(sampled_parameters)


def shuffle_list(key: PRNGKey, items: List[_T]) -> Tuple[PRNGKey, List[_T]]:
  """Shuffles a list of items.

  Args:
    key: PRNGKey
    items: list to be shuffled

  Returns:
    List containing the original items in a new order.
  """
  new_key, subkey = random.split(key)
  shuffled_indices = random.shuffle(subkey, np.array(range(len(items))))
  shuffled_items = [items[i] for i in shuffled_indices]
  return new_key, shuffled_items
