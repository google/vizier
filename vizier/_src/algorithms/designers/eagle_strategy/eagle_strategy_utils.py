# Copyright 2022 Google LLC.
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

"""Utils functions to support Eagle Strategy designer."""

import collections
import math
from typing import DefaultDict, Dict, List

import attr
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.random import random_sample


@attr.define
class FireflyAlgorithmConfig:
  """Configuration hyperparameters for Eagle Strategy / Firefly Algorithm."""
  # Gravity
  gravity: float = 1.0
  negative_gravity: float = 0.02
  # Visiblitiy
  visibility: float = 1.0
  categorical_visibility: float = 0.2
  discrete_visibility: float = 1.0
  # Perturbation
  perturbation: float = 1e-2
  categorical_perturbation_factor: float = 25
  pure_categorical_perturbation: float = 0.1
  discrete_perturbation_factor: float = 10.0
  perturbation_lower_bound: float = 1e-3
  max_perturbation: float = 0.5
  # Pool size
  pool_size_factor: float = 1.2
  # Exploration rate (value > 1.0 encourages more exploration)
  explore_rate: float = 1.0


@attr.define
class EagleStrategyUtils:
  """Eagle Strategy utils.

  Attributes:
    search_space: The search space that Eagle attempts to optimize on.
    config: The Eagle Strategy configuration.
    rng: The random generator to sample random distributions.
  """
  problem_statement: vz.ProblemStatement
  config: FireflyAlgorithmConfig
  rng: np.random.Generator
  _search_space: vz.SearchSpace = attr.field(init=False)
  _n_parameters: int = attr.field(init=False)
  _degrees_of_freedom: DefaultDict[vz.ParameterType, int] = attr.field(
      init=False, factory=lambda: collections.defaultdict(int))
  _metric_name: str = attr.field(init=False)
  _goal: vz.ObjectiveMetricGoal = attr.field(init=False)

  def __attrs_post_init__(self):
    """Initialize and cache common values and objects."""
    self._search_space = self.problem_statement.search_space
    self._n_parameters = len(self._search_space.parameters)
    self._cache_degrees_of_freedom()
    self._metric_name = self.problem_statement.single_objective_metric_name
    self._goal = self.problem_statement.metric_information.item().goal

  def compute_pull_weight_by_type(
      self,
      other_parameters: vz.ParameterDict,
      suggested_parameters: vz.ParameterDict,
      is_other_fly_better: bool,
  ) -> Dict[vz.ParameterType, float]:
    """Computes the pull wieghts by type."""
    # Compute squared distances between the vector of parameters of each type.
    squared_distances = self._compute_canonical_distance_squared_by_type(
        other_parameters, suggested_parameters)
    # Determine the direction (attraction vs. repulsion).
    if is_other_fly_better > 0:
      pull_direction = self.config.gravity
    else:
      pull_direction = self.config.negative_gravity

    pull_weights = {}
    # Iterate over the squared distance by type and compute the pull force.
    for param_type, squared_distance in squared_distances.items():
      degree_of_freedom = self._degrees_of_freedom[param_type]
      if degree_of_freedom == 0:
        pull_weights[param_type] = 0
      else:
        scaled_squared_distance = squared_distance / degree_of_freedom * 10
        # Determine the visibilty based on the parameter type.
        if param_type == vz.ParameterType.CATEGORICAL:
          visiblity = self.config.categorical_visibility
        elif param_type in [
            vz.ParameterType.DISCRETE, vz.ParameterType.INTEGER
        ]:
          visiblity = self.config.discrete_visibility
        elif param_type == vz.ParameterType.DOUBLE:
          visiblity = self.config.visibility
        # Compute the pull weight and insert to dictionary.
        pull_weights[param_type] = math.exp(
            -visiblity * scaled_squared_distance) * pull_direction

    return pull_weights

  @property
  def _param_perturb_scales(self):
    """Set the parameter perturbation scaling."""
    scales = [1.0 for _ in range(self._n_parameters)]
    for i, param_config in enumerate(self._search_space.parameters):
      if param_config.type == vz.ParameterType.CATEGORICAL:
        # For CATEGORICAL parameters, the perturbation is interpreted as a
        # probability of replacing the value with a uniformly random category.
        scales[i] = self.config.categorical_perturbation_factor
      if param_config.type == vz.ParameterType.DISCRETE:
        scales[i] = self.config.discrete_perturbation_factor / (
            param_config.num_feasible_values * self.config.perturbation)
    return np.asarray(scales)

  def _cache_degrees_of_freedom(self) -> None:
    """Computes the degrees of freedom for each parameter type and cache it.

    The function counts the number of parameter configs in the search space that
    has more than only a single feasible point.
    """
    for param_config in self._search_space.parameters:
      if param_config.num_feasible_values > 1:
        self._degrees_of_freedom[param_config.type] += 1

  def _compute_canonical_distance_squared_by_type(
      self,
      p1: vz.ParameterDict,
      p2: vz.ParameterDict,
  ) -> Dict[vz.ParameterType, float]:
    """Computes the canonical distance squared by parameter type."""
    dist_squared_by_type = {
        vz.ParameterType.DOUBLE: 0.0,
        vz.ParameterType.DISCRETE: 0.0,
        vz.ParameterType.INTEGER: 0.0,
        vz.ParameterType.CATEGORICAL: 0.0
    }

    for param_config in self._search_space.parameters:
      p1_value = p1[param_config.name].value
      p2_value = p2[param_config.name].value

      if param_config.type == vz.ParameterType.CATEGORICAL:
        dist_squared_by_type[param_config.type] += int(p1_value == p2_value)
      else:
        min_value, max_value = param_config.bounds
        dist = (p1_value - p2_value) / (max_value - min_value)
        dist_squared_by_type[param_config.type] += dist * dist

    return dist_squared_by_type

  def compute_cononical_distance(
      self,
      p1: vz.ParameterDict,
      p2: vz.ParameterDict,
  ) -> float:
    """Computes the canonical squared distance between two parameters."""
    dist_by_type = self._compute_canonical_distance_squared_by_type(p1, p2)
    return sum(dist_by_type.values())

  def compute_pool_capacity(self) -> int:
    """Computes the pool capacity."""
    return 10 + int(0.5 * self._n_parameters +
                    self._n_parameters**self.config.pool_size_factor)

  def combine_two_parameters(
      self,
      param_config: vz.ParameterConfig,
      param1: vz.ParameterDict,
      param2: vz.ParameterDict,
      param1_weight: float,
  ) -> vz.ParameterValueTypes:
    """Combnies the values of two parameters based on their weights and type.

    For decimal parameters, performs a linear combination of two parameters,
    by computing f1 * f1_weight + f2 * (1 - f1_weight). Note that this is not a
    convex combination, because 'f1_weight' can be outside [0, 1].

    For categorical parameters, uses Bernuolli to choose between the value of
    the first parameter and the second with probability that equals the weight.

    Args:
      param_config:
      param1:
      param2:
      param1_weight:

    Returns:
      The combined weighted value of the two parameters.
    """
    value1 = param1[param_config.name].value
    value2 = param2[param_config.name].value
    if param_config.type == vz.ParameterType.CATEGORICAL:
      if 0.0 < param1_weight < 1.0:
        prob1 = param1_weight
        new_value = random_sample.sample_bernoulli(self.rng, prob1, value1,
                                                   value2)
      if param1_weight <= 0.0:
        new_value = value2
      elif param1_weight >= 1.0:
        new_value = value1
    else:
      weighted_param_value = value1 * param1_weight + value2 * (1 -
                                                                param1_weight)
      if param_config.type == vz.ParameterType.DOUBLE:
        new_value = weighted_param_value
      elif param_config.type == vz.ParameterType.INTEGER:
        new_value = round(weighted_param_value)
      elif param_config.type == vz.ParameterType.DISCRETE:
        new_value = random_sample.get_closest_element(
            param_config.feasible_values, weighted_param_value)
      new_value = min(new_value, param_config.bounds[1])
      new_value = max(new_value, param_config.bounds[0])
    return new_value

  def create_perturbations(self, perturbation: float) -> List[float]:
    """Creates perturbations array."""
    if self.is_pure_categorical():
      return [
          self.config.pure_categorical_perturbation
          for _ in range(self._n_parameters)
      ]
    perturbations = self.rng.laplace(size=(self._n_parameters,))
    perturbation_direction = perturbations / max(abs(perturbations))
    perturbations = perturbation_direction * perturbation
    return [float(x) for x in perturbations * self._param_perturb_scales]

  def perturb_parameter(
      self,
      param_config: vz.ParameterConfig,
      value: vz.ParameterValueTypes,
      perturbation: float,
  ) -> vz.ParameterValueTypes:
    """Perturbs the parameter based on its type and the amount of perturbation.

    For categorical parameters, with probability 'perturbation', replace with a
    uniformly random category.

    For numeric parameters, 'perturbation' determines the fraction of the
    parameter range to add, where perturbation=-1 will always return the minimum
    possible value, and perturbation=+1 will return the maximum possible value.

    Args:
      param_config: The configuration of the parameter
      value: The parameter value before perturbation
      perturbation: The amount of pertrubation to apply

    Returns:
      The parameter value after perturbation.

    Raises:
      Exception: if the parameter has invalid type.
    """
    if param_config.type == vz.ParameterType.CATEGORICAL:
      if random_sample.sample_uniform(self.rng) < abs(perturbation):
        return random_sample.sample_categorical(self.rng,
                                                param_config.feasible_values)
      else:
        return value

    min_value, max_value = param_config.bounds
    perturb_val = value + perturbation * (max_value - min_value)

    if param_config.type == vz.ParameterType.DISCRETE:
      return random_sample.get_closest_element(param_config.feasible_values,
                                               perturb_val)

    perturb_val = min(perturb_val, param_config.bounds[1])
    perturb_val = max(perturb_val, param_config.bounds[0])
    if param_config.type == vz.ParameterType.DOUBLE:
      return perturb_val
    elif param_config.type == vz.ParameterType.INTEGER:
      return round(perturb_val)
    else:
      raise Exception('Invalid parameter type: %s' % param_config.type)

  def get_metric(self, trial: vz.Trial) -> float:
    """Returns the trial metric."""
    return trial.final_measurement.metrics[self._metric_name]

  def is_better_than(
      self,
      trial1: vz.Trial,
      trial2: vz.Trial,
  ) -> bool:
    """Checks whether the current trial is better than another trial.

    The comparison is based on the value of final measurement and whether it
    goal is MAXIMIZATION or MINIMIZATON.

    If either trials is not completed, infeasible or missing the final
    measurement returns False.

    Args:
      trial1:
      trial2:

    Returns:
      Whether trial1 is greater than trial2.
    """
    if not trial1.is_completed or not trial2.is_completed:
      return False
    if trial1.infeasible or trial2.infeasible:
      return False
    if trial1.final_measurement is None or trial2.final_measurement is None:
      return False

    if self._goal == vz.ObjectiveMetricGoal.MAXIMIZE:
      return trial1.final_measurement.metrics[
          self._metric_name] > trial2.final_measurement.metrics[
              self._metric_name]
    else:
      return trial1.final_measurement.metrics[
          self._metric_name] < trial2.final_measurement.metrics[
              self._metric_name]

  def is_pure_categorical(self) -> bool:
    """Returns True if all parameters in search_space are categorical."""
    return all([
        p.type == vz.ParameterType.CATEGORICAL
        for p in self._search_space.parameters
    ])

  @property
  def is_linear_scale(self) -> bool:
    """Returns whether all decimal parameters in search space has linear scale.
    """
    return all([
        p.scale_type is None or p.scale_type == vz.ScaleType.LINEAR
        for p in self._search_space.parameters
    ])

  def display_trial(self, trial: vz.Trial) -> str:
    """Construct a string to represent a completed trial."""
    parameters = {
        k: v if isinstance(v, str) else round(v, 3)
        for k, v in trial.parameters.as_dict().items()
    }
    if trial.final_measurement:
      obj_value = f'{list(trial.final_measurement.metrics.values())[0].value:.3f}'
      return f'Value: {obj_value}, Parameters: {parameters}'
    else:
      return f'Parameters: {parameters}'
