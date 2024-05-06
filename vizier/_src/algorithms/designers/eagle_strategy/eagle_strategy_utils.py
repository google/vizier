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

"""Utils functions to support Eagle Strategy designer."""

import collections
import copy
import math
from typing import DefaultDict, Dict, Optional
from absl import logging
import attr
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.random import random_sample

# Standardize name used to assign for trials' metric name, which simplifies
# serializing trials as part of the FireflyPool.
OBJECTIVE_NAME = 'objective'


@attr.define
class FireflyAlgorithmConfig:
  """Configuration hyperparameters for Eagle Strategy / Firefly Algorithm."""
  # Gravity
  gravity: float = 1.0
  negative_gravity: float = 0.02
  # Visiblitiy
  visibility: float = 3.0
  categorical_visibility: float = 0.2
  discrete_visibility: float = 1.0
  # Perturbation
  perturbation: float = 1e-1
  perturbation_lower_bound: float = 1e-3
  categorical_perturbation_factor: float = 25.0
  discrete_perturbation_factor: float = 10.0
  pure_categorical_perturbation: float = 0.1
  max_perturbation: float = 0.5
  # Penalize lack of improvement
  penalize_factor: float = 0.9
  # Pool size
  pool_size_factor: float = 1.2
  # Exploration rate (value > 1.0 encourages more exploration)
  explore_rate: float = 1.0
  # The factor to apply on infeasible trial repel force.
  infeasible_force_factor: float = 0.0
  # The maximum pool size.
  max_pool_size: int = 1000


@attr.define
class Firefly:
  """The Firefly class represents a single firefly in the pool.

  Attributes:
    id_: A unique firefly identifier. This is used to associate trials with
      their parent fireflies.
    perturbation: Controls the amount of exploration. Signifies the amount of
      perturbation to add to the generated trial parameters. The value of
      'perturbation' keeps decreasing if the suggested trial doesn't improve the
      objective function until it reaches 'perturbation_lower_bound' in which
      case we remove the firefly from the pool.
    generation: The number of "successful" (better than the last trial) trials
      suggested from the firefly.
    trial: The best trial associated with the firefly.
  """
  id_: int
  perturbation: float
  generation: int
  trial: vz.Trial


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
  _original_metric_name: str = attr.field(init=False)
  _goal: vz.ObjectiveMetricGoal = attr.field(init=False)

  def __attrs_post_init__(self):
    """Initialize and cache common values and objects."""
    self._search_space = self.problem_statement.search_space
    self._n_parameters = len(self._search_space.parameters)
    self._cache_degrees_of_freedom()
    self._original_metric_name = (
        self.problem_statement.single_objective_metric_name
    )
    self._goal = self.problem_statement.metric_information.item().goal
    logging.info('EagleStrategyUtils instance was created.\n%s', str(self))

  def compute_pull_weight_by_type(
      self,
      other_parameters: vz.ParameterDict,
      suggested_parameters: vz.ParameterDict,
      is_other_fly_better: bool,
  ) -> Dict[vz.ParameterType, float]:
    """Computes the pull weights by type."""
    # Compute squared distances between the vector of parameters of each type.
    squared_distances = self._compute_canonical_distance_squared_by_type(
        other_parameters, suggested_parameters)
    # Determine the direction (attraction vs. repulsion).
    if is_other_fly_better > 0:
      pull_direction = self.config.gravity
    else:
      pull_direction = -self.config.negative_gravity

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
        else:
          raise ValueError('Unsupported parameter type: %s' % param_type)
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
    df = self._n_parameters
    return min(
        10 + round((df**self.config.pool_size_factor + df) * 0.5),
        self.config.max_pool_size,
    )

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
        new_value = random_sample.sample_bernoulli(
            self.rng, prob1, value1, value2
        )
      elif param1_weight <= 0.0:
        new_value = value2
      else:  # param1_weight >= 1.0
        new_value = value1
    else:
      weighted_param_value = value1 * param1_weight + value2 * (
          1 - param1_weight
      )
      if param_config.type == vz.ParameterType.DOUBLE:
        new_value = weighted_param_value
      elif param_config.type == vz.ParameterType.INTEGER:
        new_value = round(weighted_param_value)
      elif param_config.type == vz.ParameterType.DISCRETE:
        new_value = random_sample.get_closest_element(
            param_config.feasible_values, weighted_param_value)
      else:
        raise ValueError('Invalid parameter type: %s' % param_config.type)
      new_value = min(new_value, param_config.bounds[1])
      new_value = max(new_value, param_config.bounds[0])
    return new_value

  def create_perturbations(self, perturbation: float) -> list[float]:
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
      raise ValueError('Invalid parameter type: %s' % param_config.type)

  def get_metric(self, trial: vz.Trial) -> float:
    """Returns the trial metric."""
    if trial.infeasible:
      return np.nan
    return trial.final_measurement.metrics[OBJECTIVE_NAME]  # pytype: disable=bad-return-type

  def is_better_than(
      self,
      trial1: vz.Trial,
      trial2: vz.Trial,
  ) -> bool:
    """Checks whether the 'trial1' is better than 'trial2'.

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
    if trial1.infeasible and not trial2.infeasible:
      return False
    if not trial1.infeasible and trial2.infeasible:
      return True
    if trial1.final_measurement is None or trial2.final_measurement is None:
      return False

    if self._goal == vz.ObjectiveMetricGoal.MAXIMIZE:
      return trial1.final_measurement.metrics[
          OBJECTIVE_NAME] > trial2.final_measurement.metrics[OBJECTIVE_NAME]
    else:
      return trial1.final_measurement.metrics[
          OBJECTIVE_NAME] < trial2.final_measurement.metrics[OBJECTIVE_NAME]

  def is_pure_categorical(self) -> bool:
    """Returns True if all parameters in search_space are categorical."""
    return all([
        p.type == vz.ParameterType.CATEGORICAL
        for p in self._search_space.parameters
    ])

  def standardize_trial_metric_name(self, trial: vz.Trial) -> vz.Trial:
    """Creates a new trial with canonical metric name."""
    if trial.infeasible:
      return trial
    value = trial.final_measurement.metrics[self._original_metric_name].value
    new_trial = vz.Trial(parameters=trial.parameters, metadata=trial.metadata)
    new_trial.complete(
        measurement=vz.Measurement(metrics={OBJECTIVE_NAME: value}))
    return new_trial

  def display_trial(self, trial: vz.Trial) -> str:
    """Construct a string to represent a completed trial."""
    parameters = {
        k: v if isinstance(v, str) else round(v, 3)
        for k, v in trial.parameters.as_dict().items()
    }
    if trial.final_measurement:
      obj_value = (
          f'{list(trial.final_measurement.metrics.values())[0].value:.5f}'
      )
      return f'Value: {obj_value}, Parameters: {parameters}'
    else:
      return f'Parameters: {parameters}'


@attr.define
class FireflyPool:
  """The class maintains the Firefly pool and relevent operations.

  Attributes:
    utils: Eagle Strategy utils class.
    capacity: The maximum number of non-feasible fireflies in the pool.
    size: The current number of non-feasible fireflies in the pool.
    _pool: A dictionary of Firefly objects organized by firefly id.
    _last_id: The last firefly id used to generate a suggestion. It's persistent
      across calls to ensure we don't use the same fly repeatedly.
    _max_fly_id: The maximum value of any fly id ever created. It's persistent
      persistent accross calls to ensure unique ids even if trails were deleted.
    _infeasible_count: The number of infeasible fireflies in the pool.
  """
  _utils: EagleStrategyUtils
  _capacity: int
  _pool: Dict[int, Firefly] = attr.field(init=False, default=attr.Factory(dict))
  _last_id: int = attr.field(init=False, default=0)
  _max_fly_id: int = attr.field(init=False, default=0)
  _infeasible_count: int = attr.field(init=False, default=0)

  @property
  def capacity(self) -> int:
    return self._capacity

  @property
  def size(self) -> int:
    """Returns the number of feasible fireflies in the pool."""
    return len(self._pool) - self._infeasible_count

  def remove_fly(self, fly: Firefly):
    """Removes a fly from the pool."""
    if fly.trial.infeasible:
      raise ValueError('Infeasible firefly should not be removed from pool.')
    del self._pool[fly.id_]

  def get_shuffled_flies(self, rng: np.random.Generator) -> list[Firefly]:
    """Shuffles the fireflies and returns them as a list."""
    return random_sample.shuffle_list(rng, list(self._pool.values()))

  def generate_new_fly_id(self) -> int:
    """Generates a unique fly id (starts from 0) to identify a fly in the pool."""
    self._max_fly_id += 1
    return self._max_fly_id - 1

  def get_next_moving_fly_copy(self) -> Firefly:
    """Finds the next fly, returns a copy of it and updates '_last_id'.

    To find the next moving fly, we start from index of '_last_id'+1 and
    incremently check whether the index exists in the pool. When the loop
    reaches 'max_fly_id' it goes back to index 0. We return a copy of the
    first fly we find an index for. Before returning we also set '_last_id'
    to the next moving fly id.

    Note that we don't assume the existance of '_last_id' in the pool, as the
    fly with `_last_id` might be removed from the pool.

    Returns:
      A copy of the next moving fly.
    """
    curr_id = self._last_id + 1
    while curr_id != self._last_id:
      if curr_id > self._max_fly_id:
        # Passed the maximum id. Start from the first one as ids are monotonic.
        curr_id = next(iter(self._pool))
      if curr_id in self._pool and not self._pool[curr_id].trial.infeasible:
        self._last_id = curr_id
        return copy.deepcopy(self._pool[curr_id])
      curr_id += 1

    return copy.deepcopy(self._pool[self._last_id])

  def is_best_fly(self, fly: Firefly) -> bool:
    """Checks if the 'fly' has the best final measurement in the pool."""
    for other_fly_id, other_fly in self._pool.items():
      if other_fly_id != fly.id_ and self._utils.is_better_than(
          other_fly.trial, fly.trial
      ):
        return False
    return True

  def find_parent_fly(self, parent_fly_id: Optional[int]) -> Optional[Firefly]:
    """Obtains the parent firefly associated with the trial.

    Extract the associated parent id from the trial's metadata and attempt
    to find the parent firefly in the pool. If it doesn't exist return None.

    Args:
      parent_fly_id:

    Returns:
      Firefly or None.
    """
    parent_fly = self._pool.get(parent_fly_id, None)
    return parent_fly

  def find_closest_parent(self, trial: vz.Trial) -> Firefly:
    """Finds the closest fly in the pool to a given trial."""
    if not self._pool:
      raise ValueError('Pool was empty when searching for closest parent.')

    min_dist, closest_parent = float('inf'), next(iter(self._pool.values()))
    for other_fly in self._pool.values():
      if other_fly.trial.infeasible:
        continue
      curr_dist = self._utils.compute_cononical_distance(
          other_fly.trial.parameters, trial.parameters
      )
      if curr_dist < min_dist:
        min_dist = curr_dist
        closest_parent = other_fly

    return closest_parent

  def create_or_update_fly(self, trial: vz.Trial, parent_fly_id: int) -> None:
    """Creates a new fly in the pool or update an existing one.

    This method is called when the pool is below capacity.

    The newly created fly is assigned with 'id_' equals 'parent_fly_id' which
    is taken from the trial's metadata. The fly's id is determined during the
    'Suggest' method and stored in the metadata.

    Edge case: if the 'parent_fly_id' already exists in the pool (and the pool
    is below capacity) we update the matching fly with the better trial. This
    scenario could happen if for example a batch larger than the pool capacity
    was suggested, and then trials associated with the same fly were reported
    as COMPLETED sooner than the other trials, so when the pool is updated
    during 'Update' we encounter trials from the same fly before the pool is at
    capactiy.

    Args:
      trial:
      parent_fly_id:
    """
    if parent_fly_id not in self._pool:
      # Create a new Firefly in pool.
      new_fly = Firefly(
          id_=parent_fly_id,
          perturbation=self._utils.config.perturbation,
          generation=1,
          trial=trial,
      )
      self._pool[parent_fly_id] = new_fly
      if trial.infeasible:
        self._infeasible_count += 1
    else:
      # Parent fly id already in pool. Update trial if there was improvement.
      if self._utils.is_better_than(trial, self._pool[parent_fly_id].trial):
        self._pool[parent_fly_id].trial = trial
