"""Utils functions to support the Eagle Strategy designer."""

import math
from typing import Any, Dict, List, Optional, Tuple

import attr
import jax
from jax import random
from jax.interpreters.xla import DeviceArray
from vizier import pyvizier as vz
from vizier._src.algorithms.random import random_sample

PRNGKey = Any


@attr.define
class FireflyAlgorithmConfig:
  """Configuration hyperparameters for Eagle Strategy / Firefly Algorithm."""
  perturbation_factor: float = 1e-2
  categorical_perturbation_factor: float = 25
  pure_categorical_perturbation: float = 0.1
  discrete_perturbation_factor: float = 10.0
  perturbation_lower_bound: float = 1e-3
  gravity: float = 1.0
  visibility: float = 1.0
  categorical_visibility: float = 0.2
  discrete_visibility: float = 1.0
  negative_gravity: float = 0.02
  firefly_pool_size_factor: float = 1.2
  explore_rate: float = 1.0
  max_perturbation: float = 0.5


@attr.define
class EagleStrategyUtils:
  """Eagle Strategy utils.

  The class is used to efficiently compute distances, firefly pulls,
    param perturbations, comparison between trials, etc.

  Attributes:
    search_space: The search space that Eagle attempts to optimize on.
    config: The Eagle Strategy configuration.
    metric_name: The name of the objetive metric to optimize on.
    goal: The direction of the optimization (MAXIMIZATION vs. MINIMIZATION)
  """

  search_space: vz.SearchSpace
  config: FireflyAlgorithmConfig
  metric_name: str
  goal: vz.ObjectiveMetricGoal
  _is_pure_categorical: Optional[bool] = attr.field(init=False, default=None)
  _degree_of_freedom: Dict[vz.ParameterType, int] = attr.field(
      init=False, default=attr.Factory(dict))
  _pool_capacity: Optional[int] = attr.field(init=False, default=None)
  _param_perturb_scales: Optional[DeviceArray] = attr.field(
      init=False, default=None)
  _n_parameters: int = attr.field(init=False)
  _pure_categorical_perturbations: List[float] = attr.field(init=False)

  def __attrs_post_init__(self):
    """Initialize and cache common values."""
    self._n_parameters = len(self.search_space.parameters)
    self._pure_categorical_perturbations = [
        self.config.pure_categorical_perturbation
        for _ in range(self._n_parameters)
    ]

  def compute_pull_weight_by_type(
      self,
      other_parameters: vz.ParameterDict,
      suggested_parameters: vz.ParameterDict,
      is_other_fly_better: bool,
  ) -> Dict[vz.ParameterType, float]:
    """Computes the pull wieghts by type."""
    squared_distances = self.compute_canonical_distance_squared_by_type(
        other_parameters, suggested_parameters)

    # Determine the direction (attraction vs. repulsion).
    if is_other_fly_better > 0:
      pull_direction = self.config.gravity
    else:
      pull_direction = self.config.negative_gravity

    pull_weights = {}

    # Iterate over the squared distance by type and compute the pull force.
    for param_type, squared_distance in squared_distances.items():
      degree_of_freedom = self._compute_degree_of_freedom(param_type)
      if degree_of_freedom == 0:
        # No parameters for the parameter type, set pull weight to 0.
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

  def _compute_degree_of_freedom(self, param_type: vz.ParameterType) -> int:
    """Computes the degrees of freedom for the parameter type.

    The function counts the number of parameter configs in the search space that
    has more than only a single feasible point.

    Args:
      param_type:

    Returns:
      The degrees of freedom associated with the specified parameter type.
    """
    if param_type not in self._degree_of_freedom:
      df = 0
      for param_config in self.search_space.parameters:
        if param_config.type == param_type and param_config.num_feasible_values > 1:
          df += 1
      self._degree_of_freedom[param_type] = df
    return self._degree_of_freedom[param_type]

  def compute_canonical_distance_squared_by_type(
      self,
      p1: vz.ParameterDict,
      p2: vz.ParameterDict,
  ) -> Dict[vz.ParameterType, float]:
    """Computes the canonical distance squared by parameter type.

    Args:
      p1: The first ParameterDict
      p2: The second ParameterDict

    Returns:
      Dictionary of summed squared distance of all the parameters with the type.
    """

    dist_squared_by_type = {
        vz.ParameterType.DOUBLE: 0.0,
        vz.ParameterType.DISCRETE: 0.0,
        vz.ParameterType.INTEGER: 0.0,
        vz.ParameterType.CATEGORICAL: 0.0
    }

    for param_config in self.search_space.parameters:
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
    dist_by_type = self.compute_canonical_distance_squared_by_type(p1, p2)
    print(dist_by_type)
    return sum(dist_by_type.values())

  def compute_pool_capacity(self) -> int:
    """Computes the pool capacity."""
    if not self._pool_capacity:
      self._pool_capacity = 10 + int(
          0.5 * self._n_parameters +
          self._n_parameters**self.config.firefly_pool_size_factor)
    return self._pool_capacity

  def combine_two_parameters(
      self,
      key: PRNGKey,
      param_config: vz.ParameterConfig,
      param1: vz.ParameterDict,
      param2: vz.ParameterDict,
      param1_weight: float,
  ) -> Tuple[PRNGKey, vz.ParameterValueTypes]:
    """Combnies the values of two parameters based on their weights and type.

    For decimal parameters, performs a linear combination of two features,
    by computing f1 * f1_weight + f2 * (1 - f1_weight). Note that this is not a
    convex combination, because 'f1_weight' can be outside [0, 1].

    For categorical parameters, uses Bernuolli to choose between the value of
    the
    first parameter and the second with probability that equals the weight.

    Args:
      key:
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
        key, new_value = random_sample.sample_bernoulli(key, prob1, value1,
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
    return key, new_value

  def perturb_parameter(
      self,
      key: PRNGKey,
      param_config: vz.ParameterConfig,
      value: vz.ParameterValueTypes,
      perturbation: float,
  ) -> Tuple[PRNGKey, vz.ParameterValueTypes]:
    """Perturbs the parameter based on its type and the amount of perturbation.

    Changes the 'value' by 'perturbation', which is expected to be in [-1,1].
    The sign represents the "direction" if applicable, and the absolute value
    decides the magnitude. 1 is the maximum possible change and 0 is no change.

    For categorical features, with probability 'perturbation', replace with a
    uniformly random category.

    For numeric features, move towards the end of range. perturbation=-1 will
    always return the minimum possible value, and perturbation=+1 will return
    the maximum possible value.

    Args:
      key: random key
      param_config: The configuration of the parameter
      value: The parameter value before perturbation
      perturbation: The amount of pertrubation to apply

    Returns:
      The parameter value after perturbation and the updated key

    Raises:
      Exception: if the parameter has invalid type.
    """
    if param_config.type == vz.ParameterType.CATEGORICAL:
      key, uniform = random_sample.sample_uniform(key)
      if uniform < abs(perturbation):
        return random_sample.sample_categorical(key,
                                                param_config.feasible_values)
      else:
        return key, value

    min_value, max_value = param_config.bounds
    perturb_val = value + perturbation * (max_value - min_value)

    if param_config.type == vz.ParameterType.DISCRETE:
      return key, random_sample.get_closest_element(
          param_config.feasible_values, perturb_val)

    perturb_val = min(perturb_val, param_config.bounds[1])
    perturb_val = max(perturb_val, param_config.bounds[0])

    if param_config.type == vz.ParameterType.DOUBLE:
      return key, perturb_val
    elif param_config.type == vz.ParameterType.INTEGER:
      return key, round(perturb_val)
    else:
      raise Exception('Invalid parameter type: %s' % param_config.type)

  def better_than(
      self,
      trial1: vz.Trial,
      trial2: vz.Trial,
  ) -> bool:
    """Checks whether the current trial is better than another trial.

    The comparison is based on the value of final measurement and whether it
    goal
    is MAXIMIZATION or MINIMIZATON.

    If either trials is not completed, infeasible or missing the final
    measurement
    returns False.

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

    if self.goal == vz.ObjectiveMetricGoal.MAXIMIZE:
      return trial1.final_measurement.metrics[
          self.metric_name] > trial2.final_measurement.metrics[self.metric_name]
    else:
      return trial1.final_measurement.metrics[
          self.metric_name] < trial2.final_measurement.metrics[self.metric_name]

  def is_pure_categorical(self) -> bool:
    """Returns True if all parameters in search_space are categorical."""
    if not self._is_pure_categorical:
      self._is_pure_categorical = all([
          p.type == vz.ParameterType.CATEGORICAL
          for p in self.search_space.parameters
      ])
    return self._is_pure_categorical

  def create_perturbations(
      self, key: PRNGKey,
      perturbation_factor: float) -> Tuple[PRNGKey, List[float]]:
    """"Creates perturbations vector."""
    if self.is_pure_categorical():
      return key, self._pure_categorical_perturbations
    # Initialize the parameter perturbation scaling once.
    if self._param_perturb_scales is None:
      scales = [1.0 for _ in range(self._n_parameters)]
      for i, param_config in enumerate(self.search_space.parameters):
        if param_config.type == vz.ParameterType.CATEGORICAL:
          # For CATEGORICAL parameters, multiply the perturbation by the
          # pre-configured factor. This perturbation is interpreted as a
          # probability of replacing the value with a uniformly random category.
          scales[i] = self.config.categorical_perturbation_factor

        if param_config.type == vz.ParameterType.DISCRETE:
          # For DISCRETE parameters, multiply the perturbation by the
          # pre-configured factor divided by the number of feasible points.
          scales[i] = self.config.discrete_perturbation_factor / (
              param_config.num_feasible_values *
              self.config.perturbation_factor)
      self._param_perturb_scales = jax.numpy.asarray(scales)
    # Sample vector from Laplace distribution.
    new_key, subkey = random.split(key)
    perturbations = random.laplace(subkey, (self._n_parameters,))
    # Normalize pertubations and scale by `perturbation_factor`.
    perturbation_direction = perturbations / max(abs(perturbations))
    perturbations = perturbation_direction * perturbation_factor
    # Scale the perturbation and convert to float.
    return new_key, [
        float(x) for x in perturbations * self._param_perturb_scales
    ]

  def display_trial(self, trial: vz.Trial) -> str:
    """Construct a string to represent a completed trial with metadata."""
    parameters = {k: f'{v:.3f}' for k, v in trial.parameters.as_dict().items()}
    parent_fly_id = trial.metadata.ns('eagle').get('parent_fly_id')
    if trial.final_measurement:
      obj_value = f'{list(trial.final_measurement.metrics.values())[0].value:.3f}'
    else:
      obj_value = '-'
    return (f'Parameters: {parameters}, Value: {obj_value}, Parent fly ID: '
            f'{parent_fly_id}')
