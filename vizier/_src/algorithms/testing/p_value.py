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

"""Utils function to compute p-value for random search null hypothesis.

P-values are used for performing statistical convergence test, which provides
confidence that the algorithm in question does converge due to it merits and not
due to performing large number of evaluations which evidently yeilds low simple
regret even for completely random search.

Assumptions/Limitations
-----------------------
1. The objective function has a unique global optima.
2. The algorithm is expected to converge to the global optima and is evaluated
accordingly.
3. CATEGORICAL parametes are all have the same dimension (no. of categories).
4. INTEGER parameters are not supported.
5. DISCRETE parameters are treated similarly to CATEGORICAL parameters.


Search space with both CATEGORICAL and CONTINUOUS parameters
-------------------------------------------------------------
In the case of a search space composed of both CATEGORICAL and CONTINUOUS
parameters it’s challenging to compute the joint p-value. To address it we
perform a “separated” test by computing two p-values, one for the continuous
and one for the categorical independent of one another.

Then we perform two statistical convergence sub-tests for each type, and require
that both sub-test pass. The “separated” test is stricter than the “joint” test,
and so if an algorithm passes the “separated” test we have even more confidence
that it convergences.
"""

import logging
import math
from typing import Tuple, Union

import numpy as np
from scipy import stats
from vizier import pyvizier as vz
from vizier.pyvizier import converters


def _continuous_p_value(
    n_features: int,
    evaluations: int,
    squared_dist: float,
    low: float = 0,
    high: float = 1,
    verbose: bool = True,
) -> float:
  """Computes p-value assuming continuous random search as null hypothesis.

  P-value is the probability of achieving 'best_features' (or better) by running
  random search 'evaluations' times, where the search space is [low,high]^n.

  It computes the volume of a ball with radius induced by the distance between
  'best_features' and 'optimal_features', and divide it by the volume of
  hypercube induced by the search space. It then computes the probability of at
  least one of the evaluations achieving that distance error.

  Assumption: The squared distance is small enough so that ball with associated
  radius around the optimum does not exceed the search space.


  Arguments:
    n_features:
    evaluations:
    squared_dist:
    low:
    high:
    verbose:

  Returns:
    The p-value.
  """
  if low >= high:
    raise ValueError('high (%s) has to be greater than low(%s).' % high, low)

  def ball_volume(n, r):
    return math.pi**(n / 2) / math.gamma(n / 2 + 1) * r**n

  # Compute the radius associated with the error.
  radius = np.sqrt(squared_dist)
  # The probability of randomly achieving squared error.
  p = ball_volume(n_features, radius) / (high - low)**n_features
  if p >= 1:
    if verbose:
      logging.info('Radius (%s) is too large, setting p-value to 1.', radius)
    return 1
  # The probability that at least one of the evaluations is in the ball.
  continuous_p_value = 1 - (1 - p)**evaluations
  if verbose:
    logging.info('Probability of a single "within": %s', p)
    logging.info('Squared dist from optimum: %s', squared_dist)
    logging.info('Radius: %s', radius)
    logging.info('P-value (Continuous): %s', continuous_p_value)
    logging.info('Equivalent per-axis absolute error: %s',
                 np.sqrt(squared_dist / n_features))
  return continuous_p_value


def _categorical_p_value(n_params: int,
                         dim: int,
                         wrongs: int,
                         evaluations: int,
                         verbose: bool = True) -> float:
  """Computes the p-value assuming random search on categorical search space.

  The assumption is that all categorical parameters have 'dim' feasible values.
  Note that the compuation is valid also when the encoding includes OOV padding.

  Arguments:
    n_params:
    dim:
    wrongs:
    evaluations:
    verbose:

  Returns:
    The p-value under the null hypothesis of random search.
  """
  # Compute the probability of observing best_features under random search.
  p_obs = stats.binom.cdf(n=n_params, k=wrongs, p=(dim - 1) / dim)
  # Compute the p-value as the probability of observing at least one result
  # similar or better to the observation after 'evaluations' trials.
  p_value = 1 - (1 - p_obs)**evaluations
  if verbose:
    logging.info('Number of wrong categorical features: %s', wrongs)
    logging.info('Probability of observing one result: %s', p_obs)
    logging.info('P-value (Categorical): %s', p_value)
  return p_value


def compute_trial_p_values(
    search_space: vz.SearchSpace,
    evaluations: int,
    best_trial: vz.Trial,
    optimum_trial: vz.Trial,
) -> Tuple[float, float]:
  """Compute the trial continuous and categorical p-values.

  DOUBLE parametes are normalize to [0,1] before the p-value is computed, to
  align with the assumption that all parameters have the same bounds.

  Arguments:
    search_space:
    evaluations:
    best_trial:
    optimum_trial:

  Returns:
    The trial continuous and categorical p-values.
  """
  # Initialize the continuous squared distance and the parameters count.
  squared_dist = 0
  n_continuous = 0
  # Initialize categorical wrong parameters, the parameters count and dimension.
  wrongs = 0
  n_categorical = 0
  dim = None
  # DISCRETE parameters are treated similarly to CATEGORICAL parameters.
  categorical_types = [vz.ParameterType.CATEGORICAL, vz.ParameterType.DISCRETE]

  for param_config in search_space.parameters:
    best_val = best_trial.parameters[param_config.name].value
    optimum_val = optimum_trial.parameters[param_config.name].value

    if param_config.type in categorical_types:
      if dim and len(param_config.feasible_values) != dim:
        raise ValueError('All categorical parameters should have the same dim!')
      dim = len(param_config.feasible_values)
      wrongs += int(optimum_val != best_val)
      n_categorical += 1

    elif param_config.type == vz.ParameterType.DOUBLE:
      low, high = param_config.bounds
      squared_dist += ((best_val - optimum_val) / (high - low))**2
      n_continuous += 1

    elif param_config.type == vz.ParameterType.INTEGER:
      raise ValueError('Integer type is not supported!')

  if n_continuous > 0:
    p_value_continuous = _continuous_p_value(n_continuous, evaluations,
                                             squared_dist)
  else:
    p_value_continuous = 0.0
  if n_categorical > 0:
    p_value_categorical = _categorical_p_value(n_categorical, dim, wrongs,
                                               evaluations)
  else:
    p_value_categorical = 0.0
  return p_value_continuous, p_value_categorical


def compute_array_continuous_p_value(
    n_features: int,
    evaluations: int,
    best_features: np.ndarray,
    optimum_features: Union[np.ndarray, float],
    verbose: bool = True,
) -> float:
  """Wrapper to compute continuous p-value for converted Numpy array."""
  return _continuous_p_value(
      n_features,
      evaluations,
      np.sum(np.square(best_features - optimum_features), axis=-1),
      verbose=verbose)


def compute_array_categorical_p_value(n_params: int,
                                      dim: int,
                                      best_features: np.ndarray,
                                      optimum_features: np.ndarray,
                                      evaluations: int,
                                      verbose: bool = True) -> float:
  """Wrapper to compute categorical p-value for converted Numpy array."""
  # Compute the number of wrong categorical features.
  wrongs = np.sum(np.abs(optimum_features - best_features)) / 2
  return _categorical_p_value(n_params, dim, wrongs, evaluations, verbose)


def compute_array_p_values(
    converter: converters.TrialToArrayConverter,
    evaluations: int,
    best_features: np.ndarray,
    optimum_features: np.ndarray,
) -> Tuple[float, float]:
  """Compute p-value for array of converted parameters."""
  categorical_dim = None
  continuous_indx, categorical_indx = [], []
  ind = 0
  for spec in converter.output_specs:
    if spec.type == converters.NumpyArraySpecType.CONTINUOUS:
      continuous_indx.append(ind)
      ind += 1
    elif spec.type == converters.NumpyArraySpecType.ONEHOT_EMBEDDING:
      if categorical_dim and categorical_dim != spec.num_dimensions:
        raise ValueError('All categorical parameters should have the same dim!')
      categorical_dim = spec.num_dimensions
      categorical_indx.extend(list(range(ind, ind + categorical_dim)))
      ind += categorical_dim
    else:
      raise ValueError('Type %s is not supported!' % spec.type)

    best_continuous_features = best_features[continuous_indx]
    best_categorical_features = best_features[categorical_indx]
    optimum_continuous_features = optimum_features[continuous_indx]
    optimum_categorical = optimum_features[categorical_indx]

    if not continuous_indx:
      p_value_continuous = compute_array_continuous_p_value(
          len(continuous_indx), evaluations, best_continuous_features,
          optimum_continuous_features)
    else:
      p_value_continuous = 0.0

    if not categorical_indx:
      p_value_categorical = compute_array_categorical_p_value(
          len(continuous_indx) // categorical_dim, categorical_dim,
          best_categorical_features, optimum_categorical, evaluations)
    else:
      p_value_categorical = 0.0

  return p_value_continuous, p_value_categorical
