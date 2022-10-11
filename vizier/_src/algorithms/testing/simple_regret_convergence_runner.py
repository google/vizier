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

"""Perform simple-regret convergence test.

The test is based on comparing the results to a random search. It performs
a statistical test to evaluate the probability (p-value) of observing the
best features results under the null hypothesis of running a random search, if
it had run for the same number of evaluations. The p-value is compared against
a pre-determined significnce-level (alpha) to decide if the test has passed or
not.

To expedite the convergence test, instead of running an actual random search
the theortical p-value is computed instead.

To simplify the implementation, we assume the existence of a converter that
maps the original search space to [0,1]^n as is often the case when handling
with vectorized acquisition/objective functions. Though, if needed there
shouldn't any limitation to extend the test to a generic vz.SearchSpace.

In addition, we assume that the converter only has CATEGORICAL or FLOAT
parameters and all DISCRETE parameters were converted to FLOAT to actually
account for their values (which is the default converter configuration).
"""

import logging
import math
from typing import Tuple, Union

import numpy as np
from scipy import stats
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier.pyvizier import converters


class FailedSimpleRegretConvergenceTestError(Exception):
  """Exception raised for simple-regret convergence test fails."""


def compute_continuous_p_value(
    n_features: int,
    evaluations: int,
    best_features: np.ndarray,
    optimum_features: Union[np.ndarray, float],
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
    best_features:
    optimum_features:
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
  squared_dist = np.sum(np.square(best_features - optimum_features), axis=-1)
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
    logging.info('Best continuous features %s:', best_features)
    logging.info('Optimum continuous feautres %s:', optimum_features)
    logging.info('Squared dist from optimum: %s', squared_dist)
    logging.info('Radius: %s', radius)
    logging.info('P-value (Continuous): %s', continuous_p_value)
    logging.info('Equivalent per-axis absolute error: %s',
                 np.sqrt(squared_dist / n_features))
  return continuous_p_value


def compute_categorical_p_value(n_features: int,
                                dim: int,
                                best_features: np.ndarray,
                                optimum_features: np.ndarray,
                                evaluations: int,
                                verbose: bool = True) -> float:
  """Computes the p-value assuming random search on categorical search space.

  The assumption is that all categorical parameters have 'dim' feasible values.

  Arguments:
    n_features:
    dim:
    best_features:
    optimum_features:
    evaluations:
    verbose:

  Returns:
    The p-value under the null hypothesis of random search.
  """
  # Compute the number of wrong categorical features.
  wrongs = np.sum(np.abs(optimum_features - best_features)) / 2
  # Compute the probability of observing best_features under random search.
  p_obs = stats.binom.cdf(n=n_features, k=wrongs, p=(dim - 1) / dim)
  # Compute the p-value as the probability of observing at least one result
  # similar or better to the observation over 'evaluations' trials.
  p_value = 1 - (1 - p_obs)**evaluations

  if verbose:
    logging.info('Best categorical features    %s:', best_features)
    logging.info('Optimum categorical feautres %s:', optimum_features)
    logging.info('Number of wrong categorical features: %s', wrongs)
    logging.info('Probability of observing one result: %s', p_obs)
    logging.info('P-value (Categorical): %s', p_value)
  return p_value


def compute_p_value(
    best_features: np.ndarray,
    optimum_features: np.ndarray,
    converter: converters.TrialToArrayConverter,
    evaluations: int,
) -> Tuple[float, float]:
  """Compute p-value for both continuous and categorical features.

  The p-value is computed separately for categorical and continuous parameters,
  where the combined p-value is the product of the two. But ensure that one type
  of p-value is not over-compensating for the other, it's recommended to
  evaluate each p-value against the confidence level separately, which is a
  stricter test.

  The assumptions is that all categorical parameters have the same dimension.

  Arguments:
    best_features:
    optimum_features:
    converter:
    evaluations:

  Returns:
    p_value_continuous, p_value_categorical
  """
  best_continuous_arr = []
  optimum_continuous_arr = []
  best_categorical_arr = []
  optimum_categorical_arr = []
  ind = 0
  n_continuous_features = 0
  n_categorical_features = 0
  categorical_dim = None
  is_continuous = False
  is_categorical = False

  for spec in converter.output_specs:
    if spec.type == converters.NumpyArraySpecType.CONTINUOUS:
      best_continuous_arr.append(best_features[ind])
      optimum_continuous_arr.append(optimum_features[ind])
      ind += 1
      n_continuous_features += 1
      is_continuous = True
    elif spec.type == converters.NumpyArraySpecType.ONEHOT_EMBEDDING:
      if categorical_dim and categorical_dim != spec.num_dimensions:
        raise ValueError('All categorical parameters should have the same dim!')
      categorical_dim = spec.num_dimensions
      best_categorical_arr.extend(best_features[ind:ind + categorical_dim])
      optimum_categorical_arr.extend(optimum_features[ind:ind +
                                                      categorical_dim])
      ind += categorical_dim
      n_categorical_features += 1
      is_categorical = True
    else:
      raise ValueError('The type %s is not supported!' % spec.type)

  best_continuous_features = np.array(best_continuous_arr)
  optimum_continuous_features = np.array(optimum_continuous_arr)
  best_categorical_features = np.array(best_categorical_arr)
  optimum_categorical = np.array(optimum_categorical_arr)

  if is_continuous:
    p_value_continuous = compute_continuous_p_value(
        n_continuous_features, evaluations, best_continuous_features,
        optimum_continuous_features)
  else:
    p_value_continuous = 0.0

  if is_categorical:
    p_value_categorical = compute_categorical_p_value(
        n_categorical_features, categorical_dim, best_categorical_features,
        optimum_categorical, evaluations)
  else:
    p_value_categorical = 0.0

  return p_value_continuous, p_value_categorical


def randomize_features(
    converter: converters.TrialToArrayConverter) -> np.ndarray:
  """Generate a random array of features to be used as score_fn shift."""
  features_arrays = []
  for spec in converter.output_specs:
    if spec.type == converters.NumpyArraySpecType.ONEHOT_EMBEDDING:
      dim = spec.num_dimensions - spec.num_oovs
      features_arrays.append(
          np.eye(spec.num_dimensions)[np.random.randint(0, dim)])
    elif spec.type == converters.NumpyArraySpecType.CONTINUOUS:
      features_arrays.append(np.random.uniform(0.4, 0.6, size=(1,)))
    else:
      raise ValueError('The type %s is not supported!' % spec.type)
  return np.hstack(features_arrays)


def assert_converges(
    converter: converters.TrialToArrayConverter,
    optimizer: vb.VectorizedOptimizer,
    score_fn: vb.BatchArrayScoreFunction,
    evaluations: int,
    alpha: float = 0.05,
    num_repeats: int = 3,
    success_threshold: int = 2,
) -> None:
  """Runs simple-regret convergence test.

  The assumption is that the global optimum of 'score_fn' is at [0,0,...,0].

  Arguments:
    converter: The converter.
    optimizer: The optimizer.
    score_fn: The score function to maximize.
    evaluations: The number of evaluations the optimizer performs.
    alpha: The significance level.
    num_repeats: The total number of times to run individual checks.
    success_threshold: The minimum number of success checks to consider PASS.

  Raises:
    Exception: in case the simple-regret convergence test failed.
  """
  success_count = 0
  for _ in range(num_repeats):
    optimum_features = randomize_features(converter)
    shifted_score_fn = lambda x, shift=optimum_features: score_fn(x - shift)
    optimizer.optimize(converter, shifted_score_fn)
    best_features = optimizer.strategy.best_features_results[0].features
    best_reward = optimizer.strategy.best_features_results[0].reward
    p_value_continuous, p_value_categorical = compute_p_value(
        best_features, optimum_features, converter, evaluations)
    msg = (
        f'P-value continuous: {p_value_continuous}. P-value categorical: {p_value_categorical}. '
        f'Alpha={alpha}. Best reward: {best_reward}.\nOptimum features:\n{optimum_features}'
        f'\nBest features:\n{best_features}.\nAbsolute diff:\n{np.abs(optimum_features - best_features)}'
    )
    if p_value_continuous <= alpha and p_value_categorical <= alpha:
      success_count += 1
      logging.info('Convergence test PASSED:\n %s', msg)
    else:
      logging.warning('Convergence test FAILED:\n %s', msg)

  if success_count < success_threshold:
    raise FailedSimpleRegretConvergenceTestError(
        f'{success_count} of the {num_repeats} convergence checks passed which '
        f'is below the threshold of {success_threshold}.')
