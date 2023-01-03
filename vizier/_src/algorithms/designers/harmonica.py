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

"""Harmonica algorithm for boolean search spaces from 'Hyperparameter Optimization: A Spectral Approach' (https://arxiv.org/abs/1706.00764).

This is a faithful re-implementation based off
https://github.com/callowbird/Harmonica.
"""
# pylint:disable=invalid-name
import functools
import itertools
from typing import Callable, Optional, Sequence, Set
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing

from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random


def _binary_subset_enumeration(dim: int,
                               indices: Sequence[int],
                               default_value: float = 1.0) -> np.ndarray:
  """Outputs all possible binary vectors from {-1, 1}^{dim} where only positions from `indices` are changed."""
  output = default_value * np.ones(
      shape=(2**len(indices), dim), dtype=np.float32)
  for i, binary in enumerate(
      itertools.product([-1.0, 1.0], repeat=len(indices))):
    output[i, indices] = binary
  return output


# TODO: Implement __repr__ via `attr`.
class PolynomialSparseRecovery:
  """Performs LASSO regression over low (d) degree polynomial coefficients."""

  def __init__(self,
               d: int = 3,
               num_top_monomials: int = 5,
               alpha: float = 3.0):
    """Init.

    Args:
      d: Maximum degree of monomials to use during polynomial regression.
      num_top_monomials: Number of top monomial coefficients to use.
      alpha: LASSO regularization coefficient.
    """
    self._d = d
    self._num_top_monomials = num_top_monomials
    self._alpha = alpha
    self.reset()

  def reset(self):
    self._poly_transformer = preprocessing.PolynomialFeatures(
        self._d, interaction_only=True)
    self._top_poly_indices: Optional[np.ndarray] = None
    self._top_poly_coefficients: Optional[np.ndarray] = None

  def regress(self, X: np.ndarray, Y: np.ndarray) -> None:
    """Performs LASSO regression to obtain top monomial coefficients."""

    # Computes monomial values for every vector in X.
    X_poly_features = []
    for x_vector in X:
      X_poly_features.append(
          self._poly_transformer.fit_transform(
              np.array(x_vector).reshape(1, -1))[0].tolist())
    X_poly_features = np.array(X_poly_features)

    # Find optimial coefficients on monomials to the data.
    lasso_solver = linear_model.Lasso(fit_intercept=True, alpha=self._alpha)
    lasso_solver.fit(X_poly_features, Y)

    # Sort and obtain top coefficients.
    lasso_coefficients = lasso_solver.coef_
    poly_indices = np.argsort(-np.abs(lasso_coefficients))
    self._top_poly_indices = poly_indices[:self._num_top_monomials]
    self._top_poly_coefficients = lasso_coefficients[self._top_poly_indices]

  def surrogate(self, x: np.ndarray) -> float:
    """Surrogate/Predicted function after regress()."""
    x_poly_features = self._poly_transformer.fit_transform(x.reshape(1, -1))
    top_x_poly_features = np.take_along_axis(
        x_poly_features[0], self._top_poly_indices, axis=0)
    return np.dot(top_x_poly_features, self._top_poly_coefficients)

  def index_set(self) -> Set[int]:
    """Returns parameter index set J from top coefficients.

    For example, if our monomials corresponding to top cofficients were x0*x1,
    x3, x1*x2*x3, then the output would be Union({0, 1}, {3}, {1, 2, 3}).
    """
    index_set = set()
    poly_feature_one_hots = self._poly_transformer.powers_
    for top_poly_index in self._top_poly_indices:
      active_indices = np.nonzero(
          poly_feature_one_hots[top_poly_index])[0].tolist()
      index_set = index_set | set(active_indices)
    return index_set


def _restricted_surrogate(x: np.ndarray, X_restrictors: np.ndarray,
                          replacement_indices: Sequence[int],
                          psr: PolynomialSparseRecovery):
  """New surrogate with input x's positions replaced from X_restrictor values."""
  objectives = []
  for x_restrictor in X_restrictors:
    x_copy = np.copy(x)
    x_copy[replacement_indices] = x_restrictor[replacement_indices]
    objectives.append(psr.surrogate(x_copy))
  return np.mean(objectives)


# TODO: Implement __repr__ via `attr`.
class HarmonicaQ:
  """Q-stage Harmonica.

  At each stage:
  1. Invoke PSR on the previous data (X,Y).
  2. Obtain t maximizers of the surrogate of the PSR.
  3. Redefine a 'restricted surrogate' using the t maximizers.
  4. Produce a new (X', Y') dataset via random search on this 'restricted
  surrogate'.
  """

  def __init__(self,
               psr: Optional[PolynomialSparseRecovery] = None,
               q: int = 10,
               t: int = 1,
               T: int = 300):
    """Init.

    Args:
      psr: PolynomialSparseRecovery class. If None, will be constructed with
        default arguments.
      q: Number of stages.
      t: Number of maximizers on the surrogate to use.
      T: Number of data samples to collect on the restricted surrogate.
    """
    self._psr = psr
    self._psr = psr or PolynomialSparseRecovery()
    self._q = q
    self._t = t
    self._T = T
    self.reset()

  def reset(self):
    self._restricted_surrogate: Optional[Callable[[np.ndarray], float]] = None
    self._psr.reset()

  def regress(self, X: np.ndarray, Y: np.ndarray) -> None:
    """Performs q-stage Harmonica."""
    num_vars = X.shape[-1]

    X_temp = X
    Y_temp = Y
    for _ in range(self._q):
      # Invoke PSR on data.
      self._psr.reset()
      self._psr.regress(X_temp, Y_temp)
      J = self._psr.index_set()

      # Perform brute force maximization to optain top t optimizers.
      all_X_in_J = _binary_subset_enumeration(num_vars, list(J))
      all_Y_in_J = np.array([self._psr.surrogate(x) for x in all_X_in_J])
      maximizer_idxs = np.argsort(all_Y_in_J)[-self._t:]
      X_maximizers = all_X_in_J[maximizer_idxs]

      # Define restricted surrogate and obtain data from it.
      self._restricted_surrogate = functools.partial(
          _restricted_surrogate,
          X_restrictors=X_maximizers,
          replacement_indices=list(J),
          psr=self._psr)
      X_temp = np.random.choice([-1.0, 1.0], size=(self._T, num_vars))
      Y_temp = np.array([self._restricted_surrogate(x) for x in X_temp])

  def surrogate(self, x: np.ndarray) -> float:
    if self._restricted_surrogate is None:
      raise ValueError('You must call regress() first.')
    return self._restricted_surrogate(x)


class HarmonicaDesigner(vza.Designer):
  """Harmonica Designer.

  The summary of the current implementation is as follows:

  1. Use previous trials for data collection.
  2. Perform Polynomial Lasso over the data and obtain a predictor function
  based on filtering only the highest coefficients.
  3. Obtain the "J"-set (i.e. set of input indices that only affect the
  predictor function).
  4. Find the global optimizer over the J-set domain on the predictor function,
  and return it as a suggestion.
  """

  def __init__(self,
               problem_statement: vz.ProblemStatement,
               harmonica_q: Optional[HarmonicaQ] = None,
               acquisition_samples: int = 100,
               num_init_samples: int = 10):
    """Init.

    Args:
      problem_statement: Must use a boolean search space.
      harmonica_q: HarmonicaQ class. If None, will use default class with
        default kwargs.
      acquisition_samples: Number of trial samples to optimize final acquisition
        function.
      num_init_samples: Number of initial random suggestions for seeding the
        model.
    """

    if problem_statement.search_space.is_conditional:
      raise ValueError(
          f'This designer {self} does not support conditional search.')
    for p_config in problem_statement.search_space.parameters:
      if p_config.external_type != vz.ExternalType.BOOLEAN:
        raise ValueError('Only boolean search spaces are supported.')

    self._problem_statement = problem_statement
    self._metric_name = self._problem_statement.metric_information.item().name
    self._search_space = problem_statement.search_space
    self._num_vars = len(self._search_space.parameters)

    self._harmonica_q = harmonica_q or HarmonicaQ()
    self._acquisition_samples = acquisition_samples
    self._num_init_samples = num_init_samples

    self._trials = []

  def update(self, trials: vza.CompletedTrials) -> None:
    self._trials += tuple(trials.completed)

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    """Performs entire q-stage Harmonica using previous trials for regression data.

    Args:
      count: Makes best effort to generate this many suggestions. If None,
        suggests as many as the algorithm wants.

    Returns:
      New suggestions.
    """
    count = count or 1
    if count > 1:
      raise ValueError('This designer does not support batched suggestions.')

    if len(self._trials) < self._num_init_samples:
      random_designer = random.RandomDesigner(self._search_space)
      return random_designer.suggest(count)

    # Convert previous trial data into regression data.
    X = []
    Y = []
    for t in self._trials:
      single_x = [
          1.0 if t.parameters.get_value(p.name) == 'True' else -1.0
          for p in self._search_space.parameters
      ]
      single_y = t.final_measurement.metrics[self._metric_name].value
      X.append(single_x)
      Y.append(single_y)
    X = np.array(X)
    Y = np.array(Y)

    if self._problem_statement.metric_information.item(
    ).goal == vz.ObjectiveMetricGoal.MINIMIZE:
      Y = -Y

    # Perform q-stage Harmonica.
    self._harmonica_q.reset()
    self._harmonica_q.regress(X, Y)

    # Optimize final acquisition function.
    # TODO: Allow any designer instead of just random search.
    X_temp = np.random.choice([-1.0, 1.0],
                              size=(self._acquisition_samples, self._num_vars))
    Y_temp = np.array([self._harmonica_q.surrogate(x) for x in X_temp])
    x_new = X_temp[np.argmax(Y_temp)]

    parameters = vz.ParameterDict()
    for i, p in enumerate(self._search_space.parameters):
      parameters[p.name] = 'True' if x_new[i] == 1.0 else 'False'
    return [vz.TrialSuggestion(parameters=parameters)]
