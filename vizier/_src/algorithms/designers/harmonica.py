"""Harmonica algorithm for boolean search spaces from 'Hyperparameter Optimization: A Spectral Approach' (https://arxiv.org/abs/1706.00764).

This is a faithful re-implementation based off
https://github.com/callowbird/Harmonica.
"""
# pylint:disable=invalid-name
import itertools
from typing import Optional, Sequence, Set
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


class PolynomialSparseRecovery:
  """Performs LASSO regression over low (d) degree polynomial coefficients."""

  def __init__(self, d: int, num_top_monomials: int, alpha: float):
    self._d = d
    self._num_top_monomials = num_top_monomials
    self._alpha = alpha

    self._poly_transformer = preprocessing.PolynomialFeatures(
        self._d, interaction_only=True)

    self._top_poly_indices: Optional[np.ndarray] = None

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


# TODO: Finish the q-stage variant.
class HarmonicaQ:

  def __init__(self, q: int = 10, t: int = 10):
    self._q = q
    self._t = t


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
               d: int = 3,
               num_top_monomials: int = 5,
               alpha: float = 1.0,
               num_init_samples: int = 10):
    """Init.

    Args:
      problem_statement: Must use a boolean search space.
      d: Maximum degree of monomials to use during polynomial regression.
      num_top_monomials: Number of top monomial coefficients to use.
      alpha: LASSO regularization coefficient.
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
    self._d = d
    self._num_top_monomials = num_top_monomials
    self._alpha = alpha
    self._num_init_samples = num_init_samples
    self._trials = []

  def update(self, trials: vza.CompletedTrials) -> None:
    self._trials += tuple(trials.completed)

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    """Harmonica.

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

    psr = PolynomialSparseRecovery(
        d=self._d, num_top_monomials=self._num_top_monomials, alpha=self._alpha)
    psr.regress(X, Y)
    J = psr.index_set()

    # TODO: Swap brute force optimization with any designer.
    all_X_in_J = _binary_subset_enumeration(self._num_vars, list(J))
    all_Y_in_J = np.array([psr.surrogate(x) for x in all_X_in_J])
    opt_idx = np.argmax(all_Y_in_J)
    x_new = all_X_in_J[opt_idx]

    parameters = vz.ParameterDict()
    for i, p in enumerate(self._search_space.parameters):
      parameters[p.name] = 'True' if x_new[i] == 1.0 else 'False'
    return [vz.TrialSuggestion(parameters=parameters)]
