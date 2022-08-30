"""Bayesian Optimization of Combinatorial Structures (BOCS) from https://arxiv.org/abs/1806.08838.

Code is a cleaned-up version from /BOCSpy/ in https://github.com/baptistar/BOCS.
"""
# pylint:disable=invalid-name
import enum
import itertools
from typing import Callable, Optional, Sequence, Tuple, Union

from absl import logging
import cvxpy as cvx
import numpy as np

from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random

FloatType = Union[float, np.float32, np.float64]


class _BayesianHorseshoeLinearRegression:
  """Computes conditional poster parameter distributions from a sparsity-inducing prior."""

  def _fastmvg(self, Phi: np.ndarray, alpha: np.ndarray, D: np.ndarray):
    """Fast sampler for multivariate Gaussian distributions (large p, p > n) of the form N(mu, S).

    We have:
      mu = S Phi' y
      S  = inv(Phi'Phi + inv(D))
    Reference: https://arxiv.org/abs/1506.04778

    Args:
      Phi:
      alpha:
      D:

    Returns:
    """
    n, p = Phi.shape
    d = np.diag(D)
    u = np.random.randn(p) * np.sqrt(d)
    delta = np.random.randn(n)
    v = np.dot(Phi, u) + delta
    mult_vector = np.vectorize(np.multiply)
    Dpt = mult_vector(Phi.T, d[:, np.newaxis])
    w = np.linalg.solve(np.matmul(Phi, Dpt) + np.eye(n), alpha - v)
    return u + np.dot(Dpt, w)

  def _fastmvg_rue(self, Phi: np.ndarray, PtP: np.ndarray, alpha: np.ndarray,
                   D: np.ndarray) -> np.ndarray:
    """Another sampler for multivariate Gaussians (small p) of the form N(mu, S).

    We have:
      mu = S Phi' y
      S  = inv(Phi'Phi + inv(D))
    Here, PtP = Phi'*Phi (X'X is precomputed).

    Reference: https://www.jstor.org/stable/2680602

    Args:
      Phi:
      PtP:
      alpha:
      D:

    Returns:
    """
    p = Phi.shape[1]
    Dinv = np.diag(1. / np.diag(D))

    # Regularize PtP + Dinv matrix for small negative eigenvalues.
    try:
      L = np.linalg.cholesky(PtP + Dinv)
    except np.linalg.LinAlgError:
      mat = PtP + Dinv
      Smat = (mat + mat.T) / 2.
      maxEig_Smat = np.max(np.linalg.eigvals(Smat))
      L = np.linalg.cholesky(Smat + maxEig_Smat * 1e-15 * np.eye(Smat.shape[0]))

    v = np.linalg.solve(L, np.dot(Phi.T, alpha))
    m = np.linalg.solve(L.T, v)
    w = np.linalg.solve(L.T, np.random.randn(p))
    return m + w

  def regress(self, X, y, nsamples: int = 1000, burnin: int = 0, thin: int = 1):
    """Implementation of the Bayesian horseshoe linear regression hierarchy.

    References:
      https://arxiv.org/abs/1508.03884
      https://www.jstor.org/stable/25734098
      (c) Copyright Enes Makalic and Daniel F. Schmidt, 2015
      Adapted to python by Ricardo Baptista, 2018

    Args:
      X: regressor matrix [n x p]
      y: response vector  [n x 1]
      nsamples: number of samples for the Gibbs sampler (nsamples > 0)
      burnin: number of burnin (burnin >= 0)
      thin: thinning (thin >= 1)

    Returns:
      beta: regression parameters  [p x nsamples]
      b0: regression param. for constant [1 x nsamples]
      s2: noise variance sigma^2 [1 x nsamples]
      t2: hypervariance tau^2    [1 x nsamples]
      l2: hypervariance lambda^2 [p x nsamples]
    """

    n, p = X.shape

    # Standardize y's
    muY = np.mean(y)
    y = y - muY

    # Return values
    beta = np.zeros((p, nsamples))
    s2 = np.zeros((1, nsamples))
    t2 = np.zeros((1, nsamples))
    l2 = np.zeros((p, nsamples))

    # Initial values
    sigma2 = 1.
    lambda2 = np.random.uniform(size=p)
    tau2 = 1.
    nu = np.ones(p)
    xi = 1.

    # pre-compute X'*X (used with fastmvg_rue)
    XtX = np.matmul(X.T, X)

    # Gibbs sampler
    k = 0
    iter_count = 0
    while k < nsamples:

      # Sample from the conditional posterior distribution
      sigma = np.sqrt(sigma2)
      Lambda_star = tau2 * np.diag(lambda2)
      # Determine best sampler for conditional posterior of beta's
      if (p > n) and (p > 200):
        b = self._fastmvg(X / sigma, y / sigma, sigma2 * Lambda_star)
      else:
        b = self._fastmvg_rue(X / sigma, XtX / sigma2, y / sigma,
                              sigma2 * Lambda_star)

      # Sample sigma2
      e = y - np.dot(X, b)
      shape = (n + p) / 2.
      scale = np.dot(e.T, e) / 2. + np.sum(b**2 / lambda2) / tau2 / 2.
      sigma2 = 1. / np.random.gamma(shape, 1. / scale)

      # Sample lambda2
      scale = 1. / nu + b**2. / 2. / tau2 / sigma2
      lambda2 = 1. / np.random.exponential(1. / scale)

      # Sample tau2
      shape = (p + 1.) / 2.
      scale = 1. / xi + np.sum(b**2. / lambda2) / 2. / sigma2
      tau2 = 1. / np.random.gamma(shape, 1. / scale)

      # Sample nu
      scale = 1. + 1. / lambda2
      nu = 1. / np.random.exponential(1. / scale)

      # Sample xi
      scale = 1. + 1. / tau2
      xi = 1. / np.random.exponential(1. / scale)

      # Store samples
      iter_count += 1
      if iter_count > burnin:
        # thinning
        if (iter_count % thin) == 0:
          beta[:, k] = b
          s2[:, k] = sigma2
          t2[:, k] = tau2
          l2[:, k] = lambda2
          k = k + 1

    b0 = muY
    return beta, b0, s2, t2, l2


class _GibbsLinearRegressor:
  """Uses Gibbs sampling to produce an alpha."""

  def __init__(self, order: int, num_gibbs_retries: int = 10):
    self._order = order
    self._num_gibbs_retries = num_gibbs_retries

  def _preprocess(self,
                  X: np.ndarray,
                  y: np.ndarray,
                  inf_threshold: float = 1e6):
    """Preprocess data to ensure unique points and remove outliers."""
    # Limit data to unique points.
    unique_X, x_idx = np.unique(X, axis=0, return_index=True)
    unique_y = y[x_idx]

    # separate samples based on Inf output
    y_Infidx = np.where(np.abs(unique_y) > inf_threshold)[0]
    y_nInfidx = np.setdiff1d(np.arange(len(unique_y)), y_Infidx)

    X_train = unique_X[y_nInfidx, :]
    y_train = unique_y[y_nInfidx]

    return X_train, y_train

  def regress(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Samples alpha from the data."""
    # Preprocess data for training
    X_train, y_train = self._preprocess(X, y)

    # Create matrix with all covariates based on order.
    X_train = self._order_effects(X_train)
    (nSamps, nCoeffs) = X_train.shape

    # Check if X_train contains columns w/ zeros and find corresponding indices.
    check_zero = np.all(X_train == np.zeros((nSamps, 1)), axis=0)
    idx_nnzero = np.where(check_zero == False)[0]  # pylint:disable=singleton-comparison,g-explicit-bool-comparison

    # Remove columns of zeros in X_train.
    if np.any(check_zero):
      X_train = X_train[:, idx_nnzero]

    # Run Gibbs sampler.
    bhs = _BayesianHorseshoeLinearRegression()

    counter = 0
    while counter < self._num_gibbs_retries:
      # re-run if there is an error during sampling
      counter += 1
      try:
        alphaGibbs, a0, _, _, _ = bhs.regress(X_train, y_train)
        # run until alpha matrix does not contain any NaNs
        if not np.isnan(alphaGibbs).any():
          break
      except np.linalg.LinAlgError:
        logging.error('Numerical error during Gibbs sampling. Trying again.')
        continue

    if counter >= self._num_gibbs_retries:
      raise ValueError('Gibbs sampling failed for %d tries.' %
                       self._num_gibbs_retries)

    # append zeros back - note alpha(1,:) is linear intercept
    alpha_pad = np.zeros(nCoeffs)
    alpha_pad[idx_nnzero] = alphaGibbs[:, -1]
    return np.append(a0, alpha_pad)

  def surrogate_model(self, x: np.ndarray, alpha: np.ndarray) -> FloatType:
    """Surrogate model.

    Args:
      x: Should only contain one row.
      alpha: Coefficients on for the monomial terms.

    Returns:
      Surrogate objective.
    """

    # Generate x_all (all basis vectors) based on model order.
    x_all = np.append(1, self._order_effects(x))
    return np.dot(x_all, alpha)

  def _order_effects(self, X: np.ndarray) -> np.ndarray:
    """Function computes data matrix for all coupling."""

    # Find number of variables
    n_samp, n_vars = X.shape

    # Generate matrix to store results
    x_allpairs = X

    for ord_i in range(2, self._order + 1):

      # generate all combinations of indices (without diagonals)
      offdProd = np.array(
          list(itertools.combinations(np.arange(n_vars), ord_i)))

      # generate products of input variables
      x_comb = np.zeros((n_samp, offdProd.shape[0], ord_i))
      for j in range(ord_i):
        x_comb[:, :, j] = X[:, offdProd[:, j]]
      x_allpairs = np.append(x_allpairs, np.prod(x_comb, axis=2), axis=1)

    return x_allpairs


Objective = Callable[[np.ndarray], FloatType]


class _SimulatedAnnealing:
  """Simulated Annealing solver for acquisition functions."""

  def __init__(self,
               num_vars: int,
               num_iters: int = 10,
               num_reruns: int = 5,
               initial_temp: float = 1.0,
               annealing_factor: float = 0.8):
    self._num_vars = num_vars
    self._num_iters = num_iters
    self._num_reruns = num_reruns
    self._initial_temp = initial_temp
    self._annealing_factor = annealing_factor

  def argmin(self, objective: Objective) -> np.ndarray:
    """Computes argmin via multiple rounds of Simulated Annealing."""
    SA_model = np.zeros((self._num_reruns, self._num_vars))
    SA_obj = np.zeros(self._num_reruns)

    for j in range(self._num_reruns):
      optModel, objVals = self._optimization_loop(objective)
      SA_model[j, :] = optModel[-1, :]
      SA_obj[j] = objVals[-1]

    # Find optimal solution
    min_idx = np.argmin(SA_obj)
    x_new = SA_model[min_idx, :]
    return x_new

  def _optimization_loop(self,
                         objective: Objective) -> Tuple[np.ndarray, np.ndarray]:
    """Single optimization round of Simulated Annealing."""

    # Declare vectors to save solutions
    model_iter = np.zeros((self._num_iters, self._num_vars))
    obj_iter = np.zeros(self._num_iters)

    # Set initial temperature and cooling schedule
    T = self._initial_temp
    cool = lambda t: self._annealing_factor * t

    # Set initial condition and evaluate objective
    old_x = np.zeros((1, self._num_vars))
    old_obj = objective(old_x)

    # Set best_x and best_obj
    best_x = old_x
    best_obj = old_obj

    # Run simulated annealing
    for t in range(self._num_iters):

      # Decrease T according to cooling schedule.
      T = cool(T)

      # Find new sample
      flip_bit = np.random.randint(self._num_vars)
      new_x = old_x.copy()
      new_x[0, flip_bit] = 1. - new_x[0, flip_bit]

      # Evaluate objective function.
      new_obj = objective(new_x)

      # Update current solution iterate.
      if (new_obj < old_obj) or (np.random.rand() < np.exp(
          (old_obj - new_obj) / T)):
        old_x = new_x
        old_obj = new_obj

      # Update best solution
      if new_obj < best_obj:
        best_x = new_x
        best_obj = new_obj

      # Save solution
      model_iter[t, :] = best_x
      obj_iter[t] = best_obj

    return model_iter, obj_iter


class _SemiDefiniteProgramming:
  """SDP solver for quadratic acquisition functions."""

  def __init__(self,
               num_vars: int,
               lamda: float = 1e-4,
               num_repeats: int = 100):
    self._num_vars = num_vars
    self._lamda = lamda
    self._num_repeats = num_repeats

  def argmin(self, alpha: np.ndarray) -> np.ndarray:
    """Perform SDP over the quadratic xt*A*x + bt*x.

    (A,b) is recovered from alpha.

    Args:
      alpha: Obtained from linear regression.

    Returns:
      Argmin of the SDP problem.
    """

    # Extract vector of coefficients
    b = alpha[1:self._num_vars + 1] + self._lamda
    a = alpha[self._num_vars + 1:]

    # Get indices for quadratic terms.
    idx_prod = np.array(
        list(itertools.combinations(np.arange(self._num_vars), 2)))
    n_idx = idx_prod.shape[0]

    # Check number of coefficients
    if a.size != n_idx:
      raise ValueError('Number of Coefficients does not match indices!')

    # Convert a to matrix form
    A = np.zeros((self._num_vars, self._num_vars))
    for i in range(n_idx):
      A[idx_prod[i, 0], idx_prod[i, 1]] = a[i] / 2.
      A[idx_prod[i, 1], idx_prod[i, 0]] = a[i] / 2.

    # Convert to standard form.
    bt = b / 2. + np.dot(A, np.ones(self._num_vars)) / 2.
    bt = bt.reshape((self._num_vars, 1))
    At = np.vstack((np.append(A / 4., bt / 2., axis=1), np.append(bt.T, 2.)))

    # Run SDP relaxation.
    X = cvx.Variable((self._num_vars + 1, self._num_vars + 1), PSD=True)
    obj = cvx.Minimize(cvx.trace(cvx.matmul(At, X)))
    constraints = [cvx.diag(X) == np.ones(self._num_vars + 1)]
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.CVXOPT)

    # Extract vectors and compute Cholesky.
    try:
      L = np.linalg.cholesky(X.value)
    except np.linalg.LinAlgError:
      XpI = X.value + 1e-15 * np.eye(self._num_vars + 1)
      L = np.linalg.cholesky(XpI)

    suggest_vect = np.zeros((self._num_vars, self._num_repeats))
    obj_vect = np.zeros(self._num_repeats)

    for kk in range(self._num_repeats):
      # Generate a random cutting plane vector (uniformly distributed on the
      # unit sphere - normalized vector)
      r = np.random.randn(self._num_vars + 1)
      r = r / np.linalg.norm(r)
      y_soln = np.sign(np.dot(L.T, r))

      # Convert solution to original domain and assign to output vector.
      suggest_vect[:, kk] = (y_soln[:self._num_vars] + 1.) / 2.
      obj_vect[kk] = np.dot(
          np.dot(suggest_vect[:, kk].T, A), suggest_vect[:, kk]) + np.dot(
              b, suggest_vect[:, kk])

    # Find optimal rounded solution.
    opt_idx = np.argmin(obj_vect)
    return suggest_vect[:, opt_idx]


@enum.unique
class AcqusitionOptimizer(enum.Enum):
  """Type of acquisition optimizer."""
  SA = 'Simulated Annealing'
  SDP = 'Semidefinite Programming'


class BOCSDesigner(vza.Designer):
  """BOCS Designer."""

  def __init__(self,
               problem_statement: vz.ProblemStatement,
               order: int = 2,
               acquisition_mode: AcqusitionOptimizer = AcqusitionOptimizer.SDP,
               lamda: float = 1e-4,
               num_initial_randoms: int = 10):
    """Init.

    Args:
      problem_statement: Must use a boolean search space.
      order: Statistical model order.
      acquisition_mode: Which acquisition optimizer to use.
      lamda: Sparsity regularization coefficient.
      num_initial_randoms: Number of initial random suggestions for seeding the
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
    self._current_index = 0

    self._order = order
    self._acquisition_mode = acquisition_mode
    self._lamda = lamda
    self._num_vars = len(self._search_space.parameters)
    self._num_initial_randoms = num_initial_randoms

    self._trials = []

  def update(self, trials: vza.CompletedTrials) -> None:
    self._trials += tuple(trials.completed)

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    """Core BOCS method.

    Initially will use random search for the first `num_initial_randoms`
    suggestions, and then will use SDP or Simulated Annealing for acqusition
    minimization.

    Args:
      count: Makes best effort to generate this many suggestions. If None,
        suggests as many as the algorithm wants.

    Returns:
      New suggestions.
    """
    count = count or 1
    if count > 1:
      raise ValueError('This optimizer does not support batched suggestions.')

    if len(self._trials) < self._num_initial_randoms:
      random_designer = random.RandomDesigner(self._search_space)
      return random_designer.suggest(count)

    X = []
    y = []
    for t in self._trials:
      single_X = [(t.parameters.get_value(p.name) == 'True')
                  for p in self._search_space.parameters]
      single_y = t.final_measurement.metrics[self._metric_name].value
      X.append(single_X)
      y.append(single_y)
    X = np.array(X)
    y = np.array(y)

    if self._problem_statement.metric_information.item(
    ).goal == vz.ObjectiveMetricGoal.MAXIMIZE:
      y = -y

    # Train initial statistical model
    lin_reg = _GibbsLinearRegressor(self._order)
    alpha = lin_reg.regress(X, y)

    # Run SA optimization
    if self._acquisition_mode == AcqusitionOptimizer.SA:
      # Setup acquisition function objective for SA.
      penalty = lambda x: self._lamda * np.sum(x, axis=1)
      acquisition_fn = lambda x: lin_reg.surrogate_model(x, alpha) + penalty(x)
      SA = _SimulatedAnnealing(self._num_vars)
      x_new = SA.argmin(acquisition_fn)

    # Run semidefinite relaxation for order 2 model with l1 loss
    elif self._acquisition_mode == AcqusitionOptimizer.SDP:
      if self._order != 2:
        raise ValueError('SDPs only use order=2.')
      sdp = _SemiDefiniteProgramming(self._num_vars, self._lamda)
      x_new = sdp.argmin(alpha)

    parameters = vz.ParameterDict()
    for i, p in enumerate(self._search_space.parameters):
      parameters[p.name] = 'True' if x_new[i] == 1.0 else 'False'
    return [vz.TrialSuggestion(parameters=parameters)]
