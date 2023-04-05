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

"""High-level wrappers for stochastic process hyperparameter optimizers."""

# TODO: Add optimizers that are frequently used in the literature.

import functools
import time
from typing import Generic, Optional, Protocol, TypeVar

from absl import logging
import attr
import chex
import jax
from jax import numpy as jnp
from jax import random
import jaxopt
import optax
from tensorflow_probability.substrates import jax as tfp
import tree
from vizier._src.jax import stochastic_process_model as sp

tfb = tfp.bijectors

_Params = TypeVar('_Params', bound=chex.ArrayTree)
OptState = chex.ArrayTree


class Setup(Protocol, Generic[_Params]):
  """Set up the model parameters given RNG key."""

  def __call__(self, rng: jax.random.KeyArray) -> _Params:
    """Set up the model parameters given RNG key."""
    pass


class LossFunction(Protocol, Generic[_Params]):
  """Evaluates model params and returns (loss, dict of auxiliary metrics)."""

  def __call__(self, params: _Params) -> tuple[jax.Array, chex.ArrayTree]:
    """Evaluates model params and returns (loss, dict of auxiliary metrics)."""
    pass


class Optimizer(Protocol, Generic[_Params]):
  """Optimizes the LossFunction.

  Example:

  ```python
  setup: Setup = lambda rng: jax.random.uniform(rng, minval=-5, maxval=5)

  def loss_fn(xs):  # satisfies `LossFunction` Protocol
    xs = jax.nn.sigmoid(xs)
    return jnp.cos(xs * 5 * 2 * jnp.pi) * (2 - 5 * (xs - 0.5)**2), dict()

  optimize = optimizers.OptaxTrainWithRandomRestarts(
      optax.adam(5e-3), epochs=500, verbose=True, random_restarts=50)
  optimal_params, metrics = optimize(setup, loss_fn, jax.random.PRNGKey(0))
  ```
  """

  def __call__(
      self,
      setup: Setup[_Params],
      loss_fn: LossFunction[_Params],
      rng: jax.random.KeyArray,
      *,
      constraints: Optional[sp.Constraint] = None,
  ) -> tuple[_Params, chex.ArrayTree]:
    """Optimizes a LossFunction expecting Params as input.

    When constraint bijectors are applied, note that the returned parameters are
    in the constrained space (the parameter domain), not the unconstrained space
    over which the optimization takes place.

    If the Optimizer uses lower and upper bounds, then it is responsible for
    converting `None` bounds to `+inf` or `-inf` as necessary.

    Args:
      setup: Generates initial points.
      loss_fn: Evaluates a point.
      rng: JAX PRNGKey.
      constraints: Parameter constraints.

    Returns:
      Tuple containing optimal input in the constrained space and optimization
      metrics.
    """
    pass


@attr.define
class OptaxTrainWithRandomRestarts(Optimizer[_Params]):
  """Wraps an Optax optimizer.

  It's recommended to use this optimizer with a loss function that normalizes by
  the number of observations. The unnormalized loss function for parameters `p`
  is typically of the form

  ```None
  -gp_likelihood(observations | p) + regularization(p)
  ```

  where regularization may be a negative prior log probability. The likelihood
  term is approximately proportional to the number of observations. As the
  number of observations changes over the course of a study, dividing the loss
  by this number helps ensure that loss values are roughly of the same order of
  magnitude, such that a constant learning rate may be used for gradient-based
  optimizers. Vizier library models make this adjustment automatically.

  Attributes:
    optimizer: Optax optimizer such as `optax.adam(1e-2)`.
    epochs: Number of train epochs.
    verbose: If >=1, logs the train progress. If >=2, logs the gradients.
    random_restarts: Must be positive; number of random initializations for the
      optimization.
    best_n: Number of best values to return from the initializations; must be
      less than or equal to `random_restarts`.
  """

  optimizer: optax.GradientTransformation = attr.field()
  epochs: int = attr.field(kw_only=True)
  verbose: int = attr.field(kw_only=True, default=0, converter=int)
  random_restarts: int = attr.field(kw_only=True, default=32)
  best_n: int = attr.field(kw_only=True, default=1)

  def __attrs_post_init__(self):
    if self.random_restarts < self.best_n:
      raise ValueError(
          f'Cannot generate {self.best_n} results from'
          f' {self.random_restarts} restarts'
      )

  def __call__(
      self,
      setup: Setup[_Params],
      loss_fn: LossFunction[_Params],
      rng: jax.random.KeyArray,
      *,
      constraints: Optional[sp.Constraint] = None,
  ) -> tuple[_Params, chex.ArrayTree]:
    if constraints is None or constraints.bijector is None:
      bijector = None
      unconstrained_loss_fn = loss_fn
    else:
      bijector = constraints.bijector
      unconstrained_loss_fn = lambda x: loss_fn(bijector(x))

    grad_fn = jax.value_and_grad(unconstrained_loss_fn, has_aux=True)

    def _setup_all(rng: jax.random.KeyArray) -> tuple[_Params, OptState]:
      """Sets up both model params and optimizer state."""
      params = setup(rng)
      if bijector is not None:
        params = bijector.inverse(params)
      opt_state = self.optimizer.init(params)
      return params, opt_state

    def _train_step(
        params: _Params, opt_state: OptState
    ) -> tuple[_Params, OptState, chex.ArrayTree]:
      """One train step."""
      (loss, metrics), grads = grad_fn(params)
      logging.log_if(logging.INFO, 'gradients: %s', self.verbose >= 2, grads)
      updates, opt_state = self.optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      metrics['loss'] = loss
      return params, opt_state, metrics

    if self.random_restarts > 1:
      # Random restarts are implemented via jax.vmap.
      # Note that both setup_all and train_step are vmapped.
      rngs = random.split(rng, self.random_restarts)
      params, opt_state = jax.vmap(_setup_all)(rngs)
      train_step = jax.vmap(_train_step)
    else:
      params, opt_state = _setup_all(rng)

    logging.info('Initialized parameters. %s',
                 jax.tree_map(lambda x: x.shape, params))

    # See https://jax.readthedocs.io/en/latest/faq.html#buffer-donation.
    train_step = jax.jit(train_step, donate_argnums=[0, 1])
    metrics = []
    for epoch in range(self.epochs):
      params, opt_state, step_metrics = train_step(params, opt_state)
      logging.log_if(
          logging.INFO,
          'Epoch %s: metrics: %s',
          self.verbose >= 1,
          epoch,
          step_metrics,
      )
      metrics.append(step_metrics)

    # Convert `metrics` from a list of dicts to a dict of arrays with leftmost
    # dimension corresponding to train steps.
    outer_treedef = jax.tree_util.tree_structure([0] * self.epochs)
    transposed_metrics = jax.tree_util.tree_transpose(
        outer_treedef, jax.tree_util.tree_structure(step_metrics), metrics
    )
    metrics = jax.tree_util.tree_map(
        jnp.array,
        transposed_metrics,
        is_leaf=lambda x: jax.tree_util.tree_structure(x) == outer_treedef,
    )

    final_losses = metrics['loss'][-1]
    logging.info('Final loss: %s', final_losses)
    # Extract the best only.
    argsorted = jnp.argsort(final_losses)
    logging.info('Best loss(es): %s', final_losses[argsorted[: self.best_n]])
    params = jax.tree_map(lambda x: x[argsorted[: self.best_n]], params)
    if self.best_n == 1:
      params = jax.tree_map(functools.partial(jnp.squeeze, axis=0), params)
    if bijector is not None:
      params = bijector(params)
    return params, metrics


@attr.define
class JaxoptLbfgsB(Optimizer[_Params]):
  """Jaxopt's L-BFGS-B optimizer.

  Jaxopt calls Scipy's L-BFGS-B, which wraps a Fortran implementation.
  L-BFGS-B is a version of L-BFGS that incorporates box constraints on
  parameters.
  https://digital.library.unt.edu/ark:/67531/metadc666315/m2/1/high_res_d/204262.pdf

  Attributes:
    num_line_search_steps: Maximum number of line search steps.
    random_restarts: Must be positive; number of random initializations for the
      optimization.
    best_n: Number of best values to return from the initializations; must be
      less than or equal to `random_restarts`.
    _speed_test: If True, return speed test results.
  """

  num_line_search_steps: int = attr.field(kw_only=True, default=20)
  random_restarts: int = attr.field(kw_only=True, default=4)
  best_n: int = attr.field(kw_only=True, default=1)
  _speed_test: bool = attr.field(kw_only=True, default=False)

  def __attrs_post_init__(self):
    if self.random_restarts < self.best_n:
      raise ValueError(
          f'Cannot generate {self.best_n} results from'
          f' {self.random_restarts} restarts'
      )

  def __call__(
      self,
      setup: Setup[_Params],
      loss_fn: LossFunction[_Params],
      rng: jax.random.KeyArray,
      *,
      constraints: Optional[sp.Constraint] = None,
  ) -> tuple[_Params, chex.ArrayTree]:
    # L-BFGS-B may be used on unconstrained problems (in which case it is
    # slightly different from L-BFGS, in that it uses the Cauchy point/subspace
    # minimization to choose the line search direction). Bounds must be None or
    # a tuple of size 2. The tuple must contain lower/upper bounds, which may be
    # None or a pytree with the same structure as the model parameters returned
    # by setup (otherwise Jaxopt will raise an error).
    p = setup(jax.random.PRNGKey(0))
    bounds = None if constraints is None else constraints.bounds
    is_leaf = lambda x: jax.tree_util.treedef_is_leaf(  # pylint: disable=g-long-lambda
        jax.tree_util.tree_structure(x)
    )

    def _none_to_inf(b, inf):
      """Converts None bounds to inf or -inf to pass to the optimizer."""
      if b is None:
        b = inf
      if is_leaf(b):
        # Broadcast scalars to the parameter structure.
        return tree.map_structure(lambda x: b * jnp.ones_like(x), p)
      else:
        # For structured bounds, replace `None`s in the structure with `inf`.
        return tree.map_structure(
            lambda b_, x: jnp.ones_like(x) * (inf if b_ is None else b_), b, p
        )

    if bounds is not None:
      lb = _none_to_inf(bounds[0], -jnp.inf)
      ub = _none_to_inf(bounds[1], jnp.inf)
      bounds = (lb, ub)

    loss = lambda p: loss_fn(p)[0]
    lbfgsb = jaxopt.ScipyBoundedMinimize(
        fun=loss,
        method='L-BFGS-B',
        options={'maxls': self.num_line_search_steps},
    )

    metrics = {}
    if self._speed_test:
      start_time = time.time()
      _ = lbfgsb.run(init_params=p, bounds=bounds)
      metrics['compile_time'] = time.time() - start_time

    train_times = []
    losses = []
    params = []
    for i, rng in enumerate(jax.random.split(rng, num=self.random_restarts)):
      with jax.profiler.StepTraceAnnotation('train', step_num=i):
        p = setup(rng)
        start_time = time.time()
        # TODO: Avoid using `run` since it re-JITs unnecessarily.
        position, opt_state = lbfgsb.run(init_params=p, bounds=bounds)
        train_times.append(time.time() - start_time)
        losses.append(opt_state.fun_val)
        params.append(position)
        logging.info('Loss: %s', opt_state.fun_val)
        logging.info('Last step time: %s', train_times[-1])

    all_params = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *params)
    losses = jnp.array(losses)
    argsorted = jnp.argsort(losses)
    logging.info('Best loss(es): %s', losses[argsorted[: self.best_n]])
    optimal_params = jax.tree_util.tree_map(
        lambda p: p[argsorted[: self.best_n]], all_params
    )
    if self.best_n == 1:
      optimal_params = jax.tree_map(
          functools.partial(jnp.squeeze, axis=0), optimal_params
      )

    metrics['loss'] = losses
    if self._speed_test:
      metrics['train_time'] = train_times
    return optimal_params, metrics
