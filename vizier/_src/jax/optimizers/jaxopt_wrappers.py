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

import time
from typing import Optional

from absl import logging
import attr
import chex
from flax import struct
import jax
from jax import numpy as jnp
import jaxopt
import tree
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax.optimizers import core


@struct.dataclass
class LbfgsBOptions:
  """L-BFGS-B is a version of L-BFGS that incorporates box constraints.

  https://digital.library.unt.edu/ark:/67531/metadc666315/m2/1/high_res_d/204262.pdf

  Attributes:
    num_line_search_steps: Maximum number of line search steps.
    random_restarts: Must be positive; number of random initializations for the
      optimization.
    tol: Tolerance for stopping criteria.
    maxiter: Max number of iterations.
    best_n: Number of best values to return from the initializations; must be
      less than or equal to `random_restarts`.
  """

  num_line_search_steps: int = struct.field(kw_only=True, default=20)
  random_restarts: int = struct.field(kw_only=True, default=4)
  tol: float = struct.field(kw_only=True, default=1e-8)
  maxiter: int = struct.field(kw_only=True, default=50)
  best_n: int = struct.field(kw_only=True, default=1, pytree_node=False)

  def __post_init__(self):
    if self.random_restarts < self.best_n:
      raise ValueError(
          f'Cannot generate {self.best_n} results from'
          f' {self.random_restarts} restarts'
      )


def _is_leaf(x: float) -> bool:
  return jax.tree_util.treedef_is_leaf(jax.tree_util.tree_structure(x))


def _none_to_inf(b: float, inf: float, params: chex.ArrayTree):
  """Converts None bounds to inf or -inf to pass to the optimizer."""
  if b is None:
    b = inf
  if _is_leaf(b):
    # Broadcast scalars to the parameter structure.
    return tree.map_structure(lambda x: b * jnp.ones_like(x), params)
  else:
    # For structured bounds, replace `None`s in the structure with `inf`.
    return tree.map_structure(
        lambda b_, x: jnp.ones_like(x) * (inf if b_ is None else b_), b, params
    )


# TODO: Remove support for broadcasting array -> arraytree
# and reduce maintenance burden. Alternatively, move to a separate file
# so other libraries can use it too.
def _get_bounds(
    setup: core.Setup[core.Params],
    constraints: Optional[sp.Constraint],
) -> Optional[tuple[chex.ArrayTree, chex.ArrayTree]]:
  """Returns (Lower, upper) ArrayTrees with the same shape as params."""
  if constraints is None:
    return None
  else:
    params = setup(jax.random.PRNGKey(0))
    lb = _none_to_inf(constraints.bounds[0], -jnp.inf, params)
    ub = _none_to_inf(constraints.bounds[1], jnp.inf, params)
    logging.info('constraints: %s, bounds: %s', constraints, (lb, ub))
    return (lb, ub)


@attr.define
class JaxoptScipyLbfgsB(core.Optimizer[core.Params]):
  """Jaxopt's wrapper for scipy L-BFGS-B optimizer."""

  _options: LbfgsBOptions = attr.field(default=LbfgsBOptions())
  _speed_test: bool = attr.field(kw_only=True, default=False)

  def __call__(
      self,
      setup: core.Setup[core.Params],
      loss_fn: core.LossFunction[core.Params],
      rng: jax.random.KeyArray,
      *,
      constraints: Optional[sp.Constraint] = None,
  ) -> tuple[core.Params, chex.ArrayTree]:
    # L-BFGS-B may be used on unconstrained problems (in which case it is
    # slightly different from L-BFGS, in that it uses the Cauchy point/subspace
    # minimization to choose the line search direction). Bounds must be None or
    # a tuple of size 2. The tuple must contain lower/upper bounds, which may be
    # None or a pytree with the same structure as the model parameters returned
    # by setup (otherwise Jaxopt will raise an error).

    # Pre-jit the loss function, to avoid retracing it when jitting
    # value_and_grad.
    loss_fn = jax.jit(loss_fn)
    lbfgsb = jaxopt.ScipyBoundedMinimize(
        fun=loss_fn,
        method='L-BFGS-B',
        maxiter=self._options.maxiter,
        options={
            'gtol': self._options.tol,
            'maxls': self._options.num_line_search_steps,
        },
        jit=False,
        has_aux=True,
        value_and_grad=jax.jit(jax.value_and_grad(loss_fn, has_aux=True)),
    )
    train_times = []
    losses = []
    params = []
    metrics = {}

    bounds = _get_bounds(setup, constraints)

    logging.info(
        'Using SCIPY L-BFGS-B w/ %d restarts.', self._options.random_restarts
    )
    for rng in jax.random.split(rng, num=self._options.random_restarts):
      p = setup(rng)
      start_time = time.time()
      position, opt_state = lbfgsb.run(init_params=p, bounds=bounds)
      train_times.append(time.time() - start_time)
      losses.append(opt_state.fun_val)
      params.append(position)
      logging.info(
          'Loss: %s, last step time: %s', opt_state.fun_val, train_times[-1]
      )
    losses = jnp.asarray(losses)
    all_params = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *params)

    metrics['loss'] = losses
    if self._speed_test:
      metrics['train_time'] = train_times
    return (
        core.get_best_params(losses, all_params, best_n=self._options.best_n),
        metrics,
    )

  # TODO: Remove dependencies on this property.
  @property
  def best_n(self) -> int:
    return self._options.best_n


@attr.define
class JaxoptLbfgsB(core.Optimizer[core.Params]):
  """Jaxopt's L-BFGS-B optimizer.

  Jaxopt calls the Scipy or Jax implementation of L-BFGS-B.
  L-BFGS-B is a version of L-BFGS that incorporates box constraints on
  parameters.
  https://digital.library.unt.edu/ark:/67531/metadc666315/m2/1/high_res_d/204262.pdf

  Attributes:
    num_line_search_steps: Maximum number of line search steps.
    random_restarts: Must be positive; number of random initializations for the
      optimization.
    best_n: Number of best values to return from the initializations; must be
      less than or equal to `random_restarts`.
     use_scipy: Uses the scipy version of L-BFGS-B. If False, uses the pure JAX
       L-BFGS-B in Jaxopt, which runs on accelerators.
     tol: Tolerance for stopping criteria.
     maxiter: Max number of iterations.
    _speed_test: If True, return speed test results.
  """

  _options: LbfgsBOptions = attr.field(default=LbfgsBOptions())
  _speed_test: bool = attr.field(kw_only=True, default=False)

  def __call__(
      self,
      setup: core.Setup[core.Params],
      loss_fn: core.LossFunction[core.Params],
      rng: jax.random.KeyArray,
      *,
      constraints: Optional[sp.Constraint] = None,
  ) -> tuple[core.Params, chex.ArrayTree]:
    metrics = {}
    lbfgsb = jaxopt.LBFGSB(
        fun=loss_fn,
        value_and_grad=False,
        maxls=self._options.num_line_search_steps,
        tol=self._options.tol,
        maxiter=self._options.maxiter,
        has_aux=True,
        jit=True,
        unroll=True,
    )

    logging.info(
        'Using JAX L-BFGS-B w/ %d restarts.', self._options.random_restarts
    )
    rngs = jax.random.split(rng, num=self._options.random_restarts)
    init_params = jax.vmap(setup)(rngs)
    start_time = time.time()
    bounds = _get_bounds(setup, constraints)

    all_params, opt_states = jax.lax.map(
        lambda p: lbfgsb.run(init_params=p, bounds=bounds),
        init_params,
    )  # pylint: disable=g-long-lambda
    losses = jnp.asarray(opt_states.value)
    metrics['train_time'] = time.time() - start_time
    metrics['loss'] = losses
    return (
        core.get_best_params(losses, all_params, best_n=self._options.best_n),
        metrics,
    )

  # TODO: Remove dependencies on this property.
  @property
  def best_n(self) -> int:
    return self._options.best_n

  @property
  def jittable(self) -> bool:
    return True
