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

"""High-level wrappers for stochastic process hyperparameter optimizers."""

import datetime
import time
from typing import Any, Optional

from absl import logging
import attr
import chex
import equinox as eqx
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
  # TODO: Remove this since now it's a part of optimizer API.
  random_restarts: Optional[int] = struct.field(
      kw_only=True, default=None, pytree_node=False
  )
  tol: float = struct.field(kw_only=True, default=1e-8)
  maxiter: int = struct.field(kw_only=True, default=50)
  # TODO: Remove best_n. Now it's a part of optimizer API.
  best_n: Optional[int] = struct.field(
      kw_only=True, default=None, pytree_node=False
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
    params: core.Params,
    constraints: Optional[sp.Constraint],
) -> Optional[tuple[chex.ArrayTree, chex.ArrayTree]]:
  """Returns (Lower, upper) ArrayTrees with the same shape as params."""
  if constraints is None:
    return None
  else:
    lb = _none_to_inf(constraints.bounds[0], -jnp.inf, params)
    ub = _none_to_inf(constraints.bounds[1], jnp.inf, params)
    logging.info(
        'constraints\n: %s converted to bounds:\n %s', constraints, (lb, ub)
    )
    return (lb, ub)


def _unbatch_params(batched_params: core.Params) -> list[core.Params]:
  batch_size = jax.tree_util.tree_leaves(batched_params)[0].shape[0]
  return list(
      map(
          lambda i: jax.tree.map(lambda p: p[i], batched_params),
          range(batch_size),
      )
  )


@attr.define
class JaxoptScipyLbfgsB(core.Optimizer[core.Params]):
  """Jaxopt's wrapper for scipy L-BFGS-B optimizer."""

  _options: LbfgsBOptions = attr.field(default=LbfgsBOptions())
  _speed_test: bool = attr.field(kw_only=True, default=False)
  _max_duration: Optional[datetime.timedelta] = None

  def __call__(
      self,
      init_params: core.Params,
      loss_fn: core.LossFunction[core.Params],
      rng: jax.Array,
      *,
      constraints: Optional[sp.Constraint] = None,
      best_n: Optional[int] = None,
  ) -> tuple[core.Params, chex.ArrayTree]:
    # L-BFGS-B is deterministic given initial parameters.
    del rng

    # L-BFGS-B may be used on unconstrained problems (in which case it is
    # slightly different from L-BFGS, in that it uses the Cauchy point/subspace
    # minimization to choose the line search direction). Bounds must be None or
    # a tuple of size 2. The tuple must contain lower/upper bounds, which may be
    # None or a pytree with the same structure as the model parameters returned
    # by setup (otherwise Jaxopt will raise an error).
    jax.monitoring.record_event('/vizier/jax/optimizers/scipylbfgsb/called')
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
        value_and_grad=eqx.filter_jit(
            eqx.filter_value_and_grad(loss_fn, has_aux=True)
        ),
    )
    train_times = []
    losses = []
    params = []
    metrics = {}

    init_params = _unbatch_params(init_params)
    bounds = _get_bounds(init_params[0], constraints)

    logging.info(
        'Using SCIPY L-BFGS-B w/ %d initializations: %s',
        len(init_params),
        init_params,
    )
    for p in init_params:
      if self._max_duration is not None and train_times:
        expected_worst_case_duration = max(train_times) + sum(train_times)
        if expected_worst_case_duration >= self._max_duration.total_seconds():
          logging.warn(
              'Maximum duration of %s is expected to be surpassed.'
              ' Completed only %s initializations out of %s.',
              self._max_duration,
              len(params),
              len(init_params),
          )
          break
      start_time = time.time()
      position, opt_state = lbfgsb.run(init_params=p, bounds=bounds)
      train_times.append(time.time() - start_time)
      losses.append(opt_state.fun_val)
      params.append(position)
      logging.info(
          'Loss: %s, last step time: %s, ScipyMinimizeInfo: %s',
          opt_state.fun_val,
          train_times[-1],
          opt_state,
      )
    losses = jnp.asarray(losses)
    all_params = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *params)

    metrics['loss'] = losses[jnp.newaxis, :]
    if self._speed_test:
      metrics['train_time'] = train_times
    return (
        core.get_best_params(losses, all_params, best_n=best_n),
        metrics,
    )

  # TODO: Remove dependencies on this property.
  @property
  def best_n(self) -> int:
    return self._options.best_n or 1


def _run_parallel_lbfgs(
    loss_fn: core.LossFunction[core.Params],
    init_params_batch: core.Params,
    *,
    bounds: Optional[tuple[chex.ArrayTree, chex.ArrayTree]],
    options: LbfgsBOptions,
) -> tuple[core.Params, Any]:
  """Called by JaxoptLbfgsB."""

  def _run_one_lbfgs(
      init_params: core.Params,
  ) -> tuple[core.Params, Any]:
    lbfgsb = jaxopt.LBFGSB(
        fun=loss_fn,
        maxls=options.num_line_search_steps,
        tol=options.tol,
        maxiter=options.maxiter,
        has_aux=True,
    )
    return lbfgsb.run(init_params=init_params, bounds=bounds)

  # We chose map over vmap because some of the lbfgs runs may terminate early.
  # pmap is also not fit for our use case, because we typically have a single
  # processor.
  return jax.lax.map(_run_one_lbfgs, init_params_batch)


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
      init_params: core.Params,
      loss_fn: core.LossFunction[core.Params],
      rng: jax.Array,
      *,
      constraints: Optional[sp.Constraint] = None,
      best_n: Optional[int] = None,
  ) -> tuple[core.Params, chex.ArrayTree]:
    metrics = {}
    logging.info('Using JAX L-BFGS-B w/ RNG: %s', rng)
    start_time = time.time()
    bounds = _get_bounds(_unbatch_params(init_params)[0], constraints)
    lbfgsb = jaxopt.LBFGSB(
        fun=eqx.filter_jit(loss_fn),
        value_and_grad=eqx.filter_value_and_grad(loss_fn, has_aux=True),
        maxls=self._options.num_line_search_steps,
        tol=self._options.tol,
        maxiter=self._options.maxiter,
        has_aux=True,
        jit=True,
        unroll=True,
    )

    all_params, opt_states = jax.lax.map(
        lambda p: lbfgsb.run(init_params=p, bounds=bounds),
        init_params,
    )  # pylint: disable=g-long-lambda

    losses = jnp.asarray(opt_states.value)
    metrics['train_time'] = time.time() - start_time
    metrics['loss'] = losses[jnp.newaxis, :]
    return (
        core.get_best_params(losses, all_params, best_n=best_n),
        metrics,
    )

  # TODO: Remove dependencies on this property.
  @property
  def best_n(self) -> int:
    return self._options.best_n or 1

  @property
  def jittable(self) -> bool:
    return True
