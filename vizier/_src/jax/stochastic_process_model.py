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

"""Flax module for a trainable stochastic process."""

import abc
from typing import Any, Callable, Generator, Generic, Optional, Protocol, TypeVar

from absl import logging
import attr
import chex
import equinox as eqx
from flax import config as flax_config
from flax import linen as nn
from flax import struct
import jax
from jax import numpy as jnp
from jax import tree_util
from jax.typing import ArrayLike
from tensorflow_probability.substrates import jax as tfp
import tree
from vizier._src.jax import types

flax_config.update('flax_return_frozendict', False)

tfd = tfp.distributions
tfb = tfp.bijectors
tfde = tfp.experimental.distributions
tfpke = tfp.experimental.psd_kernels

_D = TypeVar('_D', bound=tfd.Distribution)


class InitFn(Protocol):
  """Protocol for Flax parameter initialization functions."""

  @abc.abstractmethod
  def __call__(self, rng: jax.Array) -> jax.Array:
    pass


@struct.dataclass
class Constraint:
  """Class specifying parameter constraints.

  `ModelParameter`s may optionally contain a `Constraint` object that specifies
  the lower/upper bounds of the parameter and a bijector that maps from the
  space of all real numbers to the interval between the lower and upper bounds.

  Attributes:
    bounds: When expressing the bounds of a single parameter (array), `bounds`
      is a tuple of arrays containing (lower, upper) bounds for the parameter.
      When expressing the bounds for a structure of parameters (such as a Flax
      parameters dict), `bounds` is a tuple of structures of the same form as
      the parameters structure, containing lower and upper bounds of
      corresponding parameters. If `bounds` is None, the parameter (or all
      parameters in the structure) are unbounded. If the tuple of bounds
      contains None, then the parameter (or corresponding parameter in the
      structure) is unbounded (from below if None is in the first element of the
      tuple, and above if None is in the second element of the tuple).
    bijector: A TFP bijector-like constraint function mapping the parameter (or
      structure of parameters) from an unconstrained space to the parameter
      domain. A value of `None` is equivalent to the identity function. A
      bijector that maps structures of parameters will typically be an instance
      of `tfb.JointMap`.
  """

  bounds: Optional[types.Bounds] = struct.field(default=None)
  bijector: Optional[tfb.Bijector] = struct.field(default=None)

  @classmethod
  def create(
      cls,
      bounds: types.Bounds,
      bijector_fn: Callable[
          [Optional[ArrayLike], Optional[ArrayLike]], tfb.Bijector
      ],
  ) -> 'Constraint':
    """Factory that builds a `Constraint` from bounds and a bijector fn.

    The constraint's bijector is created by mapping `bijector_fn` over the tree
    structure of the lower/upper bounds. The bijector will operate on the same
    nested structure as the bounds.

    Args:
      bounds: A tuple of lower and upper bounds (may be ArrayTrees matching the
        tree structure of the parameters to be constrained).
      bijector_fn: A callable that takes a lower and upper bound (Arrays) as
        args and returns a TFP bijector that maps from an unconstrained space to
        the parameter domain.

    Returns:
      constraint: A Constraint object.

    Example:

    ```python
    lower = {'a': jnp.array(0.0), 'b': jnp.array(-1.0)}
    upper = {'a': jnp.array(10.0), 'b': None}
    constraint = Constraint.create((lower, upper), bijector_fn=tfb.SoftClip)
    x = {'a': jnp.array(-0.5), 'b': jnp.array(3.0)}
    constraint.bijector(x)  # => {'a': Array(0.47), 'b': Array(3.02)}
    ```
    """
    if all(
        tree_util.treedef_is_leaf(tree_util.tree_structure(x)) for x in bounds
    ):
      bijector = bijector_fn(*bounds)
    else:
      lb, ub = bounds
      if lb is None:
        if ub is None:
          raise ValueError(
              'At least one of lower and upper bounds must be specified.'
          )
        f = lambda u: bijector_fn(None, u)
        args = (ub,)
      elif ub is None:
        f = lambda l: bijector_fn(l, None)
        args = (lb,)
      else:
        f = bijector_fn
        args = bounds
      # Use dm-tree instead of jax.tree_util to handle structures containing
      # None.
      bijector = tfb.JointMap(tree.map_structure(f, *args))
    return Constraint(bounds=bounds, bijector=bijector)


@attr.frozen
class ModelParameter:
  """Class specifying a surrogate model parameter.

  Attributes:
    name: Also used as the Flax parameter name.
    init_fn: Initializes parameter values.
    constraint: Parameter constraint.
    regularizer: Regularizes the parameter.
  """

  name: str = attr.field()
  init_fn: InitFn = attr.field()
  constraint: Optional[Constraint] = attr.field(default=None)
  regularizer: Callable[[ArrayLike], jax.Array] = attr.field(
      kw_only=True, default=lambda x: jnp.zeros([], dtype=x.dtype)
  )

  @classmethod
  def from_prior(cls,
                 prior: tfd.Distribution,
                 constraint: Optional[Constraint] = None) -> 'ModelParameter':
    """Builds a `ModelParameter` from a `tfd.Distribution`.

    If `constraint` or `constraint.bijector` is None, then the constraint
    bijector is assumed to be the prior distribution's default event space
    bijector. See
    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution#experimental_default_event_space_bijector

    Args:
      prior: Parameter prior distribution.
      constraint: The parameter constraint.

    Returns:
      model_parameter: The parameter specification with the given prior.
    """
    if constraint is None or constraint.bounds is None:
      bounds = None
    else:
      bounds = constraint.bounds
    if constraint is None or constraint.bijector is None:
      bijector = prior.experimental_default_event_space_bijector()
    else:
      bijector = constraint.bijector

    sample = lambda seed: prior.sample(seed=seed)
    init_fn = sample
    if bounds is not None:
      init_fn = lambda s: jnp.clip(sample(s), *bounds)
    return ModelParameter(
        init_fn=init_fn,
        name=prior.name,
        constraint=Constraint(bounds=bounds, bijector=bijector),
        # TODO: `jnp.copy` is used to bypass TFP bijector caching;
        # otherwise JAX tracers are leaked from JIT-ted code.
        regularizer=lambda x: -prior.log_prob(jnp.copy(x)),
    )


ModelParameterGenerator = Generator[ModelParameter, jax.Array, _D]


class ModelCoroutine(Protocol, Generic[_D]):
  """`Protocol` to avoid inheritance.

  The coroutine pattern allows the `ModelParameter` objects, and the assembly of
  parameters into the kernel and stochastic process, to be specified
  simultaneously. The `StochasticProcessModel` Flax module runs the coroutine
  to initialize Flax parameters and build stochastic process objects.

  When a `ModelCoroutine` is called, it returns a generator-iterator, which
  should be iterated to build the `ModelParameter`s and the stochastic process
  object. See the full protocol below.
  """

  def __call__(
      self,
      inputs: Optional[types.ModelInput] = None,
      **kwargs,
  ) -> ModelParameterGenerator[_D]:
    """Coroutine function to be called from `StochasticProcessModel`.

    The coroutine is implemented via an enhanced generator
    (https://peps.python.org/pep-0342/). The generator-iterator returned by this
    method corresponds to the pytype
    `Generator[YieldType, SendType, ReturnType]`. (Python also has a newer, more
    flexible `Coroutine` type declared with `async`/`await` syntax. Here, when
    we reference "coroutines," we're referring to the simpler, more restrictive
    generator-based implementation.)

    The expected protocol is to run the coroutine for two different use cases:

    1) To build the Flax model.
    2) To implement Flax model forward passes.

    During (1), a new Flax model parameter is declared with the `name` and
    `init_fn` of each `ModelParameter` yielded by the generator. The initial
    values of each Flax parameter are generated by the `init_fn` and then sent
    into the generator as the left-hand sides of the yield statements. Once all
    `ModelParameter`s are yielded, the generator raises a `StopIteration`, and
    `StopIteration.value` contains a `tfd.Distribution` representing a
    stochastic process (e.g. `tfd.GaussianProcess` or `tfd.StudentTProcess`).
    During Flax module initialization, the returned `tfd.Distribution` is
    ignored.

    During (2), for each `ModelParameter` yielded by the generator, the Flax
    module accesses the Flax parameter of the same name, regularizes it (if
    applicable), sends the value into the generator, and stores the value of the
    regularization loss in a Flax mutable variable collection. Once all
    `ModelParameter`s are yielded, the generator raises a `StopIteration`, and
    `StopIteration.value` contains a `tfd.Distribution` on the provided index
    points. The module's `__call__` method returns this distribution.

    Example:

    ```python
    # Define a coroutine for a simple Gaussian Process model with trainable
    # kernel amplitude and observation noise variance.
    def model_coroutine(inputs=None):
      amplitude_constraint = Constraint(
          bounds=(jnp.zeros([]), None), bijector=tfb.Exp())
      amplitude = yield ModelParameter(
          init_fn=jax.random.exponential,
          regularizer=lambda x: 1e-3 * x**2,
          constraint=amplitude_constraint,
          name='amplitude')
      kernel = tfpk.ExponentiatedQuadratic(amplitude=amplitude)
      observation_noise = yield ModelParameter.from_prior(
          tfd.LogNormal(0.0, 1.0, name='observation_noise'),
          constraint=Constraint(bounds=(jnp.zeros([]), None)))
      return tfd.GaussianProcess(kernel=kernel, index_points=inputs,
          observation_noise_variance=observation_noise)
    ```

    Args:
      inputs: An ArrayTree of index points or None.
      **kwargs:
    """
    pass


def _squeeze_to_event_dims(
    dist: tfd.Distribution, labels: jax.Array
) -> jax.Array:
  """Squeezes the singleton `metrics` dimension from `labels`, if applicable."""
  if len(dist.event_shape) == 1 and labels.shape == (dist.event_shape[0], 1):
    return jnp.squeeze(labels, axis=-1)
  return labels


class StochasticProcessModel(nn.Module):
  """Builds a Stochastic Process Flax module.

  The module is instantiated with a coroutine in the pattern of
  `ModelCoroutine` and represents a trainable stochastic process
  (typically a `tfd.GaussianProcess` or `tfd.StudentTProcess`.)

  The module may also be passed a `mean_fn`, which is evaluated at the input
  points and returns the mean of the stochastic process (default is a constant
  zero mean).

  Examples:

  ```python
  from jax import random

  # Simulate some observed data.
  dim = 3
  x_observed = random.uniform(random.PRNGKey(0), shape=(20, dim))
  y_observed = x_observed.sum(axis=-1)

  # Build a GP module. `coro` follows the `ModelCoroutine` protocol.
  coro = GaussianProcessARD(dimension=dim)
  gp_model = StochasticProcessModel(coroutine=coro)

  # Initialize the Flax parameters.
  init_params = gp_model.init(random.PRNGKey(1), x_observed)

  # Build a GP with `x_observed` as index points. By default, `apply` invokes
  # the Flax module's `__call__` method.
  gp, regularization_losses = gp_model.apply(
      init_params,
      x_observed,
      mutable=('losses',))

  # Run the expensive computation (often a Cholesky decomposition) necessary to
  # compute the GP posterior predictive, and return the expensive intermediates
  # as mutable state.
  _, pp_state = gp_model.apply(
      {'params': init_state['params']},
      x_observed,
      y_observed,
      method=gp_model.precompute_predictive,
      mutable=('predictive'))

  # Now, posterior predictive GPs over different sets of index points,
  # conditioned on the observed data `x_observed` and `y_observed`, can be built
  # without recomputing the Cholesky decomposition.
  x_predicted = random.uniform(random.PRNGKey(2), shape=(5, dim))
  pp_dist = gp_model.apply(
      {'params': init_state['params'], **pp_state},
      x_predicted,
      x_observed,
      y_observed,
      method=gp_model.posterior_predictive)
  ```
  """

  coroutine: ModelCoroutine

  # `None` is zero-mean.
  mean_fn: Optional[Callable[[types.ModelInput], jax.Array]] = None

  def setup(self):
    """Builds module parameters."""
    generator = self.coroutine()
    try:
      p: ModelParameter = next(generator)
      while True:
        # Declare a Flax variable with the name and initialization function from
        # the `ModelParameter`.
        param: jax.Array = self.param(p.name, p.init_fn)
        p: ModelParameter = generator.send(param)
    except StopIteration:
      # Ignore the return value from the generator since this method only builds
      # the Flax parameters.
      pass

  def __call__(self, x: types.ModelInput, **kwargs) -> _D:
    """Returns a stochastic process distribution.

    If the Flax module's `apply` method is called with `mutable=True` or
    `mutable=('losses,')` regularization losses are additionally returned.

    Args:
      x: ArrayTree of index points in the constrained space.
      **kwargs:

    Returns:
      dist: `tfd.Distribution` instance with x as index points.
    """
    gen = self.coroutine(inputs=x, **kwargs)
    if self.is_initializing() and isinstance(self.mean_fn, nn.Module):
      # If mean_fn is a module, call it so its parameters are initialized.
      _ = self.mean_fn(x)  # pylint: disable=not-callable
    try:
      p: ModelParameter = next(gen)
      while True:
        # "params" is the name that `nn.Module` gives to the collection of read-
        # only variables.
        param: jax.Array = self.get_variable('params', p.name)
        if p.regularizer:
          self.sow(  # `sow` stores a value in a collection.
              'losses',
              f'{p.name}_regularization',
              p.regularizer(param),
              reduce_fn=lambda _, b: b,
          )
        p = gen.send(param)
    except StopIteration as e:
      # After the generator is exhausted, it raises a `StopIteration` error. The
      # `StopIteration` object has a property `value` of type `_D`.
      gp = e.value
      if self.mean_fn is None:
        tfp_mean_fn = None
      else:
        # TODO: Decide on a better pattern for incorporating
        # mean_fn.
        # pylint: disable=protected-access
        def tfp_mean_fn(x_: tfpke.ContinuousAndCategoricalValues):
          return self.mean_fn(
              types.ModelInput(
                  types.PaddedArray(
                      x_.continuous,
                      fill_value=x.continuous.fill_value,
                      _original_shape=(
                          x_.continuous.shape[0],
                          x.continuous._original_shape[1],
                      ),
                      _mask=x.continuous._mask,
                      _nopadding_done=x.continuous._nopadding_done,
                  ),
                  types.PaddedArray(
                      x_.categorical,
                      fill_value=x.categorical.fill_value,
                      _original_shape=(
                          x_.categorical.shape[0],
                          x.categorical._original_shape[1],
                      ),
                      _mask=x.categorical._mask,
                      _nopadding_done=x.categorical._nopadding_done,
                  ),
              )
          )
          # pylint: enable=protected-access

      return gp.copy(mean_fn=tfp_mean_fn)

  def precompute_predictive(
      self,
      x_observed: types.ModelInput,  # Could just take ModelData
      y_observed: types.PaddedArray,
  ) -> None:
    """Builds a stochastic process regression model conditioned on observations.

    The mutable variable returned by this method as auxiliary output should be
    passed as state to `posterior_predictive`. This avoids repeated, expensive
    operations (often Cholesky decompositions) when computing the posterior
    predictive.

    Args:
      x_observed: Index points on which to condition the posterior predictive.
      y_observed: Observations on which to condition the posterior predictive.
    """
    # Call the `tfd.Distribution` object's `posterior_predictive` method. This
    # triggers an expensive computation, typically a Cholesky decomposition, and
    # returns a new `tfd.Distribution` representing the posterior predictive.
    # Expensive intermediates are stored in the `precomputed_cholesky` Flax
    # variable and returned as auxiliary output.
    kwargs = {}
    if not y_observed._nopadding_done:  # pylint: disable=protected-access
      kwargs['observations_is_missing'] = y_observed.is_missing[0]

    prior = self(x_observed)
    observations = _squeeze_to_event_dims(prior, y_observed.padded_array)
    predictive_dist = prior.posterior_predictive(
        predictive_index_points=None, observations=observations, **kwargs
    )
    # pylint: disable=protected-access
    cached_predictive_intermediates = {
        '_precomputed_divisor_matrix_cholesky': (
            predictive_dist._precomputed_divisor_matrix_cholesky
        ),
        '_precomputed_solve_on_observation': (
            predictive_dist._precomputed_solve_on_observation
        ),
    }
    # pylint: enable=protected-access
    self.sow(
        'predictive',
        'precomputed_cholesky',
        cached_predictive_intermediates,
        reduce_fn=lambda _, b: b,
    )

  def posterior_predictive(
      self,
      x_predictive: types.ModelInput,
      x_observed: types.ModelInput,
      y_observed: types.PaddedArray,
  ) -> _D:
    """Returns a posterior predictive stochastic process.

    The posterior predictive distribution over the function values at
    `x_predictive`, typically a `tfd.GaussianProcessRegressionModel` or
    `tfd.StudentTProcessRegressionModel`, is built using the mutable variable in
    `predictive/precomputed_cholesky`. This avoids repeated, expensive
    computation (often Cholesky decompositions of the kernel matrix for observed
    data). See the class docstring for how to use `precompute_predictive` in
    combination with `posterior_predictive`.

    Args:
      x_predictive: Predictive index points.
      x_observed: Index points on which to condition the posterior predictive.
      y_observed: Observations on which to condition the posterior predictive.

    Returns:
      pp_dist: The posterior predictive distribution over `x_predictive`.
    """
    if not self.has_variable('predictive', 'precomputed_cholesky'):
      raise ValueError(
          'The mutable variable returned by '
          '`precompute_predictive` must be passed into `predict`. '
          'See the class docstring for an example.'
      )
    # Access the precomputed values stored in the Flax variable, and build the
    # distribution object over the predictive index points (avoiding
    # recomputation).
    cached_intermediates = self.get_variable(
        'predictive', 'precomputed_cholesky'
    )
    kwargs = cached_intermediates
    if not y_observed._nopadding_done:  # pylint: disable=protected-access
      kwargs = kwargs | {'observations_is_missing': y_observed.is_missing[0]}

    predictive_index_points = tfpke.ContinuousAndCategoricalValues(
        continuous=x_predictive.continuous.padded_array,
        categorical=x_predictive.categorical.padded_array,
    )

    prior = self(x_observed)
    observations = _squeeze_to_event_dims(prior, y_observed.padded_array)
    return prior.posterior_predictive(
        observations=observations,
        predictive_index_points=predictive_index_points,
        **kwargs,
    )


class VectorToArrayTree(tfb.Chain):
  """Bijector that converts a vector to a dict like `params`.

  The bijector splits the vector, reshapes the splits, and then packs the splits
  into a dictionary. The bijector's `inverse` method does the reverse.

  It also has aliases `to_params` and `to_vectors` for `forward()` and
  `inverse()` methods.

  Example:

  ```python
  params = {'a': [4.0, 3.0], 'b': -2.0, 'c': [[6.0]]}
  bij = VectorToArrayTree(params)
  bij.inverse(params)  #  => [4.0, 3.0, -2.0, 6.0]
  bij([0.0, 1.0, 2.0, 3.0])  # => {'a': [0.0, 1.0], 'b': 2.0, 'c': [[3.0]]}
  ```
  """

  def __init__(
      self, arraytree: types.ParameterDict, *, validate_args: bool = False
  ):
    """Init.

    Args:
      arraytree: A nested structure of Arrays.
      validate_args: If True, does runtime validation. It may be slow.
    """
    parameters = dict(locals())
    flat, treedef = tree_util.tree_flatten(arraytree)
    restructure = tfb.Restructure(
        output_structure=tree_util.tree_unflatten(treedef, range(len(flat))))
    reshape = tfb.JointMap(
        [tfb.Reshape(event_shape_out=f.shape) for f in flat],
        validate_args=validate_args,
    )
    split = tfb.Split([x.size for x in flat], validate_args=validate_args)
    super().__init__(
        [restructure, reshape, split],
        parameters=parameters,
        name='VectorToArrayTree',
        validate_args=validate_args,
    )

  def to_params(self, vector: ArrayLike) -> types.ParameterDict:
    return self.forward(vector)

  def to_vector(self, params: types.ParameterDict) -> jax.Array:
    return self.inverse(params)


def get_constraints(
    model: StochasticProcessModel, x: Optional[Any] = None
) -> Constraint:
  """Gets the parameter constraints from a StochasticProcessModel.

  If the model contains trainable Flax variables besides those defined by the
  coroutine (for example, if `mean_fn` is a Flax module), the non-coroutine
  variables are assumed to be unconstrained (the bijector passes them through
  unmodified, and their lower/upper bounds are `None`). In this case `x` must be
  passed, so that the structure of the non-coroutine parameters dict(s) can be
  generated. `Constraint` objects for models with constrained parameters aside
  from those defined in the coroutine must be built manually.

  This method runs the coroutine, collects the parameter constraints, and
  returns a new Constraint object in which the lower/upper bounds are dicts
  (corresponding to the parameters dict) and the bijector operates on dicts. The
  object may be passed to a Vizier Optimizer to constrain the parameters for
  optimization.

  Example:

  ```python
  def model_coroutine(inputs=None):
    amplitude_constraint = Constraint(
        bounds=(jnp.array(0.1), None))
    length_scale_constraint = Constraint.create(
        bounds=(jnp.array(0.0), jnp.array(10.0)),
        bijector_fn=tfb.Sigmoid)

    amplitude = yield ModelParameter.from_prior(
        tfd.LogNormal(0.0, 1.0, name='amplitude'),
        constraint=amplitude_constraint)
    length_scale = yield ModelParameter(
        init_fn=jax.random.exponential,
        regularizer=lambda x: 1e-3 * x**2,
        constraint=length_scale_constraint,
        name='length_scale')
    kernel = tfpk.ExponentiatedQuadratic(
        amplitude=amplitude, length_scale=length_scale)
    return tfd.GaussianProcess(kernel, index_points=inputs)

  model = StochasticProcessModel(model_coroutine)
  constraint = GetConstraints(model)
  constraint.bijector
  # => tfb.JointMap({'amplitude': tfb.Exp(),
                    'length_scale': tfb.Sigmoid(0.0, 10.0)})

  constraint.bounds
  # => ({'amplitude': jnp.array(0.1), 'length_scale': jnp.array(0.0)},
  #     {'amplitude': None, 'length_scale': jnp.array(10.0)})
  ```

  Args:
    model: A `StochasticProcessModel` instance.
    x: An input that can be passed to `model.lazy_init`. `x` must be of the same
      structure as the model inputs and may contain arrays or `ShapeDtypeStruct`
      instances (see flax.linen.Module.lazy_init docs). If `model` contains Flax
      variables aside from those defined by `model.coroutine` (e.g. in a
      trainable `mean_fn`) then this arg is required.

  Returns:
    constraint: A `Constraint` instance expressing constraints on the parameters
      specified by `coroutine`.
  """

  # Run the coroutine to extract constraints for the model parameters defined in
  # the coroutine.
  gen = model.coroutine()
  k = jax.random.PRNGKey(0)
  lower = {}
  upper = {}
  bijectors = {}
  try:
    p = next(gen)
    while True:
      v = p.init_fn(k)
      if p.constraint is None or p.constraint.bounds is None:
        lower[p.name] = None
        upper[p.name] = None
      else:
        lower[p.name] = p.constraint.bounds[0]
        upper[p.name] = p.constraint.bounds[1]
      if p.constraint is None or p.constraint.bijector is None:
        bijectors[p.name] = tfb.Identity()
      else:
        bijectors[p.name] = p.constraint.bijector
      p = gen.send(v)
  except StopIteration:
    pass

  # `tfb.JointMap` applies a structure of bijectors to a parallel structure of
  # inputs. Define a `JointMap` bijector that maps an unconstrained parameters
  # dict to a constrained parameters dict with a dict of bijectors (all dicts
  # are keyed by parameter names).
  bijector = tfb.JointMap(bijectors=bijectors)
  if x is not None:
    # Get the parameters dict keys, if any, that do not come from the coroutine
    # (e.g. `mean_fn` parameters).
    params = model.lazy_init(jax.random.PRNGKey(0), x)['params']
    non_coroutine_keys = set(params.keys()) - set(bijectors.keys())

    # Define a bijector that ignores (applies an identity transformation to)
    # non-coroutine parameters.
    if non_coroutine_keys:
      logging.info(
          (
              'Defining a constraint object that ignores the following'
              'non-coroutine parameters: %s'
          ),
          non_coroutine_keys,
      )

      def _wrap_bijector_method_to_ignore_non_coro(f):
        """Wrap bijector methods to pass non-coroutine params through."""

        def _f(p):
          p_ = p.copy()
          non_coroutine_params = {k: p_.pop(k) for k in non_coroutine_keys}
          y = f(p_)
          y.update(non_coroutine_params)
          return y

        return _f

      def _bijector_fldj_with_non_coro(p):
        """Non-coroutine params do not affect the FLDJ."""
        p_ = {k: v for k, v in p.items() if k not in non_coroutine_keys}
        return bijector.forward_log_det_jacobian(p_)

      bijector_forward_min_event_ndims = bijector.forward_min_event_ndims.copy()
      # Populate `lower` and `upper` bounds dicts with `None` values for entries
      # corresponding to non-coroutine params.
      for k in non_coroutine_keys:
        lower[k] = tree.map_structure(lambda _: None, params[k])
        upper[k] = tree.map_structure(lambda _: None, params[k])
        bijector_forward_min_event_ndims[k] = tree.map_structure(
            lambda _: 0, params[k]
        )
      bijector = tfb.Inline(
          forward_fn=_wrap_bijector_method_to_ignore_non_coro(bijector.forward),
          inverse_fn=_wrap_bijector_method_to_ignore_non_coro(bijector.inverse),
          forward_log_det_jacobian_fn=_bijector_fldj_with_non_coro,
          forward_min_event_ndims=bijector_forward_min_event_ndims,
      )
    else:
      # If the model doesn't have params aside from those defined by the
      # coroutine, its params should have the same structure as `bijectors`
      # (this assertion failing indicates a bug).
      try:
        tree.assert_same_structure(params, bijectors)
      except ValueError as exc:
        raise ValueError(
            '`params` and `bijectors` should have the same nested structure.'
            f'Saw: `params={params}` and `bijectors={bijectors}`'
        ) from exc

  return Constraint((lower, upper), bijector=bijector)


class PrecomputedPredictive(eqx.Module):
  """Precomputed model for prediction.

  Conceptually, this module corresponds to the posterior distribution obtained
  by updating `prior` with `observed_data`.

  Also see `StochasticProcessWithCoroutine.precompute_predictive`.

  Attributes:
    prior: Defines the prior distribution.
    observed_data: Observations with batch shape [B].
  """

  prior: 'StochasticProcessWithCoroutine'
  observed_data: types.ModelData
  precomputed_divisor_matrix_cholesky: jax.Array
  precomputed_solve_on_observation: jax.Array

  @property
  def _posterior_kwargs(self):
    if self.observed_data.labels.shape[-1] > 1:
      # If there is more than one metric, assume we are using a MultiTask GP, in
      # which case `observations_is_missing` should be a 2D array instead of a
      # vector.
      # TODO: Make `_mask` public on PaddedArray.
      observations_is_missing = ~self.observed_data.labels._mask  # pylint: disable=protected-access
    else:
      # If there is one metric, then squeeze out the `num_metrics` dimension
      # (which has size 1).
      observations_is_missing = self.observed_data.labels.is_missing[0]
    return dict(
        _precomputed_divisor_matrix_cholesky=self.precomputed_divisor_matrix_cholesky,
        _precomputed_solve_on_observation=self.precomputed_solve_on_observation,
        observations=self.observed_data.labels.padded_array,
        observations_is_missing=observations_is_missing,
    )

  def predict(self, x_predictive: types.ModelInput) -> tfd.Distribution:
    """Returns the posterior distribution on index points `xs`.

    Args:
      x_predictive: Array(Tree) of batch shape [B].

    Returns:
      Distribution with sample shape [B].
    """
    return self._predict(x_predictive)

  def _predict(
      self,
      x_predictive: types.ModelInput,
      expand_batch_dim: bool = False,
  ) -> tfd.Distribution:
    """Returns the posterior distribution on index points `xs`.

    Args:
      x_predictive: Array(Tree) of batch shape [B].
      expand_batch_dim: If True, expand the dimensions of `x_predictive` so that
        it broadcasts with the distribution's batch shape. This is useful for
        predictions on a batch of predictive points using an ensemble GP.

    Returns:
      Distribution with sample shape [B].
    """
    x_pred_tfp = tfpke.ContinuousAndCategoricalValues(
        continuous=x_predictive.continuous.padded_array,
        categorical=x_predictive.categorical.padded_array,
    )
    if expand_batch_dim:
      x_pred_tfp = tfpke.ContinuousAndCategoricalValues(
          continuous=x_pred_tfp.continuous[:, jnp.newaxis, ...],
          categorical=x_pred_tfp.categorical[:, jnp.newaxis, ...],
      )
    prior_gp = self.prior(self.observed_data.features)
    kwargs = self._posterior_kwargs.copy()
    kwargs['observations'] = _squeeze_to_event_dims(
        prior_gp, kwargs['observations']
    )
    return prior_gp.posterior_predictive(
        predictive_index_points=x_pred_tfp, **kwargs
    )


class UniformEnsemblePredictive(eqx.Module):
  """Uniform ensemble of predictive models.

  Ensembles the `predictives` with equal weights.
  """

  predictives: PrecomputedPredictive

  def predict(self, xs: types.ModelInput) -> tfd.Distribution:
    return self.predict_with_aux(xs)[0]

  def predict_with_aux(
      self, xs: types.ModelInput
  ) -> tuple[tfd.Distribution, chex.ArrayTree]:
    # If `xs` has a batch dimension and the predictive distribution is
    # ensembled, we expand xs' batch shape so as not to collide with the
    # ensemble dimensions. (`vmap` cannot currently be used on functions
    # returning TFP distributions).
    has_batched_hparams = (
        self.predictives.precomputed_solve_on_observation.ndim > 1
    )
    expand_x = len(xs.continuous.shape) == 3 and has_batched_hparams
    dist = self.predictives._predict(xs, expand_batch_dim=expand_x)  # pylint: disable=protected-access
    if has_batched_hparams:
      return (
          tfd.MixtureSameFamily(
              tfd.Categorical(logits=jnp.zeros(dist.batch_shape[-1])), dist
          ),
          {
              'components_mean': dist.mean().T,
              'components_stddev': dist.stddev().T,
          },
      )
    return dist, {}


def _initialize_params(
    coroutine: ModelCoroutine, rng: jax.Array
) -> chex.ArrayTree:
  """Randomly initializes a coroutine's parameters."""
  gen = coroutine()
  params = {}
  try:
    p: ModelParameter = next(gen)
    while True:
      # Declare a Flax variable with the name and initialization function from
      # the `ModelParameter`.
      rng, init_rng = jax.random.split(rng)
      param = p.init_fn(init_rng)
      params[p.name] = param
      p: ModelParameter = gen.send(param)
  except StopIteration:
    return params


class StochasticProcessWithCoroutine(eqx.Module):
  """Fully parameterized stochastic process model."""

  coroutine: ModelCoroutine = eqx.field(static=True)
  params: chex.ArrayTree

  @classmethod
  def initialize(cls, coroutine: ModelCoroutine, *, rng: jax.Array):
    """Builds module parameters."""
    # return cls(coroutine, params=_initialize_params(coroutine, rng))
    # TODO: Put back the line above. This code currently uses
    # flax's init routine to exactly match the setup so we can replicate
    # the trajectories from pre-refactoring.
    return cls(
        coroutine, StochasticProcessModel(coroutine).init(rng, None)['params']
    )

  def call_with_aux(
      self, x: types.ModelInput, /
  ) -> tuple[tfd.Distribution, chex.ArrayTree]:
    """Returns the joint distribution and auxiliary information.

    Args:
      x:

    Returns:
      (dist, aux)
      dist: Distribution
      aux: A dict where all the leaf nodes in the subtree aux[`losses`] contain
        regularization losses.
    """
    gen = self.coroutine(x)
    params = self.params
    params_loss = dict()
    try:
      p: ModelParameter = next(gen)
      while True:
        if p.regularizer:
          params_loss[p.name] = p.regularizer(params[p.name])
        # "params" is the name that `nn.Module` gives to the collection of read-
        # only variables.
        p = gen.send(params[p.name])
    except StopIteration as e:
      # After the generator is exhausted, it raises a `StopIteration` error. The
      # `StopIteration` object has a property `value` of type `_D`.
      return e.value, {'losses': params_loss}

  def __call__(self, x: types.ModelInput) -> tfd.Distribution:
    return self.call_with_aux(x)[0]

  def loss_with_aux(
      self,
      data: types.ModelData,
      seed: Optional[jax.Array] = None,
  ) -> tuple[jax.Array, chex.ArrayTree]:
    dist, aux = self.call_with_aux(data.features)

    # If `seed` is provided, that implies `dist.log_prob` is a stochastic
    # approximation that requires a random seed.
    log_prob_kwargs = {}
    if seed is not None:
      log_prob_kwargs['key'] = seed

    labels = _squeeze_to_event_dims(dist, data.labels.padded_array)
    # TODO: Enable `is_missing` for MTGP.
    if isinstance(dist, tfde.MultiTaskGaussianProcess):
      logging.warning(
          'Using a multitask GP; note that padding/masking is not yet supported'
          'in `log_prob`.'
      )
      nll_data = -dist.log_prob(labels, **log_prob_kwargs)
    else:
      nll_data = -dist.log_prob(
          labels, is_missing=data.labels.is_missing[0], **log_prob_kwargs
      )
    loss = nll_data + jax.tree_util.tree_reduce(jnp.add, aux['losses'])
    return loss, aux

  def precompute_predictive(
      self, data: types.ModelData
  ) -> PrecomputedPredictive:
    jax.monitoring.record_event(
        '/vizier/jax/coroutine_with_params/precompute_predictive/traced'
    )
    if jnp.size(data.labels.padded_array) == 0:
      # TFP `retrying_cholesky` does not handle empty observations.
      prior = self(data.features).copy(cholesky_fn=None)
    else:
      prior = self(data.features)

    observations = _squeeze_to_event_dims(prior, data.labels.padded_array)
    if isinstance(prior, tfde.MultiTaskGaussianProcess):
      # TODO: Make `_mask` public on PaddedArray.
      observations_is_missing = ~data.labels._mask  # pylint: disable=protected-access
    else:
      observations_is_missing = data.labels.is_missing[0]
    predictive = prior.posterior_predictive(
        predictive_index_points=None,
        observations=observations,
        observations_is_missing=observations_is_missing,
    )
    # pylint: disable=protected-access
    return PrecomputedPredictive(
        self,
        data,
        predictive._precomputed_divisor_matrix_cholesky,
        predictive._precomputed_solve_on_observation,
    )


class CoroutineWithData(eqx.Module):
  """Utility module for training (ARD).

  Has setup() and loss_with_aux() that can be readily `eqx.filter_jit`ed
  """

  coroutine: ModelCoroutine = eqx.field(static=True)
  data: types.ModelData

  def constraints(self):
    return get_constraints(StochasticProcessModel(self.coroutine))

  def setup(self, rng: jax.Array):
    jax.monitoring.record_event('/vizier/jax/coroutine_with_data/setup/traced')
    return StochasticProcessWithCoroutine.initialize(
        self.coroutine, rng=rng
    ).params

  def loss_with_aux(
      self, params, seed: Optional[jax.Array] = None
  ) -> tuple[jax.Array, chex.ArrayTree]:
    jax.monitoring.record_event(
        '/vizier/jax/coroutine_with_data/loss_with_aux/traced'
    )
    return StochasticProcessWithCoroutine(self.coroutine, params).loss_with_aux(
        self.data, seed=seed
    )
