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

"""Flax module for a trainable stochastic process."""

import abc
from typing import Any, Callable, Generator, Generic, Iterable, Mapping, Optional, Protocol, TypeVar, Union

import attr
import chex
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import tree_util
from tensorflow_probability.substrates import jax as tfp
import tree

Array = chex.Array
ArrayTree = chex.ArrayTree
PRNGKey = chex.PRNGKey

# An ArrayTree that allows None values.
ArrayTreeOptional = Union[
    Array,
    Iterable[Optional['ArrayTreeOptional']],
    Mapping[Any, Optional['ArrayTreeOptional']],
]

tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

_D = TypeVar('_D', bound=tfd.Distribution)
_In = TypeVar('_In', bound=ArrayTree)


class InitFn(Protocol):
  """Protocol for Flax parameter initialization functions."""

  @abc.abstractmethod
  def __call__(self, rng: PRNGKey) -> Array:
    pass


@attr.frozen
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

  bounds: Optional[tuple[Optional[ArrayTreeOptional],
                         Optional[ArrayTreeOptional]]] = None
  bijector: Optional[tfb.Bijector] = None

  @classmethod
  def create(
      cls,
      bounds: tuple[Optional[ArrayTreeOptional], Optional[ArrayTreeOptional]],
      bijector_fn: Callable[[Optional[Array], Optional[Array]], tfb.Bijector],
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
  regularizer: Callable[[Array], Array] = attr.field(
      kw_only=True, default=lambda x: jnp.zeros([], dtype=x.dtype))

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
        regularizer=lambda x: -prior.log_prob(x),
    )


ModelParameterGenerator = Generator[ModelParameter, Array, _D]


class ModelCoroutine(Protocol, Generic[_In, _D]):
  """`Protocol` to avoid inheritance.

  The coroutine pattern allows the `ModelParameter` objects, and the assembly of
  parameters into the kernel and stochastic process, to be specified
  simultaneously. The `StochasticProcessModel` Flax module runs the coroutine
  to initialize Flax parameters and build stochastic process objects.

  When a `ModelCoroutine` is called, it returns a generator-iterator, which
  should be iterated to build the `ModelParameter`s and the stochastic process
  object. See the full protocol below.
  """

  def __call__(self,
               inputs: Optional[_In] = None) -> ModelParameterGenerator[_D]:
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
    """
    pass


class StochasticProcessModel(nn.Module, Generic[_In]):
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
  # compute the GP posterior predictive, and return the predictive distribution
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
      method=gp_model.predict)
  ```
  """

  coroutine: ModelCoroutine
  mean_fn: Callable[[_In], Array] = lambda _: 0.0

  def setup(self):
    """Builds module parameters."""
    generator = self.coroutine()
    try:
      p: ModelParameter = next(generator)
      while True:
        # Declare a Flax variable with the name and initialization function from
        # the `ModelParameter`.
        param: Array = self.param(p.name, p.init_fn)
        p: ModelParameter = generator.send(param)
    except StopIteration:
      # Ignore the return value from the generator since this method only builds
      # the Flax parameters.
      pass

  def __call__(self, x: _In) -> _D:
    """Returns a stochastic process distribution.

    If the Flax module's `apply` method is called with `mutable=True` or
    `mutable=('losses,')` regularization losses are additionally returned.

    Args:
      x: ArrayTree of index points in the constrained space.

    Returns:
      dist: `tfd.Distribution` instance with x as index points.
    """
    gen = self.coroutine(inputs=x)
    if self.is_initializing() and isinstance(self.mean_fn, nn.Module):
      _ = self.mean_fn(x)  # Call mean_fn so its parameters are initialized.
    try:
      p: ModelParameter = next(gen)
      while True:
        # "params" is the name that `nn.Module` gives to the collection of read-
        # only variables.
        param: Array = self.get_variable('params', p.name)
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
      return gp.copy(mean_fn=self.mean_fn)

  def precompute_predictive(self, x_observed: _In, y_observed: Array) -> None:
    """Builds a stochastic process regression model conditioned on observations.

    The mutable variable returned by this method as auxillary output should be
    passed as state to `predict`. This avoids repeated, expensive operations
    (often Cholesky decompositions) when computing the posterior predictive.

    Args:
      x_observed: Index points on which to condition the posterior predictive.
      y_observed: Observations on which to condition the posterior predictive.
    """
    # Call the `tfd.Distribution` object's `predict` method. This
    # triggers an expensive computation, typically a Cholesky decomposition, and
    # returns a new `tfd.Distribution` representing the posterior predictive.
    # This distribution is stored in the `predictive_distribution` Flax variable
    # and returned as auxiliary output.
    predictive_dist = self(x_observed).posterior_predictive(
        index_points=None, observations=y_observed)
    self.sow(
        'predictive', 'distribution', predictive_dist, reduce_fn=lambda _, b: b
    )

  def posterior(self, x_predictive: _In, x_observed: _In,
                y_observed: Array) -> _D:
    """Returns the posterior predictive distribution.

    Recommended usage:
      Jit it as a function that takes only one argument. As long as x_observed
      and y_observed are the same, precompute_predicitve runs only once.

      ```python
      @jax.jit
      def posterior(x):
        return model.apply(params, x, x_observed, y_observed,
                           mutable=('predictive',), method=model.posterior)
      ```

    Args:
      x_predictive: predictive index points.
      x_observed: observed index points.
      y_observed: observed labels.

    Returns:
      Predictive distribution on x_predictive.
    """
    with jax.ensure_compile_time_eval():
      self.precompute_predictive(x_observed, y_observed)
    return self.predict(x_predictive)

  def predict(self, x_predictive: _In) -> _D:
    """Returns a posterior predictive stochastic process.

    The mutable variable in `predictive/distribution`, typically containing a
    `tfd.GaussianProcessRegressionModel` or
    `tfd.StudentTProcessRegressionModel`, is copied with the new predictive
    index points to avoid repeated, expensive computation (often Cholesky
    decompositions) in the distribution's constructor. See the class docstring
    for how to use `precompute_predictive` in combination with `predict`.

    Args:
      x_predictive: Predictive index points.

    Returns:
      pp_dist: The posterior predictive distribution over `x_predictive`.
    """
    if not self.has_variable('predictive', 'distribution'):
      raise ValueError('The mutable variable containing the initial posterior '
                       'predictive distribution must be set by '
                       '`precompute_predictive`before `predict` is called. See '
                       'the class docstring for an example.')
    # Access the `tfd.Distribution` stored in the Flax variable, and copy the
    # distribution object with new index points (avoiding recomputation).
    return self.get_variable('predictive', 'distribution').copy(
        index_points=x_predictive
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

  def __init__(self, arraytree: ArrayTree, *, validate_args: bool = False):
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

  def to_params(self, vector: Array) -> ArrayTree:
    return self.forward(vector)

  def to_vector(self, params: ArrayTree) -> Array:
    return self.inverse(params)


def get_constraints(coroutine: ModelCoroutine) -> Constraint:
  """Gets the parameter constraints from a ModelCoroutine.

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

  constraint = GetConstraints(model_coroutine)
  constraint.bijector
  # => tfb.JointMap({'amplitude': tfb.Exp(),
                    'length_scale': tfb.Sigmoid(0.0, 10.0)})

  constraint.bounds
  # => ({'amplitude': jnp.array(0.1), 'length_scale': jnp.array(0.0)},
  #     {'amplitude': None, 'length_scale': jnp.array(10.0)})
  ```

  Args:
    coroutine: A generator function that follows the `ModelCoroutine` protocol.

  Returns:
    constraint: A `Constraint` instance expressing constraints on the parameters
      specified by `coroutine`.
  """
  gen = coroutine()
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
  return Constraint((lower, upper), bijector=tfb.JointMap(bijectors=bijectors))
