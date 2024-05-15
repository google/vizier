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

"""Base class for vectorized acquisition optimizers."""

import abc
import datetime
import json
from typing import Callable, Generic, Optional, Protocol, TypeVar, Union

from absl import logging
import attr
import equinox as eqx
from flax import struct
import jax
from jax import numpy as jnp
import numpy as np
from vizier import pyvizier as vz
from vizier._src.jax import types
from vizier.pyvizier import converters
from vizier.utils import json_utils

_S = TypeVar('_S')  # A container of optimizer state that works as a Pytree.

# `VectorizedOptimizerInput` holds features for acquisition function
# optimization and is intended to be private to the `optimizers` submodule.
# Each component has shape (batch_size, n_parallel, n_padded_features)
VectorizedOptimizerInput = types.ContinuousAndCategorical[types.Array]


def optimizer_to_model_input_single_array(
    x: types.Array, n_features: jax.Array
) -> types.PaddedArray:
  mask = jnp.ones_like(x, dtype=bool)
  mask = jnp.logical_and(mask, jnp.arange(x.shape[-1]) < n_features)
  return types.PaddedArray(
      x,
      fill_value=jnp.zeros([], dtype=x.dtype),
      _original_shape=jnp.concatenate(
          [jnp.array(x.shape[:-1]), jnp.array([n_features])], axis=0
      ),
      _mask=mask,
      _nopadding_done=False,  # Fix
  )


def _optimizer_to_model_input(
    x: VectorizedOptimizerInput,
    n_features: types.ContinuousAndCategorical,
    squeeze_middle_dim: bool = False,
) -> types.ModelInput:
  if squeeze_middle_dim:
    x_cont = jnp.squeeze(x.continuous, axis=1)
    x_cat = jnp.squeeze(x.categorical, axis=1)
  else:
    x_cont = x.continuous
    x_cat = x.categorical
  return types.ModelInput(
      continuous=optimizer_to_model_input_single_array(
          x_cont, n_features.continuous
      ),
      categorical=optimizer_to_model_input_single_array(
          x_cat, n_features.categorical
      ),
  )


def _reshape_to_parallel_batches(
    x: types.PaddedArray, parallel_dim: int
) -> tuple[jax.Array, jax.Array]:
  """Docstring."""

  new_batch_dim = x.shape[0] // parallel_dim
  new_padded_array = jnp.reshape(
      x.padded_array[: new_batch_dim * parallel_dim],
      (new_batch_dim, parallel_dim, x.shape[-1]),
  )

  valid_batch_mask = (
      jnp.arange(new_batch_dim) < x._original_shape[0] // parallel_dim  # pylint: disable=protected-access
  )
  return new_padded_array, valid_batch_mask


class VectorizedStrategyResults(eqx.Module):
  """Container for a vectorized strategy result."""

  features: VectorizedOptimizerInput  # (batch_size, n_parallel, n_features)
  rewards: types.Array  # (batch_size,)
  aux: dict[str, jax.Array] = eqx.field(default_factory=dict)


class VectorizedStrategy(abc.ABC, Generic[_S]):
  """JIT-friendly optimizer that maintains an internal state of type `_S`.

  The strategy is responsible for generating suggestions that will maximize the
  reward. The order of calls is important. It's expected to be used in
  'suggest','update', 'suggest', 'update', etc.
  """

  @abc.abstractmethod
  def init_state(
      self,
      seed: jax.Array,
      n_parallel: int = 1,
      *,
      prior_features: Optional[VectorizedOptimizerInput] = None,
      prior_rewards: Optional[types.Array] = None,
  ) -> _S:
    """Initialize the state.

    Arguments:
      seed: Random seed for state initialization.
      n_parallel: Number of points that the acquisition function maps to a
        single value. This arg may be greater than 1 if a parallel acquisition
        function (qEI, qUCB) is used; otherwise it should be 1.
      prior_features: (n_prior_features, n_parallel, features_count)
      prior_rewards: (n_prior_features, )

    Returns:
      initial_state:
    """

  @abc.abstractmethod
  def suggest(
      self,
      seed: jax.Array,
      state: _S,
      n_parallel: int = 1,
  ) -> VectorizedOptimizerInput:
    """Generate new suggestions.

    Arguments:
      seed: Random seed.
      state: Optimizer state.
      n_parallel: Number of points that the acquisition function maps to a
        single value. This arg may be greater than 1 if a parallel acquisition
        function (qEI, qUCB) is used; otherwise it should be 1.

    Returns:
      suggested features: (batch_size, features_count)
    """

  @property
  @abc.abstractmethod
  def suggestion_batch_size(self) -> int:
    """The number of suggestions returned at every suggest call."""

  @abc.abstractmethod
  def update(
      self,
      seed: jax.Array,
      state: _S,
      batch_features: VectorizedOptimizerInput,
      batch_rewards: types.Array,
  ) -> _S:
    """Update the strategy state with the results of the last suggestions.

    Arguments:
      seed: Random seed.
      state: Optimizer state.
      batch_features: (batch_size, n_parallel, features_count)
      batch_rewards: (batch_size, )
    """


class VectorizedStrategyFactory(Protocol):
  """Factory class to generate vectorized strategy.

  It's used in VectorizedOptimizer to create a new strategy every 'optimize'
  call.
  """

  def __call__(
      self,
      converter: converters.TrialToModelInputConverter,
      *,
      suggestion_batch_size: int,
  ) -> VectorizedStrategy:
    """Create a new vectorized strategy.

    Arguments:
      converter: The trial to array converter.
      suggestion_batch_size: The number of trials to be evaluated at once.
    """
    ...


class ArrayScoreFunction(Protocol):
  """Protocol for scoring array of trials.

  This protocol is suitable for optimizing batch of candidates (each one with
  its own separate score).
  """

  def __call__(
      self,
      batched_array_trials: types.ModelInput,
      seed: jax.Array,
  ) -> types.Array:
    """Evaluates the array of batched trials.

    Arguments:
      batched_array_trials: Array of shape (batch_size, n_feature_dimensions).
      seed: Random seed.

    Returns:
      Array of shape (batch_size,).
    """


# TODO: Decide on consistent terminology for acquisition functions
# that evaluate sets of points.
class ParallelArrayScoreFunction(Protocol):
  """Protocol for scoring array of parallel trials.

  This protocol is suitable for optimizing in parallel multiple candidates
  (e.g. qUCB).
  """

  def __call__(
      self,
      parallel_array_trials: types.ModelInput,
      seed: jax.Array,
  ) -> types.Array:
    """Evaluates the array of batched trials.

    Arguments:
      parallel_array_trials: Array of shape (batch_size, n_parallel,
        n_feature_dimensions).
      seed: Random seed.

    Returns:
      Array of shape (batch_size).
    """


@struct.dataclass
class VectorizedOptimizer(Generic[_S]):
  """Vectorized strategy optimizer.

  The optimizer is stateless and will create a new vectorized strategy at the
  beginning of every 'optimize' call.

  The optimizer is responsible for running the iterative optimization process
  using the vectorized strategy, which consists of:
  1. Ask the strategy for suggestions.
  2. Evaluate the suggestions to get rewards.
  3. Tell the strategy about the rewards of its suggestion, so the strategy can
  update its internal state.

  The optimization process will terminate when the time limit or the total
  objective function evaluations limit has exceeded.

  Attributes:
    strategy: A factory for generating new strategy.
    n_feature_dimensions_with_padding: Number of feature dimensions including
      padding.
    n_feature_dimensions: Number of feature dimensions (less than or equal to
      `n_feature_dimensions_with_padding`).
    suggestion_batch_size: Number of suggested points returned at each call.
    max_evaluations: The maximum number of objective function evaluations.
    dtype: Dtype of input data.
    use_fori: Whether to use JAX's fori_loop in the suggest-evalute-update loop.
  """

  strategy: VectorizedStrategy[_S]
  n_feature_dimensions: types.ContinuousAndCategorical[jax.Array]
  n_feature_dimensions_with_padding: types.ContinuousAndCategorical[int] = (
      struct.field(pytree_node=False)
  )
  suggestion_batch_size: int = struct.field(pytree_node=False, default=25)
  max_evaluations: int = struct.field(pytree_node=False, default=75_000)
  dtype: types.ContinuousAndCategorical[jnp.dtype] = struct.field(
      pytree_node=False,
      default=types.ContinuousAndCategorical[jnp.dtype](  # pytype: disable=wrong-arg-types  # jnp-type
          jnp.float64, types.INT_DTYPE
      ),
  )
  use_fori: bool = struct.field(pytree_node=False, default=True)

  # TODO: Remove score_fn argument.
  # pylint: disable=g-bare-generic
  def __call__(
      self,
      score_fn: Union[ArrayScoreFunction, ParallelArrayScoreFunction],
      *,
      score_with_aux_fn: Optional[Callable] = None,
      count: int = 1,
      prior_features: Optional[types.ModelInput] = None,
      n_parallel: Optional[int] = None,
      seed: Optional[int] = None,
  ) -> VectorizedStrategyResults:
    """Optimize the objective function.

    The ask-evaluate-tell optimization procedure that runs until the allocated
    time or evaluations count exceeds.

    The number of suggestions is determined by the strategy, which is the
    `suggestion_count` property.

    The converter should be the same one used to convert trials to arrays in the
    format that the objective function expects to receive. It's used to convert
    backward the strategy's best array candidates to trial candidates.
    In addition, the converter is passed to the strategy so it could be aware
    of the type associated with each array index, and which indices are part of
    the same CATEGORICAL parameter one-hot encoding.

    The optimization stops when either of 'max_evaluations' or 'max_duration' is
    reached.

    Arguments:
      score_fn: A callback that expects 2D Array with dimensions (batch_size,
        features_count) or a 3D array with dimensions (batch_size, n_parallel,
        features_count), and a random seed, and returns a 1D Array
        (batch_size,).
      score_with_aux_fn: A callback similar to score_fn but additionally returns
        an array tree.
      count: The number of suggestions to generate.
      prior_features: Completed trials to be used for knowledge transfer. When
        the optimizer is used to optimize a designer's acquisition function, the
        prior trials are the previous designer suggestions provided in the
        ordered they were suggested.
      n_parallel: This arg should be specified if a parallel acquisition
        function (e.g. qEI, qUCB) is used, which evaluates a set of points of
        this size to a single value. Pass `None` when optimizing acquisition
        functions that evaluate a single point. (Note that `num_parallel=1` and
        `num_parallel=None` will each return a single suggestion, though they
        are different in that `num_parallel=1` indicates that the acquisition
        function consumes the two rightmost dimensions of the input while
        `num_parallel=None` indicates that only the rightmost dimension is
        consumed.)
      seed: The seed to use in the random generator.

    Returns:
      An array containing the best trials found in the optimization of shape
      (count, n_parallel or 1, n_feature_dimensions).
    """
    jax.monitoring.record_event('/vizier/jax/vectorized_optimizer/call/traced')
    start_time = datetime.datetime.now()
    seed = jax.random.PRNGKey(0) if seed is None else seed
    seed, acq_fn_seed = jax.random.split(seed)

    dimension_is_missing = jax.tree_util.tree_map(
        lambda pad_dim, dim: jnp.arange(pad_dim) >= dim,
        self.n_feature_dimensions_with_padding,
        self.n_feature_dimensions,
    )

    if n_parallel is None:
      # Squeeze out the singleton dimension of `features` before passing to a
      # non-parallel acquisition function to avoid batch shape collisions.
      # Use the same acquisition function seed at every call.
      eval_score_fn = lambda x: score_fn(  # pylint: disable=g-long-lambda
          _optimizer_to_model_input(
              x, self.n_feature_dimensions, squeeze_middle_dim=True
          ),
          acq_fn_seed,
      )
    else:
      eval_score_fn = lambda x: score_fn(  # pylint: disable=g-long-lambda
          _optimizer_to_model_input(x, self.n_feature_dimensions),
          acq_fn_seed,
      )

    # TODO: We should pass RNGKey to score_fn.
    prior_rewards = None
    parallel_dim = n_parallel or 1
    if prior_features is not None:
      continuous_prior, continuous_mask = _reshape_to_parallel_batches(
          prior_features.continuous, parallel_dim
      )
      categorical_prior, categorical_mask = _reshape_to_parallel_batches(
          prior_features.categorical, parallel_dim
      )
      prior_features = VectorizedOptimizerInput(
          continuous=continuous_prior, categorical=categorical_prior
      )
      prior_features = jax.tree_util.tree_map(
          lambda dim, feat: jnp.where(dim, jnp.zeros_like(feat), feat),
          dimension_is_missing,
          prior_features,
      )
      prior_rewards = eval_score_fn(prior_features)
      prior_rewards = jnp.where(
          jnp.logical_and(continuous_mask, categorical_mask),
          prior_rewards,
          -jnp.inf * jnp.ones_like(prior_rewards),
      )

    def _optimization_one_step(_, args):
      jax.monitoring.record_event(
          '/vizier/jax/vectorized_optimizer/call/one_step/traced'
      )
      state, best_results, seed = args
      suggest_seed, update_seed, new_seed = jax.random.split(seed, num=3)
      new_features = self.strategy.suggest(
          suggest_seed, state=state, n_parallel=parallel_dim
      )
      # Ensure masking out padded dimensions in new features.
      new_features = jax.tree_util.tree_map(
          lambda dim, feat: jnp.where(dim, jnp.zeros_like(feat), feat),
          dimension_is_missing,
          new_features,
      )

      new_rewards = eval_score_fn(new_features)
      new_state = self.strategy.update(
          update_seed, state, new_features, new_rewards
      )
      new_best_results = self._update_best_results(
          best_results, count, new_features, new_rewards
      )
      return new_state, new_best_results, new_seed

    init_seed, loop_seed = jax.random.split(seed)
    # TODO: Consider initializing with prior features/rewards.
    init_best_results = VectorizedStrategyResults(
        rewards=-jnp.inf * jnp.ones([count]),
        features=VectorizedOptimizerInput(
            continuous=jnp.zeros(
                [
                    count,
                    parallel_dim,
                    self.n_feature_dimensions_with_padding.continuous,
                ],
                dtype=self.dtype.continuous,
            ),
            categorical=jnp.zeros(
                [
                    count,
                    parallel_dim,
                    self.n_feature_dimensions_with_padding.categorical,
                ],
                dtype=self.dtype.categorical,
            ),
        ),
    )
    init_args = (
        self.strategy.init_state(
            init_seed,
            n_parallel=parallel_dim,
            prior_features=prior_features,
            prior_rewards=prior_rewards,
        ),
        init_best_results,
        loop_seed,
    )
    if self.use_fori:
      _, best_results, _ = jax.lax.fori_loop(
          0,
          (self.max_evaluations - 1) // self.suggestion_batch_size + 1,
          _optimization_one_step,
          init_args,
      )
    else:
      args = init_args
      for i in range(
          (self.max_evaluations - 1) // self.suggestion_batch_size + 1
      ):
        args = _optimization_one_step(i, args)
      _, best_results, _ = args

    if score_with_aux_fn:
      if n_parallel is None:
        aux = score_with_aux_fn(
            _optimizer_to_model_input(
                best_results.features,
                self.n_feature_dimensions,
                squeeze_middle_dim=True,
            ),
            seed=acq_fn_seed,
        )[1]
      else:
        aux = score_with_aux_fn(
            _optimizer_to_model_input(
                best_results.features, self.n_feature_dimensions
            ),
            seed=acq_fn_seed,
        )[1]

      best_results = VectorizedStrategyResults(
          best_results.features,
          best_results.rewards,
          aux,
      )

    logging.info(
        (
            'Optimization completed. Duration: %s. Evaluations: %s. Best'
            ' Results: %s'
        ),
        datetime.datetime.now() - start_time,
        (
            self.max_evaluations
            // self.suggestion_batch_size
            * self.suggestion_batch_size
        ),
        best_results,
    )

    return best_results

  def _update_best_results(
      self,
      best_results: VectorizedStrategyResults,
      count: int,
      batch_features: VectorizedOptimizerInput,
      batch_rewards: jax.Array,
  ) -> VectorizedStrategyResults:
    """Update the best results the optimizer seen thus far.

    The best results are kept in a heap to efficiently maintain the top 'count'
    results throughout the optimizer run.

    Arguments:
      best_results: A heap storing the best results seen thus far. Implemented
        as a list of maximum size of 'count'.
      count: The number of best results to store.
      batch_features: The current suggested features batch array with a
        dimension of (batch_size, feature_dim) or (batch_size, n_parallel,
        feature_dim).
      batch_rewards: The current reward batch array with a dimension of
        (batch_size,).

    Returns:
      trials:
    """
    all_rewards = jnp.concatenate([batch_rewards, best_results.rewards], axis=0)
    all_features = VectorizedOptimizerInput(
        continuous=jnp.concatenate(
            [batch_features.continuous, best_results.features.continuous],
            axis=0,
        ),
        categorical=jnp.concatenate(
            [batch_features.categorical, best_results.features.categorical],
            axis=0,
        ),
    )
    top_indices = jnp.argpartition(-all_rewards, count - 1)[:count]
    return VectorizedStrategyResults(
        rewards=all_rewards[top_indices],
        features=VectorizedOptimizerInput(
            continuous=all_features.continuous[top_indices],
            categorical=all_features.categorical[top_indices],
        ),
    )


# TODO: Should return suggestions not trials.
def best_candidates_to_trials(
    best_results: VectorizedStrategyResults,
    converter: converters.TrialToModelInputConverter,
) -> list[vz.Trial]:
  """Returns the best candidate trials in the original search space."""
  best_features = best_results.features
  trials = []
  sorted_ind = jnp.argsort(-best_results.rewards)
  for i in range(len(best_results.rewards)):
    # Create trials and convert the strategy features back to parameters.
    ind = sorted_ind[i]
    suggested_features = VectorizedOptimizerInput(
        best_features.continuous[ind], best_features.categorical[ind]
    )
    reward = best_results.rewards[ind]

    # Loop over the number of candidates per batch (which will be one, unless a
    # parallel acquisition function is used).
    for j in range(suggested_features.continuous.shape[0]):
      features = VectorizedOptimizerInput(
          continuous=jnp.expand_dims(suggested_features.continuous[j], axis=0),
          categorical=jnp.expand_dims(
              suggested_features.categorical[j], axis=0
          ),
      )
      trial = vz.Trial(
          parameters=converter.to_parameters(
              _optimizer_to_model_input(
                  features,
                  n_features=types.ContinuousAndCategorical(
                      len(converter.output_specs.continuous),
                      len(converter.output_specs.categorical),
                  ),
              )
          )[0]
      )
      metadata = trial.metadata.ns('devinfo')
      metadata['acquisition_optimization'] = json.dumps(
          {'acquisition': best_results.rewards[ind]}
          | jax.tree.map(
              lambda x, ind=ind: np.asarray(x[ind]), best_results.aux
          ),
          cls=json_utils.NumpyEncoder,
      )

      trial.complete(vz.Measurement({'acquisition': reward}))
      trials.append(trial)
  return trials


# TODO: This function should return jax types.
def trials_to_sorted_array(
    prior_trials: list[vz.Trial],
    converter: converters.TrialToModelInputConverter,
) -> Optional[types.ModelInput]:
  """Sorts trials by the order they were created and converts to array."""
  if prior_trials:
    prior_trials = sorted(prior_trials, key=lambda x: x.creation_time)
    prior_features = converter.to_features(prior_trials)
  else:
    prior_features = None
  return prior_features


@attr.define
class VectorizedOptimizerFactory:
  """Vectorized strategy optimizer factory."""

  strategy_factory: VectorizedStrategyFactory
  max_evaluations: int = 75_000
  suggestion_batch_size: int = 25
  use_fori: bool = True

  def __call__(
      self,
      converter: converters.TrialToModelInputConverter,
  ) -> VectorizedOptimizer:
    """Generates a new VectorizedOptimizer object."""
    strategy = self.strategy_factory(
        converter, suggestion_batch_size=self.suggestion_batch_size
    )
    n_feature_dimensions = getattr(
        strategy,
        'n_feature_dimensions',
        types.ContinuousAndCategorical(
            len(converter.output_specs.continuous),
            len(converter.output_specs.categorical),
        ),
    )
    empty_features = converter.to_features([])
    n_feature_dimensions_with_padding = getattr(
        strategy,
        'n_feature_dimensions_with_padding',
        types.ContinuousAndCategorical[int](
            empty_features.continuous.shape[-1],
            empty_features.categorical.shape[-1],
        ),
    )
    return VectorizedOptimizer(
        strategy=strategy,
        n_feature_dimensions=n_feature_dimensions,
        n_feature_dimensions_with_padding=n_feature_dimensions_with_padding,
        suggestion_batch_size=self.suggestion_batch_size,
        max_evaluations=self.max_evaluations,
        use_fori=self.use_fori,
        dtype=converter._impl.dtype,
    )
