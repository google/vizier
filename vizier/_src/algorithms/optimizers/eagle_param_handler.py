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

"""Utils for vectorized eagle strategy."""

from typing import Optional

from flax import struct
import jax
from jax import numpy as jnp
from vizier.pyvizier import converters


@struct.dataclass
class EagleParamHandler:
  """Vectorized eagle strategy utils.

  The class is used to account for the different types of Vizier parameters and
  incorporate their naunces into the vectorized eagle strategy.

  Attributes:
    n_feature_dimensions: The total number of feature indices.
    n_categorical: The number of CATEGORICAL associated indices.
    has_categorical: A flag indicating if at least one feature is categorical.
    perturbation_factors: Array of continuous and categorical perturbation
      factors.
    _categorical_param_mask: A 2D array (n_categorical, n_feature_dimensions)
      that for each categorical parameters (row) has 1s in its associated
      feature indices. The array is used for sampling categorical values.
    _categorical_mask: A 1D array (n_feature_dimensions,) with 1s in the indices
      of categorical features and otherwise 0s. The array is used for sampling
      categorical values.
    _tiebreak_mask: A 1D array (n_feature_dimensions,) with multiplies of
      epsilons used to tie breaking. The array is used for sampling categorical
      values.
    _oov_mask: A 1D array (n_feature_dimensions,) with 1s in the non-oov
      indices. The array is used to generate random features with 0 value in the
      OOV indices.
    _epsilon: A small value used in tie-breaker
  """
  # Internal variables
  perturbation_factors: jax.Array
  _categorical_params_mask: jax.Array
  _categorical_mask: jax.Array
  _tiebreak_array: jax.Array
  _oov_mask: Optional[jax.Array]
  _epsilon: float
  # Public variables created by the class
  n_feature_dimensions: int = struct.field(pytree_node=False)
  n_categorical: int = struct.field(pytree_node=False)
  has_categorical: bool = struct.field(pytree_node=False)

  @classmethod
  def build(
      cls,
      converter: converters.TrialToArrayConverter,
      categorical_perturbation_factor: float,
      pure_categorical_perturbation_factor: float,
      epsilon: float = 1e-5,
  ) -> 'EagleParamHandler':
    """Docstring."""

    valid_types = [
        converters.NumpyArraySpecType.ONEHOT_EMBEDDING,
        converters.NumpyArraySpecType.CONTINUOUS
    ]
    unsupported_params = sum(
        1 for spec in converter.output_specs if spec.type not in valid_types
    )
    if unsupported_params:
      raise ValueError('Only CATEGORICAL/CONTINUOUS parameters are supported!')

    n_feature_dimensions = converter.to_features([]).shape[-1]
    n_categorical = sum(
        1
        for spec in converter.output_specs
        if spec.type == converters.NumpyArraySpecType.ONEHOT_EMBEDDING
    )
    has_categorical = n_categorical > 0
    all_features_categorical = n_feature_dimensions == n_categorical
    oov_mask = None
    tiebreak_array = None
    categorical_mask = None
    categorical_params_mask = None
    if has_categorical:
      categorical_params_mask = jnp.zeros((n_categorical, n_feature_dimensions))
      oov_mask = jnp.ones((n_feature_dimensions,))
      row = 0
      col = 0
      # Create a flag to indicate if the converter uses OOV padding. If none of
      # the CATEGORICAL params use padding the 'oov_mask' is set to None.
      is_pad_oov = False
      for spec in converter.output_specs:
        if spec.type == converters.NumpyArraySpecType.ONEHOT_EMBEDDING:
          n_dim = spec.num_dimensions
          categorical_params_mask = categorical_params_mask.at[
              row, col : col + n_dim
          ].set(1.0)
          if spec.num_oovs:
            oov_mask = oov_mask.at[col + n_dim - 1].set(0.0)
            is_pad_oov = True
          row += 1
          col += n_dim
        else:
          col += 1

      oov_mask = oov_mask if is_pad_oov else None
      tiebreak_array = -epsilon * jnp.arange(1, n_feature_dimensions + 1)
      categorical_mask = jnp.sum(categorical_params_mask, axis=0)

    perturbation_factors = []

    if all_features_categorical:
      for spec in converter.output_specs:
        perturbation_factors.extend(
            [pure_categorical_perturbation_factor] * spec.num_dimensions
        )
    else:
      for spec in converter.output_specs:
        if spec.type == converters.NumpyArraySpecType.ONEHOT_EMBEDDING:
          perturbation_factors.extend(
              [categorical_perturbation_factor] * spec.num_dimensions
          )

        elif spec.type == converters.NumpyArraySpecType.CONTINUOUS:
          perturbation_factors.append(1.0)
    # Add any extra dimensions at the end.
    perturbation_factors.extend(
        [0.0] * (n_feature_dimensions - len(perturbation_factors))
    )
    perturbation_factors = jnp.array(perturbation_factors)
    return EagleParamHandler(
        n_feature_dimensions=n_feature_dimensions,
        n_categorical=n_categorical,
        has_categorical=has_categorical,
        perturbation_factors=perturbation_factors,
        _categorical_params_mask=categorical_params_mask,
        _categorical_mask=categorical_mask,
        _tiebreak_array=tiebreak_array,
        _oov_mask=oov_mask,
        _epsilon=epsilon,
    )

  def sample_categorical(
      self, features: jax.Array, seed: jax.random.KeyArray
  ) -> jax.Array:
    """Sample categorical features.

    The categorical sampling is used before returning suggestion to ensure that
    only actual categorical values are suggested. Non categorical features are
    left unchanged.

    For example: If 'features' has one categorical parameter with 3 values and
    one float parameter the conversion will be take the form of:
    (0.1, 0.3, 0.5, 0.4578) -> (0, 1, 0, 0.4578).

    Implementation details:
    ----------------------
    1. For each categorical parameter, isolate its indices in a separate row.
    2. Normalize new row values to probabilities through dividing by the sum.
    3. Create cummulatitive sum of probabilites to generat a CDF.
    4. Add small incremental values to CDF to tie-break (explained more later).
    5. Randomize uniform values for each row to determine which value to sample.
    6. Find the minimum index that its CDF > uniform. See code around
    'sampled_categorical_params' below for more detail and for why we need to
    add the tie-breaking array values.
    7. Flatten each row sampled values and combine with original features.

    Arguments:
      features: (batch_size, n_feature_dimensions)
      seed: Random seed.

    Returns:
      The features with sampled categorical parameters.
        (batch_size, n_feature_dimensions)
    """
    if not self.has_categorical:
      return features
    batch_size = features.shape[0]
    # Mask each row (which represents a categorical param) to remove values in
    # indices that aren't associated with the parameter indices.
    param_features = features[:, jnp.newaxis, :] * self._categorical_params_mask
    # Create probabilities from non-normalized parameter features values.
    probs = param_features / jnp.sum(param_features, axis=-1, keepdims=True)
    # Generate random uniform values to use for sampling.
    # TODO: Pre-compute random values in batch to improve performance.
    unifs = jax.random.uniform(seed, shape=(batch_size, self.n_categorical, 1))
    # Find the locations of the indices that exceed random values.
    locs = jnp.cumsum(probs, axis=-1) >= unifs
    # Multiply by 'categorical_mask' to mask off cumsum in non-categorical
    # indices, and add 'tiebreak_mask' to find the first index.
    masked_locs = locs * self._categorical_params_mask + self._tiebreak_array
    # Generate the samples so that each parameter features has a single 1 value.
    sampled_categorical_params = jnp.trunc(
        masked_locs / jnp.max(masked_locs, axis=-1, keepdims=True)
    )
    # Flatten all the categories features to dimension
    # (batch_size, n_feature_dimensions)
    sampled_features = jnp.sum(sampled_categorical_params, axis=1)
    # Mask categorical features and add the new sampled categorical values.
    return sampled_features + features * (1 - self._categorical_mask)

  def random_features(
      self, batch_size: int, seed: jax.random.KeyArray
  ) -> jax.Array:
    """Create random features with uniform distribution.

    In case there are CATEGORICAL features with OOV we use the 'oov_mask' which
    is 1D numpy array (n_feature_dimensions,) with 0s in OOV indices and
    otherwise 1s.  After multiplying (and broadcasting) the randomly generated
    features with the mask we're guaranteed that no features will be created in
    OOV indices.  Therefore when mutating fireflies' features (index by index),
    the final suggested features will also have 0s in the OOV indices as
    desired.

    Arguments:
      batch_size:
      seed: Random seed.

    Returns:
      The random features with out of vocabulary indices zeroed out.
    """
    features = jax.random.uniform(
        seed, shape=(batch_size, self.n_feature_dimensions)
    )
    if self._oov_mask is not None:
      # Don't create random values for CATEGORICAL features OOV indices.
      # Broadcasting:
      #   (batch_size, n_feature_dimensions) x (n_feature_dimensions,)
      features = features * self._oov_mask
    return features
