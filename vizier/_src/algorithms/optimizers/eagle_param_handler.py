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

import attr
import numpy as np
from vizier.pyvizier import converters


@attr.define(kw_only=True)
class EagleParamHandler:
  """Vectorized eagle strategy utils.

  The class is used to account for the different types of Vizier parameters and
  incorporate their naunces into the vectorized eagle strategy.

  Attributes:
    converter: The converter used to convert the original search space. It's
      used to get metadata on the different parameters and attribute the feature
      indices to them.
    config: The eagle strategy configuration used to set the perturbation
      factors for CATEGORICAL features.
    rng: A numpy random generator to generate parameter-aware features, and
      sample categorical features.
    categorical_perturbation_factor: The perturbation factor for categorical
      features.
    pure_categorical_perturbation_factor: The perturbation factor when all
      features are categorical.
    n_features: The total number of feature indices.
    n_categorical: The number of CATEGORICAL associated indices.
    has_categorical: A flag indicating if at least one feature is categorical.
    all_features_categorical: A flag indicating if all features are categorical.
    _categorical_param_mask: A 2D array (n_categorical, n_features) that for
      each categorical parameters (row) has 1s in its associated feature
      indices. The array is used for sampling categorical values.
    _categorical_mask: A 1D array (n_features,) with 1s in the indices of
      categorical features and otherwise 0s. The array is used for sampling
      categorical values.
    _tiebreak_mask: A 1D array (n_features,) with multiplies of epsilons used to
      tie breaking. The array is used for sampling categorical values.
    _oov_mask: A 1D array (n_features,) with 1s in the non-oov indices. The
      array is used to generate random features with 0 value in the OOV indices.
    _epsilon: A small value used in tie-breaker
  """
  converter: converters.TrialToArrayConverter
  categorical_perturbation_factor: float
  pure_categorical_perturbation_factor: float
  rng: np.random.Generator
  # Public variables created by the class
  n_features: int = attr.field(init=False)
  n_categorical: int = attr.field(init=False)
  has_categorical: bool = attr.field(init=False)
  all_features_categorical: bool = attr.field(init=False)
  # Internal variables
  _categorical_params_mask: np.ndarray = attr.field(init=False)
  _categorical_mask: np.ndarray = attr.field(init=False)
  _tiebreak_array: np.ndarray = attr.field(init=False)
  _oov_mask: Optional[np.ndarray] = attr.field(init=False, default=None)
  _epsilon: float = attr.field(init=False, default=1e-5)

  def __attrs_post_init__(self):
    self._init_utils()

  def _init_utils(self):
    """Initialize the utility class and cache arrays for repeated use."""
    valid_types = [
        converters.NumpyArraySpecType.ONEHOT_EMBEDDING,
        converters.NumpyArraySpecType.CONTINUOUS
    ]
    unsupported_params = sum([
        1 for spec in self.converter.output_specs
        if spec.type not in valid_types
    ])
    if unsupported_params:
      raise ValueError('Only CATEGORICAL/CONTINUOUS parameters are supported!')

    self.n_features = sum(
        [spec.num_dimensions for spec in self.converter.output_specs])
    self.n_categorical = sum([
        1 for spec in self.converter.output_specs
        if spec.type == converters.NumpyArraySpecType.ONEHOT_EMBEDDING
    ])
    self.has_categorical = self.n_categorical > 0
    self.all_features_categorical = self.n_features == self.n_categorical
    if self.has_categorical:
      self._create_categorical_masks()

  def _create_categorical_masks(self):
    """Create the categorical masks."""
    self._categorical_params_mask = np.zeros(
        (self.n_categorical, self.n_features))
    self._oov_mask = np.ones((self.n_features,))
    row = 0
    col = 0
    # Create a flag to indicate if the converter uses OOV padding. If none of
    # the CATEGORICAL params use padding the 'oov_mask' is set to None.
    is_pad_oov = False
    for spec in self.converter.output_specs:
      if spec.type == converters.NumpyArraySpecType.ONEHOT_EMBEDDING:
        n_dim = spec.num_dimensions
        self._categorical_params_mask[row, col:col + n_dim] = 1.0
        if spec.num_oovs:
          self._oov_mask[col + n_dim - 1] = 0.0
          is_pad_oov = True
        row += 1
        col += n_dim
      else:
        col += 1
    if not is_pad_oov:
      self._oov_mask = None

    self._tiebreak_array = np.array(
        [-self._epsilon * (i + 1) for i in range(self.n_features)])
    self._categorical_mask = np.sum(self._categorical_params_mask, axis=0)

  def sample_categorical(self, features: np.ndarray) -> np.ndarray:
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
      features: (batch_size, n_features)

    Returns:
      The features with sampled categorical parameters. (batch_size ,n_features)
    """
    if not self.has_categorical:
      return features
    batch_size = features.shape[0]
    # Broadcast features to: (batch_size, n_categorical, n_features)
    expanded_shape = (self.n_categorical,) + features.shape
    expanded_features = np.swapaxes(
        np.broadcast_to(features, expanded_shape), 1, 0)
    # Mask each row (which represents a categorical param) to remove values in
    # indices that aren't associated with the parameter indices.
    param_features = expanded_features * self._categorical_params_mask
    # Create probabilities from non-normalized parameter features values.
    probs = param_features / np.sum(param_features, axis=-1, keepdims=True)
    # Generate random uniform values to use for sampling.
    # TODO: Pre-compute random values in batch to improve performance.
    unifs = self.rng.uniform(0.0, 1.0, size=(batch_size, self.n_categorical))
    # Find the locations of the indices that exceed random values.
    locs = np.cumsum(probs, axis=-1) >= np.expand_dims(unifs, axis=-1)
    # Multiply by 'categorical_mask' to mask off cumsum in non-categorical
    # indices, and add 'tiebreak_mask' to find the first index.
    masked_locs = locs * self._categorical_params_mask + self._tiebreak_array
    # Generate the samples so that each parameter features has a single 1 value.
    sampled_categorical_params = np.float_(
        np.int_(masked_locs / np.max(masked_locs, axis=-1, keepdims=True)))
    # Flatten all the categories features to dimension (batch_size, n_features)
    sampled_features = np.sum(sampled_categorical_params, axis=1)
    # Mask categorical features and add the new sampled categorical values.
    return sampled_features + features * (1 - self._categorical_mask)

  def random_features(self, batch_size: int, n_features: int) -> np.ndarray:
    """Create random features with uniform distribution.

    In case there are CATEGORICAL features with OOV we use the 'oov_mask' which
    is 1D numpy array (n_features,) with 0s in OOV indices and otherwise 1s.
    After multiplying (and broadcasting) the randomly generated features with
    the mask we're guaranteed that no features will be created in OOV indices.
    Therefore when mutating fireflies' features (index by index), the final
    suggested features will also have 0s in the OOV indices as desired.

    Arguments:
      batch_size:
      n_features:

    Returns:
      The random features with out of vocabulary indices zeroed out.
    """
    size = (batch_size, n_features)
    if self._oov_mask is not None:
      # Don't create random values for CATEGORICAL features OOV indices.
      # Broadcasting: (batch_size, n_features) x (n_features,)
      return self.rng.uniform(0.0, 1.0, size=size) * self._oov_mask
    else:
      return self.rng.uniform(0.0, 1.0, size=size)

  @property
  def perturbation_factors(self) -> np.ndarray:
    """Create the perturbations factors.

    Returns:
      Array of perturbation factors (n_features,)
    """
    perturbation_factors = []

    if self.all_features_categorical:
      for spec in self.converter.output_specs:
        perturbation_factors.extend(
            [self.pure_categorical_perturbation_factor] * spec.num_dimensions)
    else:
      for spec in self.converter.output_specs:
        if spec.type == converters.NumpyArraySpecType.ONEHOT_EMBEDDING:
          perturbation_factors.extend([self.categorical_perturbation_factor] *
                                      spec.num_dimensions)

        elif spec.type == converters.NumpyArraySpecType.CONTINUOUS:
          perturbation_factors.append(1.0)
    return np.array(perturbation_factors)
