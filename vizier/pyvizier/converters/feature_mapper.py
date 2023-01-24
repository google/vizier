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

"""Mappers between different feature formats."""

import collections

import numpy as np
from vizier.pyvizier.converters import core


# This namedtuple was copied from TFP to reduce dependency.
ContinuousAndCategoricalValues = collections.namedtuple(
    'ContinuousAndCategoricalValues', ['continuous', 'categorical']
)


class ContinuousCategoricalFeatureMapper:
  """Maps features from TrialToArrayConverter to ContinuousAndCategoricalValues.

  The class allows to separate the continuous features from the categorical
  features and organize them in an ContinuousAndCategoricalValues namedtuple.

  Naming convension:
    parameter - Vizier parameter.
    feature - Numpy array representing a converted Vizier parameter.

  Notation for dimensions:
    C - continuous parameters count
    P - categorical parameters count
    F - total categorial dimensions (i.e. the total feasible values count, which
      associated with the onehot-encoding features total count)
    B - number of trials
  """

  def __init__(self, converter: core.TrialToArrayConverter):
    """Initiate helper objects to split features by type."""
    # Number of categorical parameters.
    self._n_categorical_params = 0
    # The continuous feature indices of output_spec, which also correspond to
    # the columns indices of continuous parameters in features.
    self._continuous_indices = []
    # The categorical feature indices.
    self._categorical_indices = []
    # The dimensions of categorical parameters.
    categorical_dims = []
    ind = 0
    # Extract the continuous and categorical feature indices.
    for spec in converter.output_specs:
      if spec.type == core.NumpyArraySpecType.CONTINUOUS:
        self._continuous_indices.append(ind)
        ind += 1
      elif spec.type == core.NumpyArraySpecType.ONEHOT_EMBEDDING:
        self._categorical_indices.extend(range(ind, ind + spec.num_dimensions))
        ind += spec.num_dimensions
        self._n_categorical_params += 1
        categorical_dims.append(spec.num_dimensions)
      elif spec.type == core.NumpyArraySpecType.DISCRETE:
        # There shouldn't be DISCRETE spec type as onehot_embed=True is the
        # default in TrialToArrayConverter, and DISCRETE parameters are either
        # converted to CONTINUOUS or ONEHOT_EMBEDDING.
        raise ValueError("DISCRETE spec type shouldn't exists.")
      else:
        raise ValueError('Unexpected spec type %s!' % spec.type)
    # Create shift array that contains the cumulative number of categorical
    # dimensions until each categorical parameter such that the categorical
    # parameter itself is not included, and therefore the padding and [0:-1].
    # This array will be used to map between the onehot active bit (index) of
    # each categorical parameter to respective integer categorical value.
    # For example, if categorical_dims=[3, 2], then cumsum(pad(..)) will
    # generate [0, 3, 5], so shift will be [0, 3].
    self.shift = np.cumsum(np.pad(categorical_dims, (1, 0)))[0:-1]  # (P,)

  def map(self, features: np.ndarray) -> ContinuousAndCategoricalValues:
    """Split features by 'continuous' and 'categorical'.

    In addition to splitting the method converts the one-hot encoding
    categorical features to integer indices.

    For example, if the search space is:
      - categorical parameter C1 with values ['a', 'b', 'c].
      - continuous parameter F1.
      - categorical parameter C2 with values ['x', 'y'].

    Then for input parameters C1='b', F1=0.23, C2='y' and associated features:
    [[0, 1, 0, 0.23, 0, 1]] the result would be:
    ContinuousAndCategoricalValues(
        continuous=[[0.23]]
        categorical=[[1,1]])

    Arguments:
      features: Numpy array (trials_num, feature_count) which is the output of
        'to_features' method.

    Returns:
      namedtuple with continuous features and categorical integer indices.
    """
    # Split features to continuous and categorical.
    continuous_features = features[:, self._continuous_indices]  # (B,C)
    # Assign empty array as the default value.
    categorical_index_features = np.zeros((features.shape[0], 0))
    if self._n_categorical_params > 0:
      categorical_features = features[:, self._categorical_indices]  # (B,F)
      # Find the non-zero column indices associated with categorical parameters.
      # For example (cont. from above), with the input of [0, 1, 0, 0.23, 0, 1]
      # 'categorical_features' is [[0,1,0,0,1]] and 'nonzero_indices' is [1,4]
      nonzero_indices = np.nonzero(categorical_features)[1]  # (P*B,)
      # Reshape the non-zero indices to align with the no. of categorical params
      # and shift by the cardinality of each parameter to compute the integer
      # category value expected by ContinuousAndCategoricalValues.
      # For example (cont. from above), [[1,4]] - [0,3] = [1, 1]
      categorical_index_features = (
          np.reshape(nonzero_indices, (-1, self._n_categorical_params))  # (B,P)
          - self.shift  # (P,)
      )  # (B,P)
    return ContinuousAndCategoricalValues(
        continuous=continuous_features, categorical=categorical_index_features
    )
