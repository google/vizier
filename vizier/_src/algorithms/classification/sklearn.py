# Copyright 2022 Google LLC.
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

"""Sklearn Classifier."""

from typing import Optional, Union

import attr
import chex
import numpy as np
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF

from vizier._src.algorithms.classification import classifier

# TODO: Replace the sklearn GP classifier with TFP GP classifier
# once implemented.


@attr.define
class SklearnClassifier(classifier.BinaryClassifier):
  """Class for Sklearn classifiers.

  Attributes:
    classifier: a sklearn classifier such as SVM and GaussianProcessClassifier.
    features: (n, d) shaped array of n samples in dimension d.
    labels: (n, 1) shaped array of binary labels in {0, 1}.
    features_test: (m, d) shaped array of m samples in dimension d.
    eval_metric: a string denoting the evaluation metric. Accepted options are
      `probability` which estimates the probability of belonging to each class
      and `decision` which estimates a non-probability based metric such as the
      margin from the classification boundary.
  """
  classifier: Optional[Union[GaussianProcessClassifier, svm.SVC]] = attr.field(
      kw_only=True,
      default=GaussianProcessClassifier(
          kernel=ConstantKernel(1.) * RBF(length_scale=1.)))
  features_test: chex.Array = attr.field(kw_only=True)
  eval_metric: str = attr.field(kw_only=True, default='probability')

  def _check_features_and_labels_shapes(self, features: chex.Array,
                                        labels: chex.Array) -> None:
    """Checks the compatibility between features and labels shapes."""
    if np.ndim(features) != 2:
      raise ValueError(f'{self} expects 2d features.')
    if labels.shape[0] != features.shape[0]:
      raise ValueError(f'There are `{features.shape[0]}` features and '
                       f'`{labels.shape[0]}` labels which is incompatible')
    if np.ndim(labels) != 1 and labels.shape[1] != 1:
      raise ValueError(
          f'{self} expects 1d labels or labels of shape (num_samples, 1), but'
          f'was given labels of shape `{labels.shape}` .')
    if self.features_test.shape[1] != features.shape[1]:
      raise ValueError(f'{self} features_test to have `{features.shape[1]}`,'
                       f'but it has `{self.features_test.shape[1]}` features.')

  def _check_labels_values(self, labels: chex.Array) -> None:
    if not set(labels).issubset({0, 1}):
      raise ValueError('Labels should be either zero or one.')
    if set(labels).issubset({0}) or set(labels).issubset({1}):
      raise ValueError(f'{self} expects at least one sample per class, but all'
                       'training labels contain the same class.')

  def _check_eval_metric(self) -> None:
    if self.eval_metric not in ['probability', 'decision']:
      raise ValueError(f'{self} expects the evaluation metric to be'
                       f'`probability` or `decision ` but `{self.eval_metric}`'
                       'was given.')

  def train(self, features: chex.Array, labels: chex.Array) -> None:
    self._check_features_and_labels_shapes(features, labels)
    self._check_labels_values(labels)
    self._check_eval_metric()
    self.classifier.fit(np.asarray(features), np.asarray(labels))

  def evaluate(self, features: chex.Array) -> chex.Array:
    if self.eval_metric == 'probability':
      return self.classifier.predict_proba(np.asarray(self.features_test))[:, 1]
    elif self.eval_metric == 'decision':
      # Only works for SVM classifiers.
      return self.classifier.decision_function(np.asarray(self.features_test))  # pytype:disable=attribute-error
