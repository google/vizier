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

"""Binary classifiers for Bayesian Optimization."""

import abc
from typing import Any, Literal
import chex

import numpy as np
from sklearn import svm


class BinaryClassifier(abc.ABC):
  """Abstract class for classification."""

  @abc.abstractmethod
  def train(self, features: chex.Array, labels: chex.Array) -> None:
    """Trains the classifier.

    Args:
      features: (m, d) shaped array of m samples in dimension d.
      labels: (m, 1) or (m,) shaped array of binary (0, 1) labels.
    """

  @abc.abstractmethod
  def evaluate(self, features: chex.Array) -> chex.Array:
    """Evaluates scores for test data.

    Args:
      features: (n, d) shaped array of n samples in dimension d.  The scores can
        be used to predict the labels for any test features. In details, given a
        threshold, labels = scores >= threshold. If scores are
        probability-based, a good default threshold is 0.5 and if scores are
        margin-based (eg. for SVM), a good default threshold is 0.

    Returns:
      (n,) shaped array of evaluated scores at features.
    """


class SupportVectorMachine(BinaryClassifier):
  """Class for SVM classifier."""

  def __init__(self,
               *,
               kernel: Literal['linear', 'poly', 'rbf', 'sigmoid',
                               'precomputed'] = 'rbf',
               penalty_misclass: float = 1.):
    """Sets up the setting for SVM.

    Args:
      kernel: string denoting the type of kernel.
      penalty_misclass: regularization parameter to penalize miscalssification
        of training labels. For more details, see
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html.
    """

    self.kernel = kernel
    self.penalty_misclass = penalty_misclass
    self.model = None
    self._assert_kernel_type()

  def _assert_kernel_type(self) -> None:
    if self.kernel not in {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}:
      raise ValueError('Accepted kernels are linear, poly, rbf, sigmoid,'
                       f'precomputed, but was given `{self.kernel}`.')

  def _check_features_and_labels_train_shapes(self, features: chex.Array,
                                              labels: chex.Array) -> Any:
    if np.ndim(features) != 2:
      raise ValueError(f'{self} expect 2d features.')
    if labels.shape[0] != features.shape[0]:
      raise ValueError(f'There are `{features.shape[0]}` features and '
                       f'`{labels.shape[0]}` labels which is incompatible')
    if np.ndim(labels) != 1 and labels.shape[1] != 1:
      raise ValueError(
          f'{self} expects 1d labels or labels of shape (num_samples, 1), but'
          f'was given labels of shape `{labels.shape}` .')

  def _check_labels_train_values(self, labels: chex.Array):
    if not set(labels).issubset({0, 1}):
      raise ValueError('Labels should be either zero or one.')
    if set(labels).issubset({0}) or set(labels).issubset({1}):
      raise ValueError(f'{self} expects at least one sample per class, but all'
                       'training labels contain the same class.')

  def train(self, features: chex.Array, labels: chex.Array) -> None:
    features = np.asarray(features)
    labels = np.asarray(labels)
    self._check_labels_train_values(labels)
    self._check_features_and_labels_train_shapes(features, labels)
    finite_labels_num = (labels == 0).sum()
    infinite_labels_num = (labels == 1).sum()
    self.model = svm.SVC(
        kernel=self.kernel,
        class_weight={
            0: finite_labels_num,
            1: infinite_labels_num
        },
        C=self.penalty_misclass)
    self.model.fit(features, labels)

  def evaluate(self, features: chex.Array) -> np.ndarray:
    features = np.asarray(features)
    return self.model.decision_function(features)
