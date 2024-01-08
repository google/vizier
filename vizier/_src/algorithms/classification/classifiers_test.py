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

"""Tests for Sklearn classifier."""

import numpy as np
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics.pairwise import euclidean_distances
from vizier._src.algorithms.classification import classifiers

from absl.testing import absltest
from absl.testing import parameterized

svm_classifier = svm.SVC(kernel='rbf', C=1.)
gpc_classifier = GaussianProcessClassifier(
    kernel=ConstantKernel(1.) * RBF(length_scale=1.))


# TODO: convert it into a generic ClassifierTest for all subclasses.
class SklearnClassifierTest(parameterized.TestCase):
  """Class for testing the classifier."""

  def setUp(self):
    super().setUp()
    self.features_train = np.array([[1.], [2.], [3.], [4.]])
    self.labels_train = np.array([0, 0, 1, 1])
    self.labels_train_invalid = np.array([10., 10., 11., 11.])
    self.features_test = np.array([[-1.], [6.], [0.], [6.]])

  def test_raise_error_invalid_labels(self):
    classifier_instance = classifiers.SklearnClassifier(
        classifier=gpc_classifier,
        features=self.features_train,
        labels=self.labels_train_invalid,
        features_test=self.features_test,
        eval_metric='probability')
    with self.assertRaises(ValueError):
      classifier_instance._check_labels_values()

  @parameterized.parameters([
      dict(classifier=svm_classifier, eval_metric='decision'),
      dict(classifier=gpc_classifier, eval_metric='probability')
  ])
  def test_scores_shape(self, classifier, eval_metric):
    classifier_instance = classifiers.SklearnClassifier(
        classifier=classifier,
        features=self.features_train,
        labels=self.labels_train,
        features_test=self.features_test,
        eval_metric=eval_metric)
    scores = classifier_instance()
    self.assertEqual(scores.shape[0], self.features_test.shape[0])

  @parameterized.parameters([
      dict(classifier=svm_classifier, eval_metric='decision', threshold=0.),
      dict(classifier=gpc_classifier, eval_metric='probability', threshold=0.5)
  ])
  def test_labels_shape(self, classifier, eval_metric, threshold):
    classifier_instance = classifiers.SklearnClassifier(
        classifier=classifier,
        features=self.features_train,
        labels=self.labels_train,
        features_test=self.features_test,
        eval_metric=eval_metric)
    scores = classifier_instance()
    labels_test_pred = (scores >= threshold).astype(float)
    labels_test_real = np.array([0, 1, 0, 1])
    self.assertTrue((labels_test_pred == labels_test_real).all())

  @parameterized.parameters([dict(label_val=0.), dict(label_val=1.)])
  def test_raise_error_identical_labels(self, label_val):
    labels_train_identical = np.ones(
        (self.features_train.shape[0],)) * label_val
    classifier_instance = classifiers.SklearnClassifier(
        classifier=gpc_classifier,
        features=self.features_train,
        labels=labels_train_identical,
        features_test=self.features_test,
        eval_metric='probability')
    with self.assertRaises(ValueError):
      classifier_instance._check_labels_values()

  @parameterized.parameters([
      dict(classifier=svm_classifier, eval_metric='decision'),
      dict(classifier=gpc_classifier, eval_metric='probability')
  ])
  def test_scores_range(self, classifier, eval_metric):
    classifier_instance = classifiers.SklearnClassifier(
        classifier=classifier,
        features=self.features_train,
        labels=self.labels_train,
        features_test=self.features_test,
        eval_metric=eval_metric)
    scores = classifier_instance()
    if eval_metric == 'probability':
      self.assertGreaterEqual(scores.min(), 0.)
      self.assertLessEqual(scores.max(), 1.)
    else:
      max_distance = max(
          euclidean_distances(self.features_test).max(),
          euclidean_distances(self.features_train).max(),
          euclidean_distances(self.features_train, self.features_test).max())
      self.assertLessEqual(scores.max(), max_distance)

  @parameterized.parameters([
      dict(classifier=svm_classifier, eval_metric='decision', threshold=0.),
      dict(classifier=gpc_classifier, eval_metric='probability', threshold=0.5)
  ])
  def test_prediction_on_train_data(self, classifier, eval_metric, threshold):
    classifier_instance = classifiers.SklearnClassifier(
        classifier=classifier,
        features=self.features_train,
        labels=self.labels_train,
        features_test=self.features_train,
        eval_metric=eval_metric)
    scores = classifier_instance()
    labels_test_pred = (scores >= threshold).astype(float)
    self.assertTrue((labels_test_pred == self.labels_train).all())


if __name__ == '__main__':
  absltest.main()
