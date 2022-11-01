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

"""Tests for classifier."""

import numpy as np
import sklearn
from vizier._src.algorithms.classification import classifier

from absl.testing import absltest
from absl.testing import parameterized


# TODO: convert it into a generic ClassifierTest for all subclasses.
class ClassifierTest(parameterized.TestCase):
  """Class for testing the classifier."""

  def setUp(self):
    super().setUp()
    self.features_train = np.array([[1.], [2.], [3.], [4.]])
    self.labels_train = np.array([0, 0, 1, 1])
    self.labels_train_invalid = np.array([10., 10., 11., 11.])
    self.features_test = np.array([[-1.], [6.], [0.], [6.]])

    self.classifier_instance = classifier.SupportVectorMachine()

  def test_raise_error_invalid_labels(self):
    with self.assertRaises(ValueError):
      self.classifier_instance._check_labels_train_values(
          self.labels_train_invalid)

  def test_scores_shape(self):
    self.classifier_instance.train(self.features_train, self.labels_train)
    scores = self.classifier_instance.evaluate(self.features_test)
    self.assertEqual(scores.shape[0], self.features_test.shape[0])

  def test_labels_shape(self, threshold=0):
    self.classifier_instance.train(self.features_train, self.labels_train)
    scores = self.classifier_instance.evaluate(self.features_test)
    labels_test_pred = (scores >= threshold).astype(float)
    labels_test_real = np.array([0, 1, 0, 1])
    self.assertTrue((labels_test_pred == labels_test_real).all())

  @parameterized.parameters([dict(label=0.), dict(label=1.)])
  def test_predictions_with_identical_labels(self, label):
    features_train = np.array([[1.], [2.], [3.], [4.]])
    labels_train = np.ones((features_train.shape[0],)) * label
    with self.assertRaises(ValueError):
      self.classifier_instance._check_labels_train_values(labels_train)

  def test_score_range(self):
    self.classifier_instance.train(self.features_train, self.labels_train)
    scores = self.classifier_instance.evaluate(self.features_test)
    max_distance = max(
        sklearn.metrics.pairwise.euclidean_distances(self.features_test).max(),
        sklearn.metrics.pairwise.euclidean_distances(self.features_train).max(),
        sklearn.metrics.pairwise.euclidean_distances(self.features_train,
                                                     self.features_test).max())
    self.assertLessEqual(scores.max(), max_distance)


if __name__ == '__main__':
  absltest.main()
