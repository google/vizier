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

"""Tests for gp_bandit."""

import numpy as np
from vizier.pyvizier.converters import padding

from absl.testing import absltest
from absl.testing import parameterized


class PaddingTest(parameterized.TestCase):

  def test_padding(self):
    features = np.random.randn(13, 7)
    labels = np.random.randn(13, 3)

    schedule = padding.PaddingSchedule(
        num_trials=padding.PaddingType.MULTIPLES_OF_10,
        num_features=padding.PaddingType.POWERS_OF_2,
        num_metrics=padding.PaddingType.POWERS_OF_2,
    )

    padded = schedule.pad_features(features)

    self.assertSequenceEqual(padded.padded_array.shape, (20, 8))
    self.assertSequenceEqual(padded.unpad().shape, features.shape)

    padded = schedule.pad_labels(labels)
    self.assertSequenceEqual(padded.padded_array.shape, (20, 4))
    self.assertSequenceEqual(padded.unpad().shape, labels.shape)

  def test_nopadding(self):
    features = np.random.randn(13, 7)
    labels = np.random.randn(13, 3)

    schedule = padding.PaddingSchedule(
        num_trials=padding.PaddingType.NONE,
        num_features=padding.PaddingType.NONE,
        num_metrics=padding.PaddingType.NONE,
    )

    padded = schedule.pad_features(features)
    self.assertSequenceEqual(padded.padded_array.shape, features.shape)

    padded = schedule.pad_features(labels)
    self.assertSequenceEqual(padded.padded_array.shape, labels.shape)


if __name__ == '__main__':
  absltest.main()
