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

"""Tests for gp_bandit."""

import numpy as np
from vizier.pyvizier.converters import padding

from absl.testing import absltest
from absl.testing import parameterized


class PaddingTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.NONE,
              num_features=padding.PaddingType.NONE,
          ),
          num_trials=17,
          num_features=5,
          expected_num_trials=17,
          expected_num_features=5,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.NONE,
              num_features=padding.PaddingType.NONE,
          ),
          num_trials=23,
          num_features=8,
          expected_num_trials=23,
          expected_num_features=8,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.MULTIPLES_OF_10,
              num_features=padding.PaddingType.MULTIPLES_OF_10,
          ),
          num_trials=17,
          num_features=5,
          expected_num_trials=20,
          expected_num_features=10,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.POWERS_OF_2,
              num_features=padding.PaddingType.MULTIPLES_OF_10,
          ),
          num_trials=23,
          num_features=2,
          expected_num_trials=32,
          expected_num_features=10,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.MULTIPLES_OF_10,
              num_features=padding.PaddingType.MULTIPLES_OF_10,
          ),
          num_trials=7,
          num_features=8,
          expected_num_trials=10,
          expected_num_features=10,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.MULTIPLES_OF_10,
              num_features=padding.PaddingType.MULTIPLES_OF_10,
          ),
          num_trials=7,
          num_features=20,
          expected_num_trials=10,
          expected_num_features=20,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.MULTIPLES_OF_10,
              num_features=padding.PaddingType.POWERS_OF_2,
          ),
          num_trials=123,
          num_features=22,
          expected_num_trials=130,
          expected_num_features=32,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.POWERS_OF_2,
              num_features=padding.PaddingType.POWERS_OF_2,
          ),
          num_trials=17,
          num_features=5,
          expected_num_trials=32,
          expected_num_features=8,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.POWERS_OF_2,
              num_features=padding.PaddingType.POWERS_OF_2,
          ),
          num_trials=23,
          num_features=2,
          expected_num_trials=32,
          expected_num_features=2,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.POWERS_OF_2,
              num_features=padding.PaddingType.MULTIPLES_OF_10,
          ),
          num_trials=7,
          num_features=8,
          expected_num_trials=8,
          expected_num_features=10,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.POWERS_OF_2,
              num_features=padding.PaddingType.POWERS_OF_2,
          ),
          num_trials=7,
          num_features=17,
          expected_num_trials=8,
          expected_num_features=32,
      ),
      dict(
          schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.POWERS_OF_2,
              num_features=padding.PaddingType.POWERS_OF_2,
          ),
          num_trials=123,
          num_features=22,
          expected_num_trials=128,
          expected_num_features=32,
      ),
  )
  def test_padding(
      self,
      schedule,
      num_trials,
      num_features,
      expected_num_trials,
      expected_num_features,
  ):
    features = np.random.randn(num_trials, num_features)
    labels = np.random.randn(num_trials)[..., np.newaxis]

    padded_features, padded_labels = padding.pad_features_and_labels(
        features, labels, schedule
    )

    self.assertLen(padded_features.is_missing, 2)
    self.assertLen(padded_labels.is_missing, 1)

    self.assertTrue(
        np.all(
            np.isclose(
                padded_features.is_missing[0], padded_labels.is_missing[0]
            )
        )
    )

    label_is_missing = padded_features.is_missing[0]
    feature_is_missing = padded_features.is_missing[1]
    padded_features = padded_features.padded_array
    padded_labels = padded_labels.padded_array

    self.assertEqual(
        padded_features.shape, (expected_num_trials, expected_num_features)
    )
    self.assertEqual(feature_is_missing.shape, (expected_num_features,))
    self.assertEqual(padded_labels.shape, (expected_num_trials, 1))
    self.assertEqual(label_is_missing.shape, (expected_num_trials,))

    self.assertTrue(
        np.all(
            np.isclose(features, padded_features[:num_trials, :num_features])
        )
    )
    self.assertTrue(np.all(np.isclose(labels, padded_labels[:num_trials])))
    self.assertTrue(np.all(~label_is_missing[:num_trials]))
    self.assertTrue(np.all(label_is_missing[num_trials:]))

    self.assertTrue(np.all(~feature_is_missing[:num_features]))
    self.assertTrue(np.all(feature_is_missing[num_features:]))


if __name__ == '__main__':
  absltest.main()
