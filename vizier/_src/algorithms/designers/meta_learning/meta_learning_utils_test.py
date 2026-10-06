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

"""Tests for meta_learning_utils."""

import random
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.meta_learning import meta_learning_utils
from absl.testing import absltest
from absl.testing import parameterized


class MetaLearningUtilsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    space = vz.SearchSpace()
    space.root.add_int_param('tuned_param', 0, 100, default_value=55)
    self.utils = meta_learning_utils.MetaLearningUtils(
        goal=vz.ObjectiveMetricGoal.MAXIMIZE,
        tuned_metric_name='tuned_obj',
        meta_metric_name='meta_obj',
        tuning_params=space,
    )
    self.meta_trials = []
    for i in range(10):
      meta_trial = vz.Trial({'meta_param': i})
      meta_trial.complete(vz.Measurement(metrics={'meta_obj': float(i)}))
      self.meta_trials.append(meta_trial)
    random.shuffle(self.meta_trials)

    self.tuned_trials = []
    for i in range(50):
      tuned_trial = vz.Trial({'tuned_param': i})
      tuned_trial.complete(vz.Measurement(metrics={'tuned_obj': float(i)}))
      self.tuned_trials.append(tuned_trial)
    random.shuffle(self.tuned_trials)

  def test_best_trial(self):
    # meta trial
    best_meta_trial = self.utils.get_best_meta_trial(self.meta_trials)
    self.assertEqual(best_meta_trial.parameters['meta_param'].value, 9)
    # tuned trial
    best_tuned_trial = self.utils.get_best_tuned_trial(self.tuned_trials)
    self.assertEqual(best_tuned_trial.parameters['tuned_param'].value, 49)

  def test_best_trial_score(self):
    # meta trial score
    best_meta_trial_score = self.utils.get_best_meta_trial_score(
        self.meta_trials
    )
    self.assertEqual(best_meta_trial_score, 9.0)
    # tuned trial score
    best_tuned_trial_score = self.utils.get_best_tuned_trial_score(
        self.tuned_trials
    )
    self.assertEqual(best_tuned_trial_score, 49.0)

  def test_generate_default_tuned_parameters(self):
    tuned_suggestion = self.utils.get_default_hyperparameters()
    self.assertEqual(tuned_suggestion.parameters['tuned_param'].value, 55)


if __name__ == '__main__':
  absltest.main()
