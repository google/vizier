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

from vizier._src.service import constants
from vizier._src.service import resources
from vizier._src.service import service_policy_supporter
from vizier._src.service import study_pb2
from vizier._src.service import vizier_service
from vizier._src.service.testing import util as test_util
from vizier.service import pyvizier

from absl.testing import absltest


class PythiaSupporterTest(absltest.TestCase):

  def setUp(self):
    self.owner_id = 'my_username'
    self.study_id = '1231223'
    self.study_name = resources.StudyResource(
        owner_id=self.owner_id, study_id=self.study_id
    ).name
    self.vs = vizier_service.VizierServicer(
        database_url=constants.SQL_MEMORY_URL
    )
    self.example_study = test_util.generate_study(self.owner_id, self.study_id)
    self.vs.datastore.create_study(self.example_study)

    self.active_trials = test_util.generate_trials(
        range(1, 7), self.owner_id, self.study_id
    )
    self.succeeded_trial = test_util.generate_trials(
        [7],
        self.owner_id,
        self.study_id,
        state=study_pb2.Trial.State.SUCCEEDED,
        final_measurement=study_pb2.Measurement(),
    )[0]

    for trial in self.active_trials + [self.succeeded_trial]:
      self.vs.datastore.create_trial(trial)

    self.policy_supporter = service_policy_supporter.ServicePolicySupporter(
        self.study_name, self.vs
    )

    super().setUp()

  def test_trial_names_filter(self):
    trials = self.policy_supporter.GetTrials(
        study_guid=self.study_name, trial_ids=[3, 4]
    )

    self.assertEqual(
        trials[0], pyvizier.TrialConverter.from_proto(self.active_trials[2])
    )
    self.assertEqual(
        trials[1], pyvizier.TrialConverter.from_proto(self.active_trials[3])
    )

  def test_min_max_filter(self):
    trials = self.policy_supporter.GetTrials(
        study_guid=self.study_name, min_trial_id=3, max_trial_id=4
    )

    self.assertEqual(
        trials[0], pyvizier.TrialConverter.from_proto(self.active_trials[2])
    )
    self.assertEqual(
        trials[1], pyvizier.TrialConverter.from_proto(self.active_trials[3])
    )

  def test_status_match_filter(self):
    trials = self.policy_supporter.GetTrials(
        study_guid=self.study_name, status_matches=pyvizier.TrialStatus.ACTIVE
    )

    self.assertLen(trials, 6)
    self.assertEqual(
        trials[0], pyvizier.TrialConverter.from_proto(self.active_trials[0])
    )

  def test_raise_value_error(self):
    def should_raise_value_error_fn():
      self.policy_supporter.GetTrials(
          study_guid=self.study_name,
          trial_ids=[1],
          status_matches=pyvizier.TrialStatus.ACTIVE,
      )

      with self.assertRaises(ValueError):
        should_raise_value_error_fn()

  def test_get_study_config(self):
    pythia_problem = self.policy_supporter.GetStudyConfig(self.study_name)
    correct_pythia_problem = pyvizier.StudyConfig.from_proto(
        self.example_study.study_spec
    ).to_problem()
    self.assertEqual(pythia_problem, correct_pythia_problem)


if __name__ == '__main__':
  absltest.main()
