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

"""Tests for vizier.service.resources."""

from vizier._src.service import resources
from vizier._src.service import study_pb2
from vizier._src.service.testing import util as test_util

from absl.testing import absltest
from absl.testing import parameterized


class UtilTest(parameterized.TestCase):

  def setUp(self):
    self.owner_id = 'my_username'
    self.study_id = '12312312'
    self.trial_id = 1
    self.client_id = 'client_0'
    self.operation_number = 5
    super().setUp()

  def test_parsing_correct(self):
    """Tests for correct resource strings only."""
    owner_resource = resources.OwnerResource(self.owner_id)
    same_owner_resource = resources.OwnerResource.from_name(owner_resource.name)
    self.assertEqual(owner_resource, same_owner_resource)

    study_resource = resources.StudyResource(self.owner_id, self.study_id)
    same_study_resource = resources.StudyResource.from_name(study_resource.name)
    self.assertEqual(study_resource, same_study_resource)

    trial_resource = resources.TrialResource(
        self.owner_id, self.study_id, self.trial_id
    )
    same_trial_resource = resources.TrialResource.from_name(trial_resource.name)
    self.assertEqual(trial_resource, same_trial_resource)
    self.assertEqual(trial_resource.study_resource, study_resource)

    early_stopping_op_resource = resources.EarlyStoppingOperationResource(
        self.owner_id, self.study_id, self.trial_id
    )
    same_early_stopping_op_resource = (
        resources.EarlyStoppingOperationResource.from_name(
            early_stopping_op_resource.name
        )
    )
    self.assertEqual(
        early_stopping_op_resource, same_early_stopping_op_resource
    )
    self.assertEqual(early_stopping_op_resource.trial_resource, trial_resource)
    self.assertEqual(
        trial_resource.early_stopping_operation_resource,
        early_stopping_op_resource,
    )

    suggestion_op_resource = resources.SuggestionOperationResource(
        self.owner_id, self.study_id, self.client_id, self.operation_number
    )
    same_suggestion_op_resource = (
        resources.SuggestionOperationResource.from_name(
            suggestion_op_resource.name
        )
    )
    self.assertEqual(same_suggestion_op_resource, same_suggestion_op_resource)

  @parameterized.named_parameters(
      ('owner', 'owner/my_username', resources.OwnerResource),
      ('study', 'owners/my_username/study/cifar10', resources.StudyResource),
      (
          'trial',
          'owners/my_username/studies/cifar10/trials/not_an_int',
          resources.TrialResource,
      ),
  )
  def test_parsing_wrong(self, bad_name, resource_class):
    """Tests for incorrect resource strings, for input validation."""
    with self.assertRaises(ValueError):
      resource_class.from_name(bad_name)

  def test_generate_studies(self):
    study_spec = test_util.generate_all_four_parameter_specs()
    self.assertLen(study_spec.parameters, 4)

  def test_generate_trials(self):
    basic_trials = test_util.generate_trials(
        range(2), self.owner_id, self.study_id
    )
    self.assertLen(basic_trials, 2)
    all_states_trials = test_util.generate_all_states_trials(
        0, self.owner_id, self.study_id
    )
    self.assertLen(all_states_trials, len(study_pb2.Trial.State.keys()))

  def test_generate_suggestion_operations(self):
    basic_operations = test_util.generate_suggestion_operations(
        range(4), self.owner_id, self.study_id, self.client_id
    )
    self.assertLen(basic_operations, 4)


if __name__ == '__main__':
  absltest.main()
