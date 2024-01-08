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

"""Tests for designer_policy."""

from typing import Optional, Sequence
from vizier import algorithms as vza
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.algorithms.policies import designer_policy as dp
from vizier.interfaces import serializable
from absl.testing import absltest


class _FakeDesigner(vza.Designer):

  def __init__(self):
    self.num_incorporated_completed_trials = 0
    self._num_incorporated_active_trials = 0
    self.last_completed = []
    self.last_active = []

  def suggest(
      self, count: Optional[int] = None
  ) -> Sequence[vz.TrialSuggestion]:
    return [vz.TrialSuggestion(vz.ParameterDict())] * count

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    self.last_completed = completed.trials
    self.last_active = all_active.trials
    self.num_incorporated_completed_trials += len(completed.trials)
    self._num_incorporated_active_trials += len(all_active.trials)


class _FakeSerializableDesigner(vza.PartiallySerializableDesigner,
                                vza.SerializableDesigner):

  def __init__(self):
    self.num_incorporated_completed_trials = 0
    self.num_incorporated_active_trials = 0
    self.last_completed = []
    self.last_active = []

    # For debugging and testing. Not stored in the metadata.
    self.last_completed = None

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    return [vz.TrialSuggestion(vz.ParameterDict())] * count

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ):
    self.last_completed = completed.trials
    self.last_active = all_active.trials
    self.num_incorporated_completed_trials += len(completed.trials)
    self.num_incorporated_active_trials += len(all_active.trials)

  def dump(self) -> vz.Metadata:
    md = vz.Metadata()
    md.ns('ns1')['foo1'] = 'bar1'
    md.ns('ns1').ns('ns11')['foo11'] = 'bar11'
    md['num_incorporated_completed_trials'] = str(
        self.num_incorporated_completed_trials
    )
    return md

  @classmethod
  def recover(cls, md: vz.Metadata) -> '_FakeSerializableDesigner':
    try:
      designer = cls()
      designer.num_incorporated_completed_trials = int(
          md['num_incorporated_completed_trials']
      )
      return designer
    except KeyError as e:
      raise KeyError(f'Cannot find in {md}') from e

  def load(self, md: vz.Metadata):
    try:
      self.num_incorporated_completed_trials = int(
          md['num_incorporated_completed_trials']
      )
    except KeyError as e:
      raise serializable.DecodeError(f'Cannot find in {md}') from e


_NUM_INITIAL_COMPLETED_TRIALS = 10
_NUM_INITIAL_ACTIVE_TRIALS = 2


def _create_runner() -> pythia.InRamPolicySupporter:
  """Creates the default runner with completed and active trials."""
  runner = pythia.InRamPolicySupporter(vz.ProblemStatement())
  runner.AddTrials(
      [
          vz.Trial().complete(vz.Measurement())
          for _ in range(_NUM_INITIAL_COMPLETED_TRIALS)
      ]
  )
  # The default status is ACTIVE.
  runner.AddTrials([vz.Trial() for _ in range(_NUM_INITIAL_ACTIVE_TRIALS)])
  return runner


class DesignerPolicyNormalOperationTest(absltest.TestCase):
  """Tests Designer policies under error-free conditions."""

  def setUp(self):
    super().setUp()
    self.maxDiff = None

  def test_restore_fully_serializable_designer(self):
    runner = _create_runner()
    policy = dp.SerializableDesignerPolicy(
        problem_statement=vz.ProblemStatement(),
        supporter=runner,
        designer_factory=lambda _, **kwargs: _FakeSerializableDesigner(),
        designer_cls=_FakeSerializableDesigner,
        ns_root='test',
        verbose=2,
    )
    runner.SuggestTrials(policy, 5)
    # Simulate restoring the designer from metadata.
    metadata = policy.dump()
    restored_policy = dp.SerializableDesignerPolicy(
        problem_statement=vz.ProblemStatement(metadata=metadata),
        supporter=runner,
        designer_factory=lambda _, **kwargs: _FakeSerializableDesigner(),
        designer_cls=_FakeSerializableDesigner,
        ns_root='test',
        verbose=2,
    )
    restored_policy.load(metadata)
    designer = restored_policy.designer
    self.assertEqual(
        getattr(designer, 'num_incorporated_completed_trials'),
        _NUM_INITIAL_COMPLETED_TRIALS,
    )

  def test_restore_partially_serializable_designer(self):
    runner = _create_runner()
    policy = dp.PartiallySerializableDesignerPolicy(
        vz.ProblemStatement(),
        runner,
        lambda _, **kwargs: _FakeSerializableDesigner(),
        ns_root='test',
        verbose=2,
    )
    runner.SuggestTrials(policy, 5)
    metadata = policy.dump()
    # Simluate restoring the policy from the metadata.
    restored_policy = dp.PartiallySerializableDesignerPolicy(
        vz.ProblemStatement(),
        runner,
        lambda _, **kwargs: _FakeSerializableDesigner(),
        ns_root='test',
        verbose=2,
    )
    restored_policy.load(metadata)
    self.assertLen(
        restored_policy._cache._incorporated_completed_trial_ids,
        _NUM_INITIAL_COMPLETED_TRIALS,
    )
    self.assertEqual(
        getattr(restored_policy.designer, 'num_incorporated_completed_trials'),
        _NUM_INITIAL_COMPLETED_TRIALS,
    )

  def test_update_stateless_designer(self):
    runner = _create_runner()
    designer = _FakeDesigner()
    policy = dp.DesignerPolicy(runner, lambda _, **kwargs: designer)
    runner.SuggestTrials(policy, 5)
    self.assertLen(
        designer.last_completed,
        _NUM_INITIAL_COMPLETED_TRIALS,
    )
    self.assertLen(
        designer.last_active,
        _NUM_INITIAL_ACTIVE_TRIALS,
    )
    # Add newly completed trials
    runner.AddTrials([vz.Trial().complete(vz.Measurement()) for _ in range(33)])
    runner.SuggestTrials(policy, 15)
    # Check that all completed trials are passed again.
    self.assertLen(
        designer.last_completed,
        _NUM_INITIAL_COMPLETED_TRIALS + 33,
    )
    # Check that all active trials are passed again, including the new ones.
    self.assertLen(
        designer.last_active,
        _NUM_INITIAL_ACTIVE_TRIALS + 5,
    )

  def test_update_stateful_designer(self):
    runner = _create_runner()
    designer = _FakeSerializableDesigner()
    policy = dp.PartiallySerializableDesignerPolicy(
        vz.ProblemStatement(),
        runner,
        lambda _, **kwargs: designer,
        ns_root='test',
        verbose=2,
    )
    runner.SuggestTrials(policy, 11)
    self.assertLen(
        designer.last_completed,
        _NUM_INITIAL_COMPLETED_TRIALS,
    )
    self.assertLen(
        designer.last_active,
        _NUM_INITIAL_ACTIVE_TRIALS,
    )
    # Add newly completed trials
    runner.AddTrials([vz.Trial().complete(vz.Measurement()) for _ in range(33)])
    runner.SuggestTrials(policy, 5)
    # Check that completed trials are not passed again.
    self.assertLen(designer.last_completed, 33)
    # Check that the newly active trials are passed.
    self.assertLen(designer.last_active, _NUM_INITIAL_ACTIVE_TRIALS + 11)


if __name__ == '__main__':
  absltest.main()
