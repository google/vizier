"""Tests for designer_policy."""

import copy

from typing import Optional, Sequence
from vizier import algorithms as vza
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.algorithms.policies import designer_policy as dp
from vizier.interfaces import serializable
from absl.testing import absltest


class _FakeSerializableDesigner(vza.PartiallySerializableDesigner,
                                vza.SerializableDesigner):
  # If set, load() and recover() raise the provided Exception.
  _error_on_load: Optional[Exception] = None

  def __init__(self):
    self._num_incorporated_trials = 0

    # For debugging and testing. Not stored in the metadata.
    self._last_delta = None

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    return [vz.TrialSuggestion(vz.ParameterDict())] * count

  def update(self, delta: vza.CompletedTrials):
    self._last_delta = delta
    self._num_incorporated_trials += len(delta.completed)

  def dump(self) -> vz.Metadata:
    md = vz.Metadata()
    md.ns('ns1')['foo1'] = 'bar1'
    md.ns('ns1').ns('ns11')['foo11'] = 'bar11'
    md['num_incorporated_trials'] = str(self._num_incorporated_trials)
    return md

  @classmethod
  def recover(cls, md: vz.Metadata) -> '_FakeSerializableDesigner':
    if cls._error_on_load is not None:
      raise cls._error_on_load  # pylint:disable=raising-bad-type
    try:
      designer = cls()
      designer._num_incorporated_trials = int(md['num_incorporated_trials'])
      return designer
    except KeyError as e:
      raise KeyError(f'Cannot find in {md}') from e

  def load(self, md: vz.Metadata):
    if self._error_on_load is not None:
      raise self._error_on_load  # pylint:disable=raising-bad-type
    try:
      self._num_incorporated_trials = int(md['num_incorporated_trials'])
    except KeyError as e:
      raise serializable.DecodeError(f'Cannot find in {md}') from e


# TODO: Add more tests

_NUM_INITIAL_COMPLETED_TRIALS = 10


class DesignerPolicyNormalOperationTest(absltest.TestCase):
  """Tests Designer policies under error-free conditions."""

  def setUp(self):
    super().setUp()
    self.maxDiff = None  # pylint: disable=invalid-name
    designer = _FakeSerializableDesigner()
    runner = pythia.LocalPolicyRunner(vz.StudyConfig())
    runner.AddTrials([
        vz.Trial().complete(vz.Measurement())
        for _ in range(_NUM_INITIAL_COMPLETED_TRIALS)
    ])

    # Run with a policy
    policy = dp.PartiallySerializableDesignerPolicy(
        runner, lambda _: designer, ns_root='test', verbose=2)
    trials = runner.SuggestTrials(policy, 5)
    self.assertLen(designer._last_delta.completed,
                   _NUM_INITIAL_COMPLETED_TRIALS)
    self.assertEqual(
        runner.study_descriptor().config.metadata.ns('test').ns('designer')
        ['num_incorporated_trials'], str(_NUM_INITIAL_COMPLETED_TRIALS))

    # Complete trials
    for t in trials[::2]:
      t.complete(vz.Measurement())

    self.runner = runner
    self.designer = designer
    self.trials = trials

  def test_partially_serializable(self):
    runner, designer, trials = self.runner, self.designer, self.trials

    # Mimick the server environment by creating a new policy.
    policy = dp.PartiallySerializableDesignerPolicy(
        runner, lambda _: designer, ns_root='test', verbose=2)
    runner.SuggestTrials(policy, 1)
    # Delta should consist only of the newly completed trials.
    self.assertSequenceEqual(designer._last_delta.completed, trials[::2])
    self.assertEqual(designer._num_incorporated_trials, 13)
    self.assertEqual(
        runner.study_descriptor().config.metadata.ns('test').ns('designer')
        ['num_incorporated_trials'], str(13))

  def test_fully_serializable(self):
    runner, _, trials = self.runner, self.designer, self.trials

    # Mimick the server environment by creating a new policy.
    def raise_error(*args, **kwargs):
      raise ValueError('This code should not be called')

    policy = dp.SerializableDesignerPolicy(
        runner,
        designer_factory=raise_error,  # should not be used.
        designer_cls=_FakeSerializableDesigner,
        ns_root='test',
        verbose=2)
    runner.SuggestTrials(policy, 1)

    # `policy` creates a new designer object by loading from metadata.
    # Delta should consist only of the newly completed trials.
    new_designer = policy.designer
    self.assertSequenceEqual(new_designer._last_delta.completed, trials[::2])
    self.assertEqual(new_designer._num_incorporated_trials, 13)
    self.assertEqual(
        runner.study_descriptor().config.metadata.ns('test').ns('designer')
        ['num_incorporated_trials'], str(13))

  def test_non_serializable(self):
    runner, designer, trials = self.runner, self.designer, self.trials

    # Mimick the server environment by creating a new policy.
    policy = dp.DesignerPolicy(copy.deepcopy(runner), lambda _: designer)
    runner.SuggestTrials(policy, 1)
    # Designer is updated with all trials, including both the initial batch
    # of completed trials and the newly completed trials.
    self.assertLen(designer._last_delta.completed,
                   _NUM_INITIAL_COMPLETED_TRIALS + len(trials[::2]))


if __name__ == '__main__':
  absltest.main()
