"""Tests for vizier.pythia.base.policy."""

from vizier.pythia.base import policy
from vizier.pyvizier import pythia as vz

from absl.testing import absltest


class PolicyTest(absltest.TestCase):
  """Test the creation of attrib classes."""

  def test_stoping_decision(self):
    policy.EarlyStopDecision(
        50,
        'just because',
        metadata=vz.Metadata({'key': 'value'}),
        predicted_final_measurement=vz.Measurement())
    policy.EarlyStopDecision(50, 'just because')

  def test_stop_request(self):
    policy.EarlyStopRequest(
        vz.StudyDescriptor(config=vz.StudyConfig()), (5, 10, 12), 'checkpoint')

  def test_suggest_decision(self):
    self.assertLen(policy.SuggestDecisions.from_trials([vz.Trial()] * 2), 2)

  def test_suggest_request(self):
    policy.SuggestRequest(
        vz.StudyDescriptor(config=vz.StudyConfig()), 10, 'checkpoint')


if __name__ == '__main__':
  absltest.main()
