"""Tests for vizier.pyvizier.shared.context."""

from vizier._src.pyvizier.shared import context
from absl.testing import absltest


class ContextTest(absltest.TestCase):

  def testDefaultsNotShared(self):
    """Make sure default parameters are not shared between instances."""
    context1 = context.Context()
    context2 = context.Context()
    context1.parameters['x1'] = context.ParameterValue(5)
    self.assertEmpty(context2.parameters)


if __name__ == '__main__':
  absltest.main()
