"""Tests for local_service."""

from vizier.service.testing import local_service
from absl.testing import absltest


class LocalServiceTest(absltest.TestCase):

  def test_creation(self):
    service = local_service.LocalVizierTestService()
    self.assertIsNotNone(service.stub)
    self.assertIsNotNone(service.endpoint)
    self.assertIsNotNone(service.datastore)


if __name__ == '__main__':
  absltest.main()
