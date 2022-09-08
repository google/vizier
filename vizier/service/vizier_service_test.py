"""Test for vizier_service."""

from vizier.service import vizier_service
from absl.testing import absltest


class LocalServiceTest(absltest.TestCase):

  def test_creation(self):
    service = vizier_service.DefaultVizierService()
    self.assertIsNotNone(service.stub)
    self.assertIsNotNone(service.endpoint)
    self.assertIsNotNone(service.datastore)


if __name__ == '__main__':
  absltest.main()
