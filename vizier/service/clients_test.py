"""Tests for clients."""

from absl import flags
from absl import logging
import grpc
from vizier.client import client_abc_testing
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import vizier_service

from absl.testing import absltest

FLAGS = flags.FLAGS


class VizierClientTest(client_abc_testing.TestCase):
  _service: vizier_service.DefaultVizierService
  _owner: str
  _channel: grpc.Channel

  @classmethod
  def setUpClass(cls):
    logging.info('Test setup started.')
    super().setUpClass()
    cls._service = vizier_service.DefaultVizierService()
    clients.environment_variables.service_endpoint = cls._service.endpoint
    logging.info('Test setup finished.')

  def create_study(self, problem: vz.ProblemStatement,
                   study_id: str) -> clients.Study:
    config = vz.StudyConfig.from_problem(problem)
    config.algorithm = vz.Algorithm.RANDOM_SEARCH
    study = clients.Study.from_study_config(
        config, owner='owner', study_id=study_id)
    return study

  def test_e2e_tuning(self):
    self.assertPassesE2ETuning()

  @classmethod
  def tearDownClass(cls):
    cls._service._server.stop(None)
    super().tearDownClass()


if __name__ == '__main__':
  absltest.main()
