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

"""Tests for clients."""
import functools

from absl import logging
from vizier._src.service import clients
from vizier._src.service import constants
from vizier._src.service import vizier_server
from vizier.client import client_abc_testing
from vizier.service import pyvizier as vz

from absl.testing import absltest


# Affects local Vizier servicer tests only.
clients.environment_variables.servicer_use_sql_ram()


class VizierClientTest(client_abc_testing.TestCase):
  _owner: str

  def create_study(
      self, problem: vz.ProblemStatement, study_id: str
  ) -> clients.Study:
    config = vz.StudyConfig.from_problem(problem)
    config.algorithm = vz.Algorithm.RANDOM_SEARCH
    study = clients.Study.from_study_config(
        config, owner='owner', study_id=study_id
    )
    return study

  def test_e2e_tuning(self):
    self.assertPassesE2ETuning()

  def test_e2e_tuning_gp(self):
    def create_study(
        problem: vz.ProblemStatement, study_id: str
    ) -> clients.Study:
      config = vz.StudyConfig.from_problem(problem)
      config.algorithm = vz.Algorithm.GAUSSIAN_PROCESS_BANDIT
      study = clients.Study.from_study_config(
          config, owner='owner', study_id=study_id
      )
      return study

    study_factory = functools.partial(create_study, study_id=self.id())
    self.assertPassesE2ETuning(study_factory=study_factory, num_iterations=2)


class VizierClientTestOnServicer(VizierClientTest):

  @classmethod
  def setUpClass(cls):
    logging.info('Test setup started.')
    super().setUpClass()
    clients.environment_variables.server_endpoint = constants.NO_ENDPOINT
    logging.info('Test setup finished.')


class VizierClientTestOnDefaultServer(VizierClientTest):
  _server: vizier_server.DefaultVizierServer

  @classmethod
  def setUpClass(cls):
    logging.info('Test setup started.')
    super().setUpClass()
    cls._server = vizier_server.DefaultVizierServer(
        database_url=constants.SQL_MEMORY_URL
    )
    clients.environment_variables.server_endpoint = cls._server.endpoint
    logging.info('Test setup finished.')

  @classmethod
  def tearDownClass(cls):
    cls._server._server.stop(None)
    super().tearDownClass()


class VizierClientTestOnDistributedPythiaServer(VizierClientTest):
  _server: vizier_server.DistributedPythiaVizierServer

  @classmethod
  def setUpClass(cls):
    logging.info('Test setup started.')
    super().setUpClass()
    cls._server = vizier_server.DistributedPythiaVizierServer(
        database_url=constants.SQL_MEMORY_URL
    )
    clients.environment_variables.server_endpoint = cls._server.endpoint
    logging.info('Test setup finished.')

  @classmethod
  def tearDownClass(cls):
    cls._server._server.stop(None)
    cls._server._pythia_server.stop(None)
    super().tearDownClass()


if __name__ == '__main__':
  absltest.main()
