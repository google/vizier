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

"""Large-scale stress tests (multiple clients, mulithreading, etc.) for Vizier Service."""

import multiprocessing.pool
import time
from absl import logging

from vizier._src.service import constants
from vizier._src.service import vizier_client
from vizier._src.service import vizier_server
from vizier.benchmarks import experimenters
from vizier.service import pyvizier

from absl.testing import absltest
from absl.testing import parameterized


class PerformanceTest(parameterized.TestCase):
  server: vizier_server.DefaultVizierServer

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.server = vizier_server.DefaultVizierServer(
        database_url=constants.SQL_MEMORY_URL
    )
    vizier_client.environment_variables.server_endpoint = cls.server.endpoint

  @parameterized.parameters(
      (1, 10, 2),
      (2, 10, 2),
      (10, 10, 2),
      (50, 5, 2),
      (100, 5, 2),
  )
  def test_multiple_clients_basic(
      self, num_simultaneous_clients, num_trials_per_client, dimension
  ):
    def fn(client_id: int):
      experimenter = experimenters.BBOBExperimenterFactory(
          'Sphere', dimension
      )()
      problem_statement = experimenter.problem_statement()
      study_config = pyvizier.StudyConfig.from_problem(problem_statement)
      study_config.algorithm = pyvizier.Algorithm.NSGA2

      client = vizier_client.create_or_load_study(
          owner_id='my_username',
          study_id=self.id(),  # Use the testcase name.
          study_config=study_config,
          client_id=str(client_id),
      )

      for _ in range(num_trials_per_client):
        suggestions = client.get_suggestions(suggestion_count=1)
        experimenter.evaluate(suggestions)
        for completed_trial in suggestions:
          client.complete_trial(
              completed_trial.id, completed_trial.final_measurement
          )

      return client

    client_ids = range(num_simultaneous_clients)
    pool = multiprocessing.pool.ThreadPool(num_simultaneous_clients)

    start = time.time()
    clients = pool.map(fn, client_ids)
    end = time.time()
    pool.close()

    self.assertEqual(
        max({t.id for t in clients[0].list_trials()}),
        num_simultaneous_clients * num_trials_per_client,
    )

    logging.info(
        'For %d clients to evaluate %d trials each, it took %f seconds total.',
        num_simultaneous_clients,
        num_trials_per_client,
        end - start,
    )


if __name__ == '__main__':
  absltest.main()
