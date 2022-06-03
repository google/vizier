"""Large-scale stress tests (multiple clients, mulithreading, etc.) for Vizier Service."""
from concurrent import futures
import datetime
import multiprocessing.pool
import time
from absl import logging
import grpc
import portpicker

from vizier import benchmarks
from vizier.service import pyvizier
from vizier.service import vizier_client
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc

from absl.testing import absltest
from absl.testing import parameterized


class PerformanceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Setup Vizier Service.
    self.early_stop_recycle_period = datetime.timedelta(seconds=1)
    self.servicer = vizier_server.VizierService(
        early_stop_recycle_period=self.early_stop_recycle_period)

    # Setup local networking.
    self.port = portpicker.pick_unused_port()
    self.address = f'localhost:{self.port}'

    # Setup server.
    self.server = grpc.server(futures.ThreadPoolExecutor())
    vizier_service_pb2_grpc.add_VizierServiceServicer_to_server(
        self.servicer, self.server)
    self.server.add_secure_port(self.address, grpc.local_server_credentials())
    self.server.start()

  @parameterized.parameters(
      (1, 10, 2),
      (10, 10, 2),
      (50, 10, 2),
      (100, 10, 2),
  )
  def test_multiple_clients_basic(self, num_simultaneous_clients,
                                  num_trials_per_client, dimension):

    def fn(client_id: int):
      func = benchmarks.bbob.Sphere
      experimenter = benchmarks.NumpyExperimenter(
          func, benchmarks.bbob.DefaultBBOBProblemStatement(dimension))
      problem_statement = experimenter.problem_statement()
      study_config = pyvizier.StudyConfig.from_problem(problem_statement)
      study_config.algorithm = pyvizier.Algorithm.RANDOM_SEARCH

      client = vizier_client.create_or_load_study(
          service_endpoint=self.address,
          owner_id='my_username',
          study_display_name='study_name',
          study_config=study_config,
          client_id=str(client_id))

      for _ in range(num_trials_per_client):
        suggestions = client.get_suggestions(suggestion_count=1)
        completed_trials = experimenter.evaluate(suggestions)
        for completed_trial in completed_trials:
          client.complete_trial(completed_trial.id,
                                completed_trial.final_measurement)

      return client

    client_ids = range(num_simultaneous_clients)
    pool = multiprocessing.pool.ThreadPool(num_simultaneous_clients)

    start = time.time()
    clients = pool.map(fn, client_ids)
    end = time.time()
    pool.close()

    study_name = clients[0].study_name
    self.assertEqual(
        self.servicer.datastore.max_trial_id(study_name),
        num_simultaneous_clients * num_trials_per_client)

    logging.info(
        'For %d clients to evaluate %d trials each, it took %f seconds total.',
        num_simultaneous_clients, num_trials_per_client, end - start)


if __name__ == '__main__':
  absltest.main()
