# Copyright 2023 Google LLC.
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

"""Large-scale stress tests (multiple workers, multithreading) for PyGlove-Vizier integration."""
import multiprocessing.pool
import os
import random
import time
from absl import logging
import pyglove as pg

from vizier._src.pyglove import oss_vizier as vizier
from vizier.service import vizier_server

from absl.testing import absltest
from absl.testing import parameterized


NUM_TRIALS_PER_WORKER = 10
NUM_WORKERS = 10

_server = None


def setUpModule():
  hostname = os.uname()[1]
  global _server
  _server = vizier_server.DefaultVizierServer(host=hostname)
  logging.info(_server.endpoint)
  vizier._global_states.vizier_tuner = None
  vizier.init(vizier_endpoint=_server.endpoint)
  logging.info('Setupmodule done!')


class PerformanceTest(parameterized.TestCase):

  @parameterized.parameters(
      (multiprocessing.pool.ThreadPool,),
      # (multiprocessing.Pool,),  # Fails currently.
  )
  def test_multiple_workers(self, pool_creator):
    def work_fn(worker_id: int):
      del worker_id
      algorithm = pg.evolution.regularized_evolution()
      for _, feedback in pg.sample(
          hyper_value=pg.Dict(x=pg.oneof([1, 2, 3])),
          algorithm=algorithm,
          num_examples=NUM_TRIALS_PER_WORKER,
          name='performance_testing',
      ):
        feedback(reward=random.random())

    with pool_creator(NUM_WORKERS) as pool:
      start = time.time()
      pool.map(work_fn, range(NUM_WORKERS))
      end = time.time()

    logging.info(
        'For %d workers to evaluate %d trials each, it took %f seconds total.',
        NUM_WORKERS,
        NUM_TRIALS_PER_WORKER,
        end - start,
    )


if __name__ == '__main__':
  absltest.main()
