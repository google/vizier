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
import random
import time
from absl import logging
import pyglove as pg

from vizier._src.pyglove import oss_vizier as vizier

from absl.testing import absltest
from absl.testing import parameterized

vizier.init()


class PerformanceTest(parameterized.TestCase):

  @parameterized.parameters(
      (1, 10),
      (2, 10),
      (10, 10),
  )
  def test_multithreaded_workers(
      self, num_simultaneous_workers, num_trials_per_worker
  ):
    def thread_fn(client_id: int):
      del client_id
      algorithm = pg.evolution.regularized_evolution()
      for _, feedback in pg.sample(
          hyper_value=pg.Dict(x=pg.oneof([1, 2, 3])),
          algorithm=algorithm,
          num_examples=num_simultaneous_workers,
          name='multithread_testing',
      ):
        feedback(reward=random.random())

    pool = multiprocessing.pool.ThreadPool(num_simultaneous_workers)

    start = time.time()
    pool.map(thread_fn, range(num_simultaneous_workers))
    end = time.time()
    pool.close()

    logging.info(
        'For %d workers to evaluate %d trials each, it took %f seconds total.',
        num_simultaneous_workers,
        num_trials_per_worker,
        end - start,
    )


if __name__ == '__main__':
  absltest.main()
