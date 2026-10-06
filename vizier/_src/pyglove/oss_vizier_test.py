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

"""Tests for pyglove.tuner.vizier2.oss_vizier_test."""
import datetime
import os

from absl import logging
from vizier._src.pyglove import oss_vizier as vizier
from vizier._src.pyglove import vizier_test_lib
from vizier._src.service import constants
from vizier._src.service import vizier_server

from absl.testing import absltest


class OSSVizierSampleTest(vizier_test_lib.SampleTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    server = vizier_server.DefaultVizierServer(
        host=os.uname()[1],
        database_url=constants.SQL_MEMORY_URL,
        early_stop_recycle_period=datetime.timedelta(seconds=0.0),
    )
    logging.info(server.endpoint)
    vizier._services.reset_for_testing()
    vizier.init(vizier_endpoint=server.endpoint)
    cls.server = server
    logging.info('Vizier service has been set up!')

  def __init__(self, *args, **kwargs):
    super().__init__(
        vizier.OSSVizierBackend,
        *args,
        builtin_multiobjective_algorithm_to_test='NSGA2',
        **kwargs,
    )


if __name__ == '__main__':
  absltest.main()
