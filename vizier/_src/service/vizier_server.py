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

"""Classes for starting the Vizier Server.

Extensive tests can be found in `clients_test.py`.
"""

from concurrent import futures
import datetime
import time

import attr
import grpc
import portpicker
from vizier import pythia
from vizier._src.service import constants
from vizier._src.service import datastore
from vizier._src.service import policy_factory as service_policy_factory_lib
from vizier._src.service import pythia_service
from vizier._src.service import pythia_service_pb2_grpc
from vizier._src.service import stubs_util
from vizier._src.service import vizier_service
from vizier._src.service import vizier_service_pb2_grpc


@attr.define
class DefaultVizierServer:
  """Vizier Server which runs Pythia and Vizier Servicers in the same process.

  Both servicers have access to the others' literal class instances.

  NOTE: When using this in a test, the database_url should be in-memory
  (SQL_MEMORY_URL) since tests don't easily allow arbitrarily filepaths.
  """

  _host: str = attr.field(init=True, default='localhost')
  _database_url: str = attr.field(
      init=True, default=constants.SQL_LOCAL_URL, kw_only=True
  )
  _policy_factory: pythia.PolicyFactory = attr.field(
      init=True,
      factory=service_policy_factory_lib.DefaultPolicyFactory,
      kw_only=True,
  )
  _early_stop_recycle_period: datetime.timedelta = attr.field(
      init=True, default=datetime.timedelta(seconds=0.1), kw_only=True
  )
  _port: int = attr.field(init=False, factory=portpicker.pick_unused_port)
  _servicer: vizier_service.VizierServicer = attr.field(init=False)
  _server: grpc.Server = attr.field(init=False)
  stub: vizier_service_pb2_grpc.VizierServiceStub = attr.field(init=False)

  @property
  def datastore(self) -> datastore.DataStore:
    return self._servicer.datastore

  @property
  def endpoint(self) -> str:
    return f'{self._host}:{self._port}'

  def __attrs_post_init__(self):
    # Setup Vizier server.
    self._servicer = vizier_service.VizierServicer(
        database_url=self._database_url,
        early_stop_recycle_period=self._early_stop_recycle_period,
    )
    self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=30))
    vizier_service_pb2_grpc.add_VizierServiceServicer_to_server(
        self._servicer, self._server
    )
    self._server.add_insecure_port(self.endpoint)
    self._server.start()
    self.stub = stubs_util.create_vizier_server_stub(self.endpoint)

    # Re-set the default Pythia Service to allow custom policy factories.
    default_pythia_service = pythia_service.PythiaServicer(
        self._servicer, policy_factory=self._policy_factory
    )
    self._servicer.default_pythia_service = default_pythia_service

  def wait_for_early_stop_recycle_period(self) -> None:
    time.sleep(self._early_stop_recycle_period.total_seconds())


@attr.define
class DistributedPythiaVizierServer(DefaultVizierServer):
  """Separates Pythia from Vizier via over-the-wire distributed communication.

  This is for testing / demonstration purposes only, as in normal use-cases, the
  Pythia server should actually be created in a separate process from the Vizier
  server.
  """

  _pythia_port: int = attr.field(
      init=False, factory=portpicker.pick_unused_port
  )
  _pythia_servicer: pythia_service.PythiaServicer = attr.field(init=False)
  _pythia_server: grpc.Server = attr.field(init=False)
  pythia_stub: pythia_service_pb2_grpc.PythiaServiceStub = attr.field(
      init=False
  )

  @property
  def pythia_endpoint(self) -> str:
    return f'{self._host}:{self._pythia_port}'

  def __attrs_post_init__(self):
    super().__attrs_post_init__()
    # Setup Pythia server.
    self._pythia_servicer = pythia_service.PythiaServicer(
        policy_factory=self._policy_factory
    )
    # `max_workers=1` is used since we can only run one Pythia thread at a time.
    self._pythia_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pythia_service_pb2_grpc.add_PythiaServiceServicer_to_server(
        self._pythia_servicer, self._pythia_server
    )
    self._pythia_server.add_insecure_port(self.pythia_endpoint)
    self._pythia_server.start()
    self.pythia_stub = stubs_util.create_pythia_server_stub(
        self.pythia_endpoint
    )

    # Connect Vizier and Pythia servers together.
    self._servicer.default_pythia_service = self.pythia_stub
    self._pythia_servicer.connect_to_vizier(self.endpoint)
