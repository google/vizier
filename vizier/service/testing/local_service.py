"""Vizier Service with in-RAM data storage."""

from concurrent import futures
import datetime
import time

import attr
import grpc
import portpicker
from vizier.service import datastore
from vizier.service import vizier_client
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc


@attr.define
class LocalVizierTestService:
  """Local Vizier service with InRAM data storage."""
  _early_stop_recycle_period: datetime.timedelta = attr.field(
      init=False, default=datetime.timedelta(seconds=1))
  _port = attr.field(init=False, factory=portpicker.pick_unused_port)
  _servicer: vizier_server.VizierService = attr.field(init=False)
  _server: grpc.Server = attr.field(init=False)
  stub: vizier_service_pb2_grpc.VizierServiceStub = attr.field(init=False)

  @property
  def datastore(self) -> datastore.DataStore:
    return self._servicer.datastore

  @property
  def endpoint(self) -> str:
    return f'localhost:{self._port}'

  def __init__(self):
    self.__attrs_init__()
    self._servicer = vizier_server.VizierService(
        early_stop_recycle_period=self._early_stop_recycle_period)
    # Setup server.
    self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=30))

    vizier_service_pb2_grpc.add_VizierServiceServicer_to_server(
        self._servicer, self._server)
    self._server.add_secure_port(self.endpoint, grpc.local_server_credentials())
    self._server.start()
    self.stub = vizier_client.create_server_stub(self.endpoint)

  def wait_for_early_stop_recycle_period(self) -> None:
    time.sleep(self._early_stop_recycle_period.total_seconds())
