"""Sets up the Vizier Service. This should be done on a server machine."""

from concurrent import futures
import socket
import time
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import grpc
import portpicker

from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc

flags.DEFINE_string(
    'host', socket.gethostname(),
    'Host location for the server. For distributed cases, use the IP address.')

FLAGS = flags.FLAGS

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Setup local networking.
  port = portpicker.pick_unused_port()
  address = f'{FLAGS.host}:{port}'
  logging.info('Running Vizier server on: %s', address)

  # Setup server.
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

  # Setup Vizier Service.
  servicer = vizier_server.VizierService()
  vizier_service_pb2_grpc.add_VizierServiceServicer_to_server(servicer, server)
  server.add_secure_port(address, grpc.local_server_credentials())
  server.start()

  # prevent the main thread from exiting
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    server.stop(0)


if __name__ == '__main__':
  app.run(main)
