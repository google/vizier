"""Utility functions for creating GRPC stubs."""

import functools
from absl import logging
import grpc

from vizier.service import pythia_service_pb2_grpc
from vizier.service import vizier_service_pb2_grpc


def _create_channel(endpoint: str) -> grpc.Channel:
  """Creates GRPC channel."""
  logging.info('Securing channel to %s.', endpoint)
  channel = grpc.secure_channel(endpoint, grpc.local_channel_credentials())
  grpc.channel_ready_future(channel).result()
  logging.info('Secured channel to %s.', endpoint)
  return channel


@functools.lru_cache
def create_pythia_server_stub(
    endpoint: str) -> pythia_service_pb2_grpc.PythiaServiceStub:
  """Creates the GRPC stub.

  This method uses LRU cache so we create a single stub per endpoint (which is
  effectively one per binary). Stub and channel are both thread-safe and can
  take a while to create. The LRU cache makes binaries run faster, especially
  for unit tests.

  Args:
    endpoint:

  Returns:
    Pythia service stub at service_endpoint.
  """
  return pythia_service_pb2_grpc.PythiaServiceStub(_create_channel(endpoint))


@functools.lru_cache
def create_vizier_server_stub(
    endpoint: str) -> vizier_service_pb2_grpc.VizierServiceStub:
  """Creates the GRPC stub.

  This method uses LRU cache so we create a single stub per endpoint (which is
  effectively one per binary). Stub and channel are both thread-safe and can
  take a while to create. The LRU cache makes binaries run faster, especially
  for unit tests.

  Args:
    endpoint:

  Returns:
    Vizier service stub at service_endpoint.
  """
  return vizier_service_pb2_grpc.VizierServiceStub(_create_channel(endpoint))
