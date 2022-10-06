# Copyright 2022 Google LLC.
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
