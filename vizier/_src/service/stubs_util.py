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

"""Utility functions for creating GRPC stubs."""

import functools
from typing import Optional
from absl import logging
import grpc

from vizier._src.service import pythia_service_pb2_grpc
from vizier._src.service import vizier_service_pb2_grpc


def _create_channel(
    endpoint: str, timeout: Optional[float] = None
) -> grpc.Channel:
  """Creates GRPC channel."""
  logging.info('Securing channel to %s.', endpoint)
  channel = grpc.insecure_channel(endpoint)
  grpc.channel_ready_future(channel).result(timeout=timeout)
  logging.info('Created channel to %s.', endpoint)
  return channel


@functools.lru_cache(maxsize=128)
def create_pythia_server_stub(
    endpoint: str, timeout: Optional[float] = 10.0
) -> pythia_service_pb2_grpc.PythiaServiceStub:
  """Creates the GRPC stub.

  This method uses LRU cache so we create a single stub per endpoint (which is
  effectively one per binary). Stub and channel are both thread-safe and can
  take a while to create. The LRU cache makes binaries run faster, especially
  for unit tests.

  Args:
    endpoint: Pythia server endpoint.
    timeout: Timeout in seconds. If None, no timeout will be used.

  Returns:
    Pythia Server stub at endpoint.
  """
  return pythia_service_pb2_grpc.PythiaServiceStub(
      _create_channel(endpoint, timeout)
  )


@functools.lru_cache(maxsize=128)
def create_vizier_server_stub(
    endpoint: str, timeout: Optional[float] = 10.0
) -> vizier_service_pb2_grpc.VizierServiceStub:
  """Creates the GRPC stub.

  This method uses LRU cache so we create a single stub per endpoint (which is
  effectively one per binary). Stub and channel are both thread-safe and can
  take a while to create. The LRU cache makes binaries run faster, especially
  for unit tests.

  Args:
    endpoint: Vizier server endpoint.
    timeout: Timeout in seconds. If None, no timeout will be used.

  Returns:
    Vizier Server stub at endpoint.
  """
  return vizier_service_pb2_grpc.VizierServiceStub(
      _create_channel(endpoint, timeout)
  )
