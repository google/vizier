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

"""Utility functions for the Pythia protocol code."""

import logging
import threading
from typing import Generic, Optional, Protocol, TypeVar
from vizier import pythia


class _HasErrorDetails(Protocol):
  error_details: str


_T = TypeVar('_T', bound=_HasErrorDetails)


class ResponseWaiter(Generic[_T]):
  """A stored message with a Wait() mechanism.

  This is a bridge between two threads; it stores a message, and
  the reader thread waits until the writer thread calls Report(..., done=True).
  Once Report() is called, it unblocks WaitForRespose().
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._response: Optional[_T] = None
    self._wait = threading.Event()

  def Report(self, update: _T) -> None:
    """Called by the gRPC thread with a message from Vizier.

    When called, this unblocks WaitForResponse().

    Args:
      update:

    Raises:
      PythiaProtocolError
    """
    logging.info('About to take _lock in ResponseWaiter.Report()')
    with self._lock:
      if self._wait.is_set():
        raise pythia.PythiaProtocolError(
            'ResponseWaiter.Report() called after wait is set')
      self._response = update
      self._wait.set()

  def WaitForResponse(self) -> _T:
    """Returns the result or raises an error.

    Raises:
      PythiaProtocolError: Due to a failure of the Pythia protocol.
      VizierDatabaseError: Vizier was unable to service the request.
    """
    self._wait.wait()
    logging.info('About to take _lock in ResponseWaiter.WaitForResponse()')
    with self._lock:
      if self._response is None:
        logging.info('Raise')
        raise pythia.PythiaProtocolError('No response.')
      elif self._response.error_details:
        raise pythia.VizierDatabaseError(self._response.error_details)
      logging.info('No raise -- WaitForResponse done')
      return self._response

  def CancelWait(self) -> None:
    """Cancel the wait."""
    logging.info('About to CancelWait()')
    self._wait.set()
