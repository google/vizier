"""Utility functions for the Pythia protocol code."""

import logging
import threading
from typing import Any, Generic, Optional, TypeVar
from vizier import pythia
from vizier import pyvizier
from vizier._src.pyvizier.shared import common
from google.protobuf import any_pb2

_TT = TypeVar('_TT')


class ResponseWaiter(Generic[_TT]):
  """A stored message with a Wait() mechanism.

  This is a bridge between two threads; it stores a message, and
  the reader thread waits until the writer thread calls Report(..., done=True).
  Once Report() is called, it unblocks WaitForRespose().
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._response: Optional[_TT] = None
    self._wait = threading.Event()

  def Report(self, update: _TT):
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

  def WaitForResponse(self) -> _TT:
    """Returns the result or raises an error.

    Raises:
      PythiaProtocolError: Due to a failure of the Pythia protocol.
      VizierDatabaseError: Vizier was unable to service the request.
    """
    self._wait.wait()
    logging.info('About to take _lock in ResponseWaiter.WaitForResponse()')
    with self._lock:
      if not self._response:
        logging.info('Raise')
        raise pythia.PythiaProtocolError('No response.')
      if self._response.error_details:
        raise pythia.VizierDatabaseError(self._response.error_details)
      logging.info('No raise -- WaitForResponse done')
      return self._response

  def CancelWait(self) -> None:
    """Cancel the wait."""
    logging.info('About to CancelWait()')
    self._wait.set()


def AssignKeyValuePlus(container: Any, *, trial_id: Optional[int], key: str,
                       ns: common.Namespace,
                       value: pyvizier.MetadataValue) -> None:
  """Insert or assign (key, value) to container.metadata.

  Args:
    container: A container class (e.g. a protobuf) that contains a $metadata
      field.  The $metadata field should be type KeyValuePlus.
    trial_id: To become container.metadata[i].trial_id; when unset, it means
      Study-wide metadata not associated with a Trial.
    key: To become container.metadata[i].key.
    ns: To become container.metadata[i].ns.
    value: To become either the $value or the $proto field of
      container.metadata[i], depending on the type of $value.   If $value is a
      string, it is inserted into ....value; else if it is a protobuf, it is
      packed into ....proto.
  """
  ns: str = ns.encode()
  str_trial_id: Optional[str] = str(trial_id) if trial_id is not None else None
  # The key is already in the metadata.
  for kv in container.metadata:
    trial_match = (not kv.HasField('trial_id')
                   if trial_id is None else kv.trial_id == str_trial_id)
    if trial_match and kv.k_v.key == key and kv.k_v.ns == ns:
      if isinstance(value, str):
        kv.ClearField('proto')
        kv.value = value
      elif isinstance(value, any_pb2.Any):
        kv.ClearField('value')
        kv.proto.CopyFrom(value)
      else:
        kv.ClearField('value')
        kv.proto.Pack(value)
      return

  # The key does not yet exist in the metadata.
  new_item = container.metadata.add(trial_id=str_trial_id)
  new_kv = new_item.k_v
  new_kv.key = key
  new_kv.ns = ns
  if isinstance(value, str):
    new_kv.value = value
  elif isinstance(value, any_pb2.Any):
    new_kv.proto = value
  else:
    new_kv.proto.Pack(value)
