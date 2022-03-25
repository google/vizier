"""Utility functions for handling vizier metadata."""

from typing import Tuple, Union, Optional, TypeVar, Type
from vizier.service import key_value_pb2
from vizier.service import study_pb2
from google.protobuf import any_pb2
from google.protobuf.message import Message

T = TypeVar('T')


def assign(
    container: Union[study_pb2.StudySpec, study_pb2.Trial], *, key: str,
    ns: str, value: Union[str, any_pb2.Any,
                          Message]) -> Tuple[key_value_pb2.KeyValue, bool]:
  """Insert or assign (key, value) to container.metadata.

  Args:
    container: container.metadata must be repeated KeyValue (protobuf) field.
    key:
    ns: A namespace for the key (defaults to '', which is the user's namespace).
    value: Behavior depends on the type. `str` is copied to KeyValue.value
      `any_pb2.Any` is copied to KeyValue.proto Other types are packed to
      any_pb2.Any proto, which is then copied to KeyValue.proto.

  Returns:
    (proto, inserted) where
    proto is the protobuf that was just inserted into the $container, and
    inserted is True if the proto was newly inserted, False if it was replaced.
  """

  for kv in container.metadata:
    if kv.key == key and kv.ns == ns:
      if isinstance(value, str):
        kv.ClearField('proto')
        kv.value = value
      elif isinstance(value, any_pb2.Any):
        kv.ClearField('value')
        kv.proto.CopyFrom(value)
      else:
        kv.ClearField('value')
        kv.proto.Pack(value)
      return kv, False

  # The key does not exist in the metadata.
  if isinstance(value, str):
    metadata = container.metadata.add(key=key, ns=ns, value=value)
  elif isinstance(value, any_pb2.Any):
    metadata = container.metadata.add(key=key, ns=ns, proto=value)
  else:
    metadata = container.metadata.add(key=key, ns=ns)
    metadata.proto.Pack(value)
  return metadata, True


def get(container: Union[study_pb2.StudySpec, study_pb2.Trial], *, key: str,
        ns: str) -> Optional[str]:
  """Returns the metadata value associated with key, or None.

  Args:
    container: A Trial of a StudySpec in protobuf form.
    key: The key of a KeyValue protobuf.
    ns: A namespace for the key (defaults to '', which is the user's namespace).
  """

  for kv in container.metadata:
    if kv.key == key and kv.ns == ns:
      if not kv.HasField('proto'):
        return kv.value
  return None


def get_proto(container: Union[study_pb2.StudySpec, study_pb2.Trial], *,
              key: str, ns: str, cls: Type[T]) -> Optional[T]:
  """Unpacks the proto metadata into message.

  Args:
    container: (const) StudySpec or Trial to search the metadata from.
    key: (const) Lookup key of the metadata.
    ns: A namespace for the key (defaults to '', which is the user's namespace).
    cls: Pass in a proto ***class***, not a proto object.

  Returns:
    Proto message, if the value associated with the key exists and
    can be parsed into proto; None otherwise.
  """
  for kv in container.metadata:
    if kv.key == key and kv.ns == ns:
      if kv.HasField('proto'):
        message = cls()
        success = kv.proto.Unpack(message)
        return message if success else None
  return None
