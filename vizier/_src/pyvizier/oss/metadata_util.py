"""Utility functions for handling vizier metadata."""

from typing import Tuple, Union, Optional, TypeVar, Type, Literal

from vizier._src.pyvizier.shared import trial
from vizier.service import key_value_pb2
from vizier.service import study_pb2
from vizier.service import vizier_service_pb2
from google.protobuf import any_pb2
from google.protobuf.message import Message

T = TypeVar('T')


def assign_value(metadatum: key_value_pb2.KeyValue,
                 value: Union[str, any_pb2.Any, Message]) -> None:
  """Assigns value to the metadatum."""
  if isinstance(value, str):
    metadatum.ClearField('proto')
    metadatum.value = value
  elif isinstance(value, any_pb2.Any):
    metadatum.ClearField('value')
    metadatum.proto.CopyFrom(value)
  else:
    metadatum.ClearField('value')
    metadatum.proto.Pack(value)


def assign(
    container: Union[study_pb2.StudySpec, study_pb2.Trial],
    *,
    key: str,
    ns: str,
    value: Union[str, any_pb2.Any, Message],
    mode: Literal['insert_or_assign', 'insert_or_error', 'insert'] = 'insert'
) -> Tuple[key_value_pb2.KeyValue, bool]:
  """Insert and/or assign (key, value) to container.metadata.

  Args:
    container: container.metadata must be repeated KeyValue (protobuf) field.
    key:
    ns: A namespace for the key (defaults to '', which is the user's namespace).
    value: Behavior depends on the type. `str` is copied to KeyValue.value
      `any_pb2.Any` is copied to KeyValue.proto Other types are packed to
      any_pb2.Any proto, which is then copied to KeyValue.proto.
    mode: `insert_or_assign` overrides the value if (ns, key)-pair already
      exists and `insert_or_error` raises ValueError if duplicate (ns, key)-pair
      exists. `insert` blindly inserts. This is fastest and should be used if
      the data source can be trusted.

  Returns:
    (proto, inserted) where
    proto is the protobuf that was just inserted into the $container, and
    inserted is True if the proto was newly inserted, False if it was replaced.
  """
  inserted = True

  # Find existing metadatum, unless in `insert` mode.
  existing_metadatum = None
  if mode in ('insert_or_assign', 'insert_or_error'):
    for metadatum in container.metadata:
      if metadatum.key == key and metadatum.ns == ns:
        inserted = False
        if mode == 'insert_or_error':
          raise ValueError(f'Duplicate (ns, key) pair: '
                           f'({metadatum.ns}, {metadatum.key})')
        existing_metadatum = metadatum
        break

  # If the metadatum does not exist, then add the (ns, key) pair.
  metadatum = existing_metadatum or container.metadata.add(key=key, ns=ns)
  assign_value(metadatum, value)

  return metadatum, inserted


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


def to_request_proto(
    study_resource_name: str,
    delta: trial.MetadataDelta) -> vizier_service_pb2.UpdateMetadataRequest:
  """Create an UpdateMetadataRequest proto.

  Args:
    study_resource_name:
    delta:

  Returns:

  """
  request = vizier_service_pb2.UpdateMetadataRequest(name=study_resource_name)

  # Study Metadata
  for ns in delta.on_study.namespaces():
    for k, v in delta.on_study.abs_ns(ns).items():
      metadatum = request.delta.add().metadatum
      metadatum.key = k
      metadatum.ns = ns.encode()
      assign_value(metadatum, v)

  # Trial Metadata
  for trial_id, trial_metadata in delta.on_trials.items():
    for ns in trial_metadata.namespaces():
      for k, v in trial_metadata.abs_ns(ns).items():
        # TODO: Verify this implementation.
        # Should this be "resources.StudyResource.from_name(
        # study_resource_name).trial_resource(trial_id=str(trial_id)).name"?
        metadatum = request.delta.add(trial_id=str(trial_id)).metadatum
        metadatum.key = k
        metadatum.ns = ns.encode()
        assign_value(metadatum, v)
  return request
