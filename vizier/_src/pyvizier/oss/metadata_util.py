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

"""Utility functions for handling vizier metadata."""

from typing import Dict, Iterable, Literal, Optional, Tuple, Type, TypeVar, Union
from absl import logging

from vizier._src.pyvizier.shared import common
from vizier._src.pyvizier.shared import trial
from vizier._src.service import key_value_pb2
from vizier._src.service import study_pb2
from vizier._src.service import vizier_service_pb2
from google.protobuf import any_pb2
from google.protobuf.message import Message

T = TypeVar('T')


def _assign_value(
    metadatum: key_value_pb2.KeyValue, value: Union[str, any_pb2.Any, Message]
) -> None:
  """Assigns value to $metadatum."""
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
    mode: Literal['insert_or_assign', 'insert_or_error', 'insert'] = 'insert',
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
          raise ValueError(
              f'Duplicate (ns, key) pair: ({metadatum.ns}, {metadatum.key})'
          )
        existing_metadatum = metadatum
        break

  # If the metadatum does not exist, then add the (ns, key) pair.
  metadatum = existing_metadatum or container.metadata.add(key=key, ns=ns)
  _assign_value(metadatum, value)

  return metadatum, inserted


def get(
    container: Union[study_pb2.StudySpec, study_pb2.Trial], *, key: str, ns: str
) -> Optional[str]:
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


def get_proto(
    container: Union[study_pb2.StudySpec, study_pb2.Trial],
    *,
    key: str,
    ns: str,
    cls: Type[T],
) -> Optional[T]:
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


def make_key_value_list(
    metadata: common.Metadata,
) -> list[key_value_pb2.KeyValue]:
  """Convert $metadata to a list of KeyValue protobufs."""
  result = []
  for ns, k, v in metadata.all_items():
    item = key_value_pb2.KeyValue(key=k, ns=ns.encode())
    _assign_value(item, v)
    result.append(item)
  return result


def from_key_value_list(
    kv_s: Iterable[key_value_pb2.KeyValue],
) -> common.Metadata:
  """Converts a list of KeyValue protos into a Metadata object."""
  metadata = common.Metadata()
  for kv in kv_s:
    metadata.abs_ns(common.Namespace.decode(kv.ns))[kv.key] = (
        kv.proto if kv.HasField('proto') else kv.value
    )
  return metadata


def trial_metadata_to_update_list(
    trial_metadata: dict[int, common.Metadata]
) -> list[vizier_service_pb2.UnitMetadataUpdate]:
  """Convert a dictionary of Trial.id:Metadata to a list of UnitMetadataUpdate.

  Args:
    trial_metadata: Typically MetadataDelta.on_trials.

  Returns:
    a list of UnitMetadataUpdate objects.
  """
  result = []
  for trial_id, md in trial_metadata.items():
    for kv in make_key_value_list(md):
      # TODO: Verify this implementation.
      # Should str(trial_id) below be "resources.StudyResource.from_name(
      # study_resource_name).trial_resource(trial_id=str(trial_id)).name"?
      result.append(
          vizier_service_pb2.UnitMetadataUpdate(
              trial_id=str(trial_id), metadatum=kv
          )
      )
  return result


def study_metadata_to_update_list(
    study_metadata: common.Metadata,
) -> list[vizier_service_pb2.UnitMetadataUpdate]:
  """Convert `on_study` metadata to list of metadata update protos."""
  unit_metadata_updates = []
  for ns, k, v in study_metadata.all_items():
    unit_metadata_update = vizier_service_pb2.UnitMetadataUpdate()
    metadatum = unit_metadata_update.metadatum
    metadatum.key = k
    metadatum.ns = ns.encode()
    _assign_value(metadatum, v)
    unit_metadata_updates.append(unit_metadata_update)
  return unit_metadata_updates


def to_request_proto(
    study_resource_name: str, delta: trial.MetadataDelta
) -> vizier_service_pb2.UpdateMetadataRequest:
  """Create an UpdateMetadataRequest proto.

  Args:
    study_resource_name:
    delta:

  Returns:
  """
  request = vizier_service_pb2.UpdateMetadataRequest(name=study_resource_name)

  # Study Metadata
  request.delta.extend(study_metadata_to_update_list(delta.on_study))
  # Trial metadata
  request.delta.extend(trial_metadata_to_update_list(delta.on_trials))
  return request


def merge_study_metadata(
    study_spec: study_pb2.StudySpec,
    new_metadata: Iterable[key_value_pb2.KeyValue],
) -> None:
  """Merges $new_metadata into a Study's existing metadata."""
  metadata_dict: Dict[Tuple[str, str], key_value_pb2.KeyValue] = {}
  for kv in study_spec.metadata:
    metadata_dict[(kv.ns, kv.key)] = kv
  for kv in new_metadata:
    metadata_dict[(kv.ns, kv.key)] = kv
  study_spec.ClearField('metadata')
  study_spec.metadata.extend(
      sorted(metadata_dict.values(), key=lambda kv: (kv.ns, kv.key))
  )


def merge_trial_metadata(
    trial_proto: study_pb2.Trial,
    new_metadata: Iterable[vizier_service_pb2.UnitMetadataUpdate],
) -> None:
  """Merges $new_metadata into a Trial's existing metadata.

  Args:
    trial_proto: A representation of a Trial; this will be modified.
    new_metadata: Metadata that will add or update metadata in the Trial.
  NOTE: the metadata updates in $new_metadata should have the same ID as
    $trial_proto.
  """
  metadata_dict: Dict[Tuple[str, str], key_value_pb2.KeyValue] = {}
  for kv in trial_proto.metadata:
    metadata_dict[(kv.ns, kv.key)] = kv
  for md_update in new_metadata:
    if md_update.trial_id == trial_proto.id:
      metadata_dict[(md_update.metadatum.ns, md_update.metadatum.key)] = (
          md_update.metadatum
      )
    else:
      logging.warning(
          'Metadata associated with wrong trial: %s instead of %s',
          md_update.trial_id,
          trial_proto.id,
      )
  trial_proto.ClearField('metadata')
  trial_proto.metadata.extend(
      sorted(metadata_dict.values(), key=lambda kv: (kv.ns, kv.key))
  )
