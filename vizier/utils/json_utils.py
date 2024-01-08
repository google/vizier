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

"""Json utils."""

from typing import Mapping
import json
from typing import Any

import numpy as np
from vizier import pyvizier as vz


class NumpyEncoder(json.JSONEncoder):
  """Example: json.dumps(np.array(...), cls=NumpyEncoder)."""

  def default(self, o: Any) -> Any:
    if not isinstance(o, dict) and isinstance(o, Mapping):
      # Handles FrozenDict and any dict-like structures.
      return dict(o)
    elif set(dir(o)).issuperset(set(['tolist', 'dtype', 'shape'])):
      # Any numpy.array-like objects.
      # Shape must be dumped and restored, in case that the array has shape
      # that includes 0-sized dimensions.
      return {
          'dtype': np.dtype(o.dtype).name,
          'value': o.tolist(),
          'shape': o.shape
      }
    else:
      return o


def numpy_hook(obj: Any) -> Any:
  """Example: json.loads(..., object_hook=numpy_hook)."""
  if 'dtype' not in obj:
    return obj
  if 'shape' not in obj:
    return obj
  return np.array(obj['value'], dtype=obj['dtype']).reshape(obj['shape'])


class NumpyDecoder(json.JSONDecoder):
  """Convert list to numpy if we see a list during json decoding.

  Example: json.loads(json_string, cls=NumpyDecoder).
  """

  def __init__(self, *args, **kargs):
    super(NumpyDecoder, self).__init__(object_hook=numpy_hook, *args, **kargs)


class MetadataEncoder(json.JSONEncoder):

  def default(self, o: Any) -> Any:
    if isinstance(o, vz.Metadata):
      d = dict(o)
      d |= {ns.encode(): o.ns(ns) for ns in o.subnamespaces()}
      return d
    elif not isinstance(o, dict) and isinstance(o, Mapping):
      # Handles FrozenDict and any dict-like structures.
      return dict(o)
    return o
