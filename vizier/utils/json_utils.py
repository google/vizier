"""Json utils."""

import json
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
  """Example: json.dumps(np.array(...), cls=NumpyEncoder)."""

  def default(self, obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
      # Shape must be dumped and restored, in case that the array has shape
      # that includes 0-sized dimensions.
      return {
          'dtype': np.dtype(obj.dtype).name,
          'value': obj.tolist(),
          'shape': obj.shape
      }
    return json.JSONEncoder.default(self, obj)


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
