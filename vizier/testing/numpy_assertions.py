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

"""Useful assertions for testing."""
from typing import Any, Mapping

import numpy as np


def assert_arraytree_allclose(
    d1: Mapping[str, Any], d2: Mapping[str, Any], **kwargs
) -> None:
  """Assertion function for comparing two (nested) dictionaries."""
  np.testing.assert_equal(d1.keys(), d2.keys())

  for k, v in d1.items():
    if isinstance(v, dict):
      assert_arraytree_allclose(v, d2[k], **kwargs)
    else:
      try:
        np.testing.assert_allclose(v, d2[k], **kwargs)
      except TypeError as e:
        np.testing.assert_equal(
            v, d2[k], err_msg=f'Using assert_equal due to typeerror: {e}.'
        )
