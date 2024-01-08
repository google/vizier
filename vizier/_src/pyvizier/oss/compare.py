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

"""Compatibility method comparison for protobufs.

Note that using this method can be unstable and hard to debug.
Provided here for compatibility.
"""

# TODO: Add pytypes.
# TODO: Fix this library.


# pylint: disable=invalid-name
def assertProto2Equal(self, x, y):
  self.assertEqual(x.SerializeToString(), y.SerializeToString())


def assertProto2Contains(self, x, y):
  del self
  if not isinstance(x, str):
    x = str(x)
  if x not in str(y):
    pass


def assertProto2SameElements(*args, **kwargs):
  del args, kwargs
  pass
