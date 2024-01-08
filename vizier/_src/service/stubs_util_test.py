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

"""Tests for vizier.service.stubs_util."""

import grpc
from vizier._src.service import stubs_util
from absl.testing import absltest
from absl.testing import parameterized


class StubsUtilTest(parameterized.TestCase):

  @parameterized.parameters(
      (stubs_util.create_pythia_server_stub,),
      (stubs_util.create_vizier_server_stub,),
  )
  def test_bad_endpoint(self, stub_creator):
    # Make sure connection errors out instead of stalling forever.
    endpoint = 'i_dont_exist'
    with self.assertRaises(grpc.FutureTimeoutError):
      stub_creator(endpoint, timeout=0.1)


if __name__ == '__main__':
  absltest.main()
