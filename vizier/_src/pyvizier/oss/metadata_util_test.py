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

"""Tests for vizier.pyvizier.oss.metadata_util."""

from vizier._src.pyvizier.oss import metadata_util
from vizier._src.service import study_pb2

from absl.testing import absltest


class MetadataUtilTest(absltest.TestCase):

  def test_get(self):
    meta_trial = study_pb2.Trial(id='meta_trial')
    trial = study_pb2.Trial(id='trial')
    metadata_util.assign(trial, key='any', ns='', value=meta_trial)
    metadata_util.assign(trial, key='text', ns='x', value='x-value')
    metadata_util.assign(trial, key='text', ns='', value='value')
    metadata_util.assign(trial, key='text', ns='y', value='y-value')
    self.assertEqual(
        metadata_util.get_proto(trial, key='any', ns='', cls=study_pb2.Trial),
        meta_trial)
    self.assertEqual(metadata_util.get(trial, key='text', ns=''), 'value')
    self.assertEqual(metadata_util.get(trial, key='text', ns='x'), 'x-value')

    self.assertIsNone(
        metadata_util.get_proto(trial, key='any', ns='', cls=study_pb2.Study))
    self.assertIsNone(metadata_util.get(trial, key='TYPO', ns=''))
    self.assertIsNone(
        metadata_util.get_proto(trial, key='TYPO', ns='', cls=study_pb2.Trial))

  def test_assign(self):
    trial = study_pb2.Trial(id='trial')
    metadata_util.assign(trial, key='k', ns='', value='value')
    self.assertEqual(metadata_util.get(trial, key='k', ns=''), 'value')
    metadata_util.assign(
        trial, key='k', ns='', value='222', mode='insert_or_assign'
    )
    self.assertEqual(metadata_util.get(trial, key='k', ns=''), '222')


if __name__ == '__main__':
  absltest.main()
