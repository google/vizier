"""Tests for vizier.pyvizier.oss.metadata_util."""

from vizier._src.pyvizier.oss import metadata_util
from vizier.service import study_pb2

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


if __name__ == '__main__':
  absltest.main()
