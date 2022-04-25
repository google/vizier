"""Tests for json_utils."""

import json
import numpy as np

from vizier.utils import json_utils
from absl.testing import absltest
from absl.testing import parameterized


class JsonUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(shape=[3, 0]),
      dict(shape=[3, 5]),
  )
  def test_dump_and_recover(self, shape):
    original = {'a': np.zeros(shape), 'b': 3}
    dumped = json.dumps(original, cls=json_utils.NumpyEncoder)
    loaded = json.loads(dumped, cls=json_utils.NumpyDecoder)
    np.testing.assert_array_equal(original['a'], loaded['a'])
    self.assertEqual(original['b'], loaded['b'])


if __name__ == '__main__':
  absltest.main()
