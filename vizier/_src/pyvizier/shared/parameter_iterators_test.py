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

"""Tests for parameter iterators."""

from typing import Sequence
from typing import Literal

from vizier import pyvizier as vz
from vizier._src.pyvizier.shared import parameter_iterators as pi

from absl.testing import absltest
from absl.testing import parameterized


class ParameterIteratorsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          traverse_order='bfs',
          expected_order=[
              'model', 'apply_preprocessing', 'num_layers', 'preprocessor'
          ]),
      dict(
          traverse_order='dfs',
          expected_order=[
              'model', 'num_layers', 'apply_preprocessing', 'preprocessor'
          ]))
  def test_e2e(self, traverse_order: Literal['bfs', 'dfs'],
               expected_order: Sequence[str]):
    valid_params = {
        'model': 'dnn',
        'apply_preprocessing': True,
        'preprocessor': 'augment',
        'num_layers': 1,  # child parameter comes last.
    }
    valid_params = {k: valid_params[k] for k in expected_order}

    space = vz.SearchSpace()
    model = space.root.add_categorical_param('model', ['dnn', 'gbdt'])
    model.select_values(['dnn']).add_int_param('num_layers', 1, 4)
    model.select_values(['gbdt']).add_discrete_param('num_estimators',
                                                     [100, 200])
    preprocessing = space.root.add_bool_param('apply_preprocessing')
    preprocessing.select_values([True]).add_categorical_param(
        'preprocessor', ['normalize', 'augment'])

    builder = pi.SequentialParameterBuilder(
        space, traverse_order=traverse_order)
    for parameter_config in builder:
      builder.choose_value(valid_params[parameter_config.name])

    self.assertEqual(builder.parameters.as_dict(), valid_params)


if __name__ == '__main__':
  absltest.main()
