# Copyright 2022 Google LLC.
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

from typing import Generator
from vizier.utils import generator_utils
from absl.testing import absltest


class GeneratorUtilsTest(absltest.TestCase):

  def test_give_me_a_name(self):

    params: dict[str, int] = dict()

    def coroutine() -> Generator[str, int, dict[str, int]]:
      names = ['a', 'aaa', 'aa', 'aaaa']
      for name in names:
        params[name] = (yield name)
      return params

    gen = generator_utils.BetterGenerator(coroutine())
    for element in gen:
      gen.send(len(element))

    self.assertDictEqual(gen.result, {'a': 1, 'aaa': 3, 'aa': 2, 'aaaa': 4})


if __name__ == '__main__':
  absltest.main()
