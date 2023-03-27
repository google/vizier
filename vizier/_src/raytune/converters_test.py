# Copyright 2023 Google LLC.
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

# Cannot be tested internally but can be on GitHub.
from ray import tune
from vizier import pyvizier as vz
from vizier._src.raytune import converters

from absl.testing import absltest


class ConvertersTest(absltest.TestCase):

  def test_run_study(self):
    space = vz.SearchSpace()
    root = space.select_root()
    root.add_float_param('uniform', 0.5, 1.5)

    def trainable(config):  # Pass a "config" dictionary into your trainable.
      score = config['uniform'] * config['uniform']
      return {'score': score}

    param_space = converters.SearchSpaceConverter.to_dict(space)
    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=10),
    )
    tuner.fit()
    self.assertLen(tuner.get_results(), 10)


if __name__ == '__main__':
  absltest.main()
