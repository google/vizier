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
from vizier.benchmarks import experimenters

from absl.testing import absltest


class ConvertersTest(absltest.TestCase):

  def test_run_study_with_search_space(self):
    space = vz.SearchSpace()
    root = space.select_root()
    root.add_float_param('uniform', 0.5, 1.5)
    root.add_float_param('loguniform', 0.1, 2.5, scale_type=vz.ScaleType.LOG)
    root.add_int_param('int_uniform', 1, 10)
    root.add_discrete_param('discrete', [1.0, 2.3, 0.5])
    root.add_categorical_param('categorical', ['a', 'b', 'c'])

    def trainable(config):  # Pass a "config" dictionary into your trainable.
      score = config['uniform'] * config['loguniform'] + config['int_uniform']
      if config['categorical'] == 'a':
        score += config['discrete']
      else:
        score -= config['discrete']

      return {'score': score}

    param_space = converters.SearchSpaceConverter.to_dict(space)
    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=10),
    )
    tuner.fit()
    self.assertLen(tuner.get_results(), 10)

  def test_conversion(self):
    dim = 2
    bbob_factory = experimenters.BBOBExperimenterFactory(name='Sphere', dim=dim)
    exptr = bbob_factory()
    search_space = exptr.problem_statement().search_space
    param_space = converters.SearchSpaceConverter.to_dict(search_space)
    trainable = converters.ExperimenterConverter.to_callable(exptr)

    config = {k: 1.5 for k in param_space.keys()}
    output_dict = trainable(config)
    self.assertEqual(type(output_dict['bbob_eval']), float)

  def test_run_study_with_experimenter(self):
    dim = 4
    bbob_factory = experimenters.BBOBExperimenterFactory(name='Sphere', dim=dim)
    exptr = bbob_factory()

    search_space = exptr.problem_statement().search_space
    self.assertLen(search_space.parameters, dim)

    param_space = converters.SearchSpaceConverter.to_dict(search_space)
    trainable = converters.ExperimenterConverter.to_callable(exptr)
    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=10),
    )
    tuner.fit()
    self.assertLen(tuner.get_results(), 10)


if __name__ == '__main__':
  absltest.main()
