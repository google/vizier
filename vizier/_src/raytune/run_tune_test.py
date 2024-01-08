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

"""Github test for run_tune."""
# pytype: skip-file
import itertools

import numpy as np
from ray import tune
from vizier._src.raytune import run_tune

from absl.testing import absltest


class RunTuneTest(absltest.TestCase):

  def test_simple_fit(self):
    # Uses random search by default.
    tune_config = tune.TuneConfig(num_samples=7, max_concurrent_trials=1)
    results = run_tune.run_tune_bbob(
        function_name='Sphere',
        dimension=5,
        shift=np.ones(5),
        tune_config=tune_config,
    )
    self.assertLen(results, 7)

  # TODO: Remove this when able to dedicate multi-threading.
  @absltest.skip('Parallelized fit fails due to multiple worker crashing.')
  def test_parallelized_fit(self):
    tune_config = tune.TuneConfig(num_samples=3)
    function_names = ['Sphere']
    dimensions = [4]
    product_list = list(itertools.product(function_names, dimensions))
    args_list = []
    for product in product_list:
      # Add args for shift, tune_config, run_config.
      args = product + (None, tune_config, None)
      args_list.append(args)
    results_list = run_tune.run_tune_distributed(
        args_list, run_tune.run_tune_bbob
    )

    # There should be 1 studies in the product, each with 3 Trials.
    self.assertLen(results_list, 1)
    for result in results_list:
      self.assertLen(result['result'], 3)

if __name__ == '__main__':
  absltest.main()
