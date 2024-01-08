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

"""Tests for input_warping."""

import numpy as np
from vizier import pyvizier as vz
from vizier.pyvizier.converters import core as converters
from vizier.pyvizier.converters import input_warping

from absl.testing import absltest


class InputWarpingTest(absltest.TestCase):

  def test_kumaraswamy_transformation(self):
    rng = np.random.default_rng(seed=0)
    for a in np.linspace(0.01, 2.0, 50):
      for b in np.linspace(0.01, 2.0, 50):
        x = rng.uniform(size=(50, 20))
        f = input_warping.kumaraswamy_cdf(x, a, b)
        # Test that all values are in [0,1]
        self.assertEqual(np.sum((0 <= f) & (f <= 1)), x.shape[0] * x.shape[1])
        # Test the inverse CDF.
        np.testing.assert_allclose(
            input_warping.kumaraswamy_inv_cdf(f, a, b), x, atol=1e-07
        )

  def test_convert_trials(self):
    rng = np.random.default_rng(seed=0)
    search_space = vz.SearchSpace()
    for i in range(16):
      search_space.root.add_float_param(f"x{i}", 0.0, 1.0)
    problem = vz.ProblemStatement(search_space)
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    input_warper = input_warping.KumaraswamyInputWarpingConverter(
        converter, a=0.1, b=0.8
    )

    trial1 = vz.Trial(parameters={f"x{i}": rng.uniform() for i in range(16)})
    trial2 = vz.Trial(parameters={f"x{i}": rng.uniform() for i in range(16)})
    trial3 = vz.Trial(parameters={f"x{i}": rng.uniform() for i in range(16)})
    trials = [trial1, trial2, trial3]

    features = input_warper.to_features(trials)
    self.assertEqual(np.sum((0 <= features) & (features <= 1)), 3 * 16)
    parameters = input_warper.to_parameters(features)
    for trial_idx, trial in enumerate(trials):
      for i in range(16):
        self.assertAlmostEqual(
            parameters[trial_idx][f"x{i}"].value,
            trial.parameters[f"x{i}"].value,
        )


if __name__ == "__main__":
  absltest.main()
