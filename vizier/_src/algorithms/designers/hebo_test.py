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

"""Tests for gp_bandit."""

from typing import Any

import optax
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import hebo
from vizier._src.algorithms.testing import test_runners
from vizier._src.jax import optimizers
from vizier.testing import test_studies

from absl.testing import absltest
from absl.testing import parameterized

ensemble_ard_optimizer = optimizers.OptaxTrainWithRandomRestarts(
    optax.adam(5e-3), epochs=100, verbose=False, random_restarts=10, best_n=5)

noensemble_ard_optimizer = optimizers.OptaxTrainWithRandomRestarts(
    optax.adam(5e-3), epochs=100, verbose=False, random_restarts=10, best_n=1)


class GoogleGpBanditTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(iters=3, batch_size=5, num_seed_trials=5),
      # dict(iters=5, batch_size=1, num_seed_trials=2),
      # dict(ard_optimizer='ensemble'),
      # dict(ard_optimizer='noensemble')
  )
  def test_on_flat_continuous_space(self,
                                    *,
                                    iters: int = 5,
                                    batch_size: int = 1,
                                    num_seed_trials: int = 1,
                                    ard_optimizer: Any = 'noensemble'):
    # We use string names so that test case names are readable. Convert them
    # to objects.
    if ard_optimizer == 'noensemble':
      ard_optimizer = noensemble_ard_optimizer
    elif ard_optimizer == 'ensemble':
      ard_optimizer = ensemble_ard_optimizer

    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling())
    problem.metric_information.append(
        vz.MetricInformation(
            name='metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE))

    hebo_designer = hebo.VizierHebo(
        problem,
        num_seed_trials=num_seed_trials,
        ard_optimizer=ard_optimizer,
        nsga_batch_size=5,
        nsga_num_evaluations=100)

    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=iters,
            batch_size=batch_size,
            verbose=1,
            validate_parameters=True,
        ).run_designer(hebo_designer), iters * batch_size)


if __name__ == '__main__':
  absltest.main()
