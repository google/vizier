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

"""Convergence test for Eagle Strategy."""

import logging

import numpy as np
from vizier import algorithms as vza
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.testing import convergence_runner
from vizier._src.benchmarks.experimenters import categorical_experimenter
from vizier._src.benchmarks.experimenters import shifting_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest
from absl.testing import parameterized

continuous_dim = 4  # number of continuous parameters
categorical_dim = 8  # number of categorical parameters
num_categories = 5  # number of categories in each categorical parameter


# TODO: Test convergence on a mix problem.
class EagleStrategyConvergenceTest(parameterized.TestCase):

  def _eagle_designer_factory(self,
                              problem: vz.ProblemStatement) -> vza.Designer:
    return eagle_strategy.EagleStrategyDesiger(problem, seed=1)

  # TODO: Add more BBOB functions.
  @parameterized.named_parameters(
      ('Sphere', bbob.Sphere, np.zeros(continuous_dim)),
      ('LinearSlope', bbob.LinearSlope, 5.0 * np.ones(continuous_dim)),
  )
  def test_continuous_convergence(self, func, z_opt):

    problem = bbob.DefaultBBOBProblemStatement(continuous_dim)
    # Randomize a shift.
    rng = np.random.default_rng(0)
    shift = rng.uniform(low=-2.0, high=2.0, size=(continuous_dim,))
    logging.info('Shift: %s', shift)
    # Create numpy experimenter with shift.
    shifted_experimenter = shifting_experimenter.ShiftingExperimenter(
        exptr=benchmarks.NumpyExperimenter(func, problem), shift=shift)
    # Set the shifted optimum trial to compare against.
    optimum_trial = vz.Trial()
    for i, param_config in enumerate(problem.search_space.parameters):
      optimum_trial.parameters[param_config.name] = float(z_opt[i])
    shifted_experimenter.shift([optimum_trial])
    # Run the convergence test.

    benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=self._eagle_designer_factory,
        experimenter=shifted_experimenter,
    )
    convergence_runner.assert_benchmark_state_converges(
        benchmark_state_factory=benchmark_state_factory,
        optimum_trial=optimum_trial,
        evaluations=15_000,
        alpha=0.05,
        num_repeats=1,
        success_threshold=1,
    )

  def test_categorical_convergece(self):
    categorical_exptr = categorical_experimenter.CategoricalExperimenter(
        categorical_dim, num_categories)

    benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=self._eagle_designer_factory,
        experimenter=categorical_exptr,
    )
    convergence_runner.assert_benchmark_state_converges(
        benchmark_state_factory=benchmark_state_factory,
        optimum_trial=categorical_exptr.optimum_trial,
        evaluations=15_000,
        alpha=0.05,
        num_repeats=1,
        success_threshold=1,
    )


if __name__ == '__main__':
  absltest.main()
