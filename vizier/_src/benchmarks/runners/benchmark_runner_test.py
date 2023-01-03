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

"""Tests for base_runner."""

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.benchmarks.experimenters import experimenter_factory
from vizier._src.benchmarks.runners import benchmark_runner

from absl.testing import absltest
from absl.testing import parameterized


class BaseRunnerTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'runner':
              benchmark_runner.BenchmarkRunner(
                  benchmark_subroutines=[
                      benchmark_runner.GenerateSuggestions(),
                      benchmark_runner.EvaluateActiveTrials()
                  ],
                  num_repeats=7),
          'expected_trials':
              7
      }, {
          'runner':
              benchmark_runner.BenchmarkRunner(
                  benchmark_subroutines=[
                      benchmark_runner.GenerateAndEvaluate(10)
                  ],
                  num_repeats=5),
          'expected_trials':
              50
      })
  def test_benchmark_run(self, runner, expected_trials):
    dim = 10
    experimenter = experimenter_factory.BBOBExperimenterFactory('Sphere', dim)()

    def _designer_factory(config: vz.ProblemStatement, seed: int):
      return random.RandomDesigner(config.search_space, seed=seed)

    benchmark_state_factory = benchmark_runner.DesignerBenchmarkStateFactory(
        designer_factory=_designer_factory, experimenter=experimenter)

    benchmark_state = benchmark_state_factory(seed=5)

    runner.run(benchmark_state)
    self.assertEmpty(
        benchmark_state.algorithm.supporter.GetTrials(
            status_matches=vz.TrialStatus.ACTIVE))
    all_trials = benchmark_state.algorithm.supporter.GetTrials()
    self.assertLen(all_trials, expected_trials)
    for trial in all_trials:
      self.assertEqual(trial.status, vz.TrialStatus.COMPLETED)


if __name__ == '__main__':
  absltest.main()
