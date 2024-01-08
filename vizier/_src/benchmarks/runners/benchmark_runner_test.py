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

"""Tests for base_runner."""

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.benchmarks.experimenters import experimenter_factory
from vizier._src.benchmarks.runners import benchmark_runner
from vizier._src.benchmarks.runners import benchmark_state

from absl.testing import absltest
from absl.testing import parameterized


def _get_designer_benchmark_state_factory():
  dim = 10
  experimenter = experimenter_factory.BBOBExperimenterFactory('Sphere', dim)()

  def _designer_factory(config: vz.ProblemStatement, seed: int):
    return random.RandomDesigner(config.search_space, seed=seed)

  benchmark_state_factory = benchmark_state.DesignerBenchmarkStateFactory(
      designer_factory=_designer_factory, experimenter=experimenter
  )

  return benchmark_state_factory


class BaseRunnerTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'runner': benchmark_runner.BenchmarkRunner(
              benchmark_subroutines=[
                  benchmark_runner.GenerateSuggestions(),
                  benchmark_runner.EvaluateActiveTrials(),
              ],
              num_repeats=7,
          ),
          'expected_trials': 7,
      },
      {
          'runner': benchmark_runner.BenchmarkRunner(
              benchmark_subroutines=[benchmark_runner.GenerateAndEvaluate(10)],
              num_repeats=5,
          ),
          'expected_trials': 50,
      },
      {
          'runner': benchmark_runner.BenchmarkRunner(
              benchmark_subroutines=[
                  benchmark_runner.FillActiveTrials(10),
                  benchmark_runner.EvaluateActiveTrials(),
              ],
              num_repeats=5,
          ),
          'expected_trials': 50,
      },
  )
  def test_benchmark_run(self, runner, expected_trials):
    benchmark_state_factory = _get_designer_benchmark_state_factory()
    bench_state = benchmark_state_factory(seed=5)
    runner.run(bench_state)
    self.assertEmpty(
        bench_state.algorithm.supporter.GetTrials(
            status_matches=vz.TrialStatus.ACTIVE
        )
    )
    all_trials = bench_state.algorithm.supporter.GetTrials()
    self.assertLen(all_trials, expected_trials)
    for trial in all_trials:
      self.assertEqual(trial.status, vz.TrialStatus.COMPLETED)

  def test_active_trials(self):
    benchmark_state_factory = _get_designer_benchmark_state_factory()
    bench_state = benchmark_state_factory(seed=5)
    runner = benchmark_runner.BenchmarkRunner(
        benchmark_subroutines=[
            benchmark_runner.GenerateSuggestions(10),
            benchmark_runner.EvaluateActiveTrials(6),
        ],
        num_repeats=3,
    )
    runner.run(bench_state)
    self.assertLen(
        bench_state.algorithm.supporter.GetTrials(
            status_matches=vz.TrialStatus.ACTIVE
        ),
        4 * 3,
    )
    self.assertLen(
        bench_state.algorithm.supporter.GetTrials(
            status_matches=vz.TrialStatus.COMPLETED
        ),
        6 * 3,
    )

  def test_fill_active_trials(self):
    benchmark_state_factory = _get_designer_benchmark_state_factory()
    bench_state = benchmark_state_factory(seed=5)
    runner = benchmark_runner.BenchmarkRunner(
        benchmark_subroutines=[
            benchmark_runner.FillActiveTrials(10),
            benchmark_runner.EvaluateActiveTrials(6),  # 6 Completions
            benchmark_runner.FillActiveTrials(3),  # No-op
            benchmark_runner.EvaluateActiveTrials(6),  # 4 Completions
            benchmark_runner.FillActiveTrials(10),
            benchmark_runner.EvaluateActiveTrials(6),  # 6 completions
        ]
    )
    runner.run(bench_state)
    self.assertLen(
        bench_state.algorithm.supporter.GetTrials(
            status_matches=vz.TrialStatus.ACTIVE
        ),
        4,
    )
    self.assertLen(
        bench_state.algorithm.supporter.GetTrials(
            status_matches=vz.TrialStatus.COMPLETED
        ),
        16,
    )

  def test_benchmark_run_from_exptr_factory(self):
    benchmark_state_factory = _get_designer_benchmark_state_factory()
    bench_state = benchmark_state_factory(seed=5)
    runner = benchmark_runner.BenchmarkRunner(
        benchmark_subroutines=[benchmark_runner.GenerateAndEvaluate(10)],
        num_repeats=5,
    )
    runner.run(bench_state)
    self.assertEmpty(
        bench_state.algorithm.supporter.GetTrials(
            status_matches=vz.TrialStatus.ACTIVE
        )
    )
    all_trials = bench_state.algorithm.supporter.GetTrials()
    self.assertLen(all_trials, 50)
    for trial in all_trials:
      self.assertEqual(trial.status, vz.TrialStatus.COMPLETED)

  def test_add_prior(self):
    dim = 10
    exptr_factory = experimenter_factory.BBOBExperimenterFactory('Discus', dim)

    def _designer_factory(config: vz.ProblemStatement, seed: int):
      return random.RandomDesigner(config.search_space, seed=seed)

    prior_benchmark_state_factory = (
        benchmark_state.ExperimenterDesignerBenchmarkStateFactory(
            designer_factory=_designer_factory,
            experimenter_factory=exptr_factory,
        )
    )
    prior_runner = benchmark_runner.BenchmarkRunner(
        benchmark_subroutines=[benchmark_runner.GenerateAndEvaluate(10)],
        num_repeats=5,
    )
    prior_study_guid = 'prior'
    runner = benchmark_runner.EvaluateAndAddPriorStudy(
        benchmark_runner=prior_runner,
        benchmark_state_factory=prior_benchmark_state_factory,
        study_guid=prior_study_guid,
    )

    benchmark_state_factory = _get_designer_benchmark_state_factory()
    bench_state = benchmark_state_factory(seed=5)
    runner.run(bench_state)
    self.assertEmpty(bench_state.algorithm.supporter.GetTrials())

    prior_trials = bench_state.algorithm.supporter.GetTrials(
        study_guid=prior_study_guid
    )
    self.assertLen(prior_trials, 50)
    for trial in prior_trials:
      self.assertEqual(trial.status, vz.TrialStatus.COMPLETED)


if __name__ == '__main__':
  absltest.main()
