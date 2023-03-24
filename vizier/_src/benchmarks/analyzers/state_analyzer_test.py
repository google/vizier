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

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.benchmarks.analyzers import state_analyzer
from vizier._src.benchmarks.experimenters import experimenter_factory
from vizier._src.benchmarks.runners import benchmark_runner
from vizier._src.benchmarks.runners import benchmark_state

from absl.testing import absltest


class StateAnalyzerTest(absltest.TestCase):

  def test_curve_conversion(self):
    dim = 10
    experimenter = experimenter_factory.BBOBExperimenterFactory('Sphere', dim)()

    def _designer_factory(config: vz.ProblemStatement, seed: int):
      return random.RandomDesigner(config.search_space, seed=seed)

    benchmark_state_factory = benchmark_state.DesignerBenchmarkStateFactory(
        designer_factory=_designer_factory, experimenter=experimenter
    )
    num_trials = 20
    runner = benchmark_runner.BenchmarkRunner(
        benchmark_subroutines=[benchmark_runner.GenerateAndEvaluate()],
        num_repeats=num_trials,
    )

    states = []
    num_repeats = 3
    for i in range(num_repeats):
      bench_state = benchmark_state_factory(seed=i)
      runner.run(bench_state)
      states.append(bench_state)

    curve = state_analyzer.BenchmarkStateAnalyzer.to_curve(states)
    self.assertEqual(curve.ys.shape, (num_repeats, num_trials))

  def test_empty_curve_error(self):
    with self.assertRaisesRegex(ValueError, 'Empty'):
      state_analyzer.BenchmarkStateAnalyzer.to_curve([])

  def test_different_curve_error(self):
    exp1 = experimenter_factory.BBOBExperimenterFactory('Sphere', dim=2)()
    exp2 = experimenter_factory.BBOBExperimenterFactory('Sphere', dim=3)()

    def _designer_factory(config: vz.ProblemStatement, seed: int):
      return random.RandomDesigner(config.search_space, seed=seed)

    state1_factory = benchmark_state.DesignerBenchmarkStateFactory(
        designer_factory=_designer_factory, experimenter=exp1
    )
    state2_factory = benchmark_state.DesignerBenchmarkStateFactory(
        designer_factory=_designer_factory, experimenter=exp2
    )

    runner = benchmark_runner.BenchmarkRunner(
        benchmark_subroutines=[benchmark_runner.GenerateAndEvaluate()],
        num_repeats=10,
    )

    state1 = state1_factory()
    state2 = state2_factory()
    runner.run(state1)
    runner.run(state2)

    with self.assertRaisesRegex(ValueError, 'must have same problem'):
      state_analyzer.BenchmarkStateAnalyzer.to_curve([state1, state2])


if __name__ == '__main__':
  absltest.main()
