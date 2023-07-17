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

from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.ensemble import ensemble_design
from vizier._src.algorithms.ensemble import ensemble_designer
from vizier.benchmarks import experimenters

from absl.testing import absltest
from absl.testing import parameterized


class EnsembleDesignerTest(parameterized.TestCase):

  # pylint: disable=g-long-lambda
  @parameterized.parameters(
      (ensemble_design.EXP3UniformEnsembleDesign),
      (ensemble_design.EXP3IXEnsembleDesign),
      (
          lambda ind: ensemble_design.AdaptiveEnsembleDesign(
              indices=ind, max_lengths=[50, 100, 200]
          )
      ),
  )
  def testSmartEnsemblingWithGenerator(self, ensemble_design_factory):
    dim = 6
    bbob_factory = experimenters.BBOBExperimenterFactory('Sphere', dim)
    exptr_factory = experimenters.SingleObjectiveExperimenterFactory(
        bbob_factory
    )

    def ensemble_designer_factory(config: vz.ProblemStatement, seed: int):
      random_designer = random.RandomDesigner(config.search_space, seed=seed)
      eagle = eagle_strategy.EagleStrategyDesigner(config, seed=seed)
      reward_generator = ensemble_designer.ObjectiveRewardGenerator(
          config, reward_regularization=0.1
      )
      return ensemble_designer.EnsembleDesigner(
          {'random': random_designer, 'eagle': eagle},
          ensemble_design_factory=ensemble_design_factory,
          reward_generator=reward_generator,
      )

    benchmark_state_factory = (
        benchmarks.ExperimenterDesignerBenchmarkStateFactory(
            designer_factory=ensemble_designer_factory,
            experimenter_factory=exptr_factory,
        )
    )
    bench_state = benchmark_state_factory()
    runner = benchmarks.BenchmarkRunner(
        benchmark_subroutines=[benchmarks.GenerateAndEvaluate(5)],
        num_repeats=50,
    )
    runner.run(bench_state)
    self.assertEmpty(
        bench_state.algorithm.supporter.GetTrials(
            status_matches=vz.TrialStatus.ACTIVE
        )
    )
    all_trials = bench_state.algorithm.supporter.GetTrials()
    self.assertLen(all_trials, 250)

    num_random_trials = 0
    num_eagle_trials = 0
    for t in all_trials:
      if (
          t.metadata.ns(ensemble_designer.ENSEMBLE_NS).get(
              ensemble_designer.EXPERT_KEY
          )
          == 'random'
      ):
        num_random_trials += 1
      elif (
          t.metadata.ns(ensemble_designer.ENSEMBLE_NS).get(
              ensemble_designer.EXPERT_KEY
          )
          == 'eagle'
      ):
        num_eagle_trials += 1
    self.assertEqual(num_random_trials + num_eagle_trials, 250)


if __name__ == '__main__':
  absltest.main()
