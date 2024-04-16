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

import jax.numpy as jnp
from vizier import algorithms as vza
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.designers import scalarization
from vizier._src.algorithms.designers import scalarizing_designer
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

  @parameterized.parameters(
      (ensemble_design.EXP3UniformEnsembleDesign),
      (ensemble_design.EXP3IXEnsembleDesign),
      (
          lambda ind: ensemble_design.AdaptiveEnsembleDesign(
              indices=ind, max_lengths=[50, 100, 200]
          )
      ),
  )
  def testPendingMulitobjectiveUpdate(self, ensemble_design_factory):
    dim = 2
    func1 = experimenters.bbob.Sphere
    func2 = experimenters.bbob.Rastrigin
    exptr1 = experimenters.NumpyExperimenter(
        func1, experimenters.bbob.DefaultBBOBProblemStatement(dim)
    )
    exptr2 = experimenters.NumpyExperimenter(
        func2, experimenters.bbob.DefaultBBOBProblemStatement(dim)
    )
    exptr = experimenters.MultiObjectiveExperimenter(
        {'m1': exptr1, 'm2': exptr2}
    )

    def ensemble_designer_factory(config: vz.ProblemStatement, seed: int):
      random_designer = random.RandomDesigner(config.search_space, seed=seed)

      def eagle_designer_factory(ps, seed):
        return eagle_strategy.EagleStrategyDesigner(
            problem_statement=ps, seed=seed
        )

      scalarized_eagle = scalarizing_designer.ScalarizingDesigner(
          config,
          eagle_designer_factory,
          scalarizer=scalarization.HyperVolumeScalarization(
              weights=jnp.ones(len(config.metric_information))
          ),
      )

      reward_generator = ensemble_designer.ObjectiveRewardGenerator(
          config, reward_regularization=0.1
      )
      return ensemble_designer.EnsembleDesigner(
          {'random': random_designer, 'eagle': scalarized_eagle},
          ensemble_design_factory=ensemble_design_factory,
          reward_generator=reward_generator,
      )

    benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=ensemble_designer_factory,
        experimenter=exptr,
    )
    bench_state = benchmark_state_factory()
    runner = benchmarks.BenchmarkRunner(
        benchmark_subroutines=[benchmarks.GenerateSuggestions(1)],
        num_repeats=5,
    )
    runner.run(bench_state)
    self.assertLen(
        bench_state.algorithm.supporter.GetTrials(
            status_matches=vz.TrialStatus.ACTIVE
        ),
        5,
    )

  def testInfeasibleTrials(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('x1', 0.0, 1.0)
    problem.metric_information = [
        vz.MetricInformation(name='obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    ]
    random_designer = random.RandomDesigner(problem.search_space)
    eagle = eagle_strategy.EagleStrategyDesigner(problem)
    ens_designer = ensemble_designer.EnsembleDesigner(
        {'random': random_designer, 'eagle': eagle},
    )
    trial1 = ens_designer.suggest(num_suggestions=1)[0].to_trial()
    trial2 = ens_designer.suggest(num_suggestions=1)[0].to_trial()
    # Complete trial1 with feasible value.
    trial1.complete(vz.Measurement({'obj': 1240.5}), inplace=True)
    # Complete trial2 with infeasible value.
    trial2.complete(
        vz.Measurement(), infeasibility_reason='infeasible', inplace=True
    )
    ens_designer.update(
        completed=vza.CompletedTrials([trial1, trial2]),
        all_active=vza.ActiveTrials([]),
    )
    ens_designer.suggest(num_suggestions=1)


if __name__ == '__main__':
  absltest.main()
