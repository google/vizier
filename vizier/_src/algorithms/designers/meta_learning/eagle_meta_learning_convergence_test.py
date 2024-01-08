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

"""Convergence test for Eagle Auto Tuner."""

from typing import Optional
import attrs
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy
from vizier._src.algorithms.designers.eagle_strategy import testing
from vizier._src.algorithms.designers.meta_learning import eagle_meta_learning
from vizier._src.algorithms.designers.meta_learning import meta_learning
from vizier._src.algorithms.testing import comparator_runner
from vizier._src.benchmarks.experimenters.synthetic import bbob
from absl.testing import absltest
from absl.testing import parameterized


def _eagle_designer_factory(
    problem: vz.ProblemStatement, seed: Optional[int], **kwargs
):
  """Creates an EagleStrategyDesigner with hyper-parameters and seed."""
  config = eagle_strategy.FireflyAlgorithmConfig()
  # Unpack the hyperparameters into the Eagle config class.
  for param_name, param_value in kwargs.items():
    if param_name not in attrs.asdict(config):
      raise ValueError(f"'{param_name}' is not in FireflyAlgorithmConfig!")
    setattr(config, param_name, param_value)
  return eagle_strategy.EagleStrategyDesigner(
      problem_statement=problem,
      seed=seed,
      config=config,
  )


class EagleEagleMetaLearningConvergenceTest(parameterized.TestCase):
  """Convergence test for meta Eagle-Eagle designer.

  Note that all optimization problems are MINIMIZATION.
  """

  @parameterized.parameters(
      testing.create_continuous_exptr(bbob.Rastrigin),
      testing.create_categorical_exptr(),
  )
  def test_convergence(self, exptr):
    def _random_designer_factory(problem, seed):
      return random.RandomDesigner(problem.search_space, seed=seed)

    def _meta_eagle_eagle_designer_factory(problem, seed):
      meta_config = meta_learning.MetaLearningConfig(
          num_trials_per_tuning=100,
          tuning_max_num_trials=1000,
          tuning_min_num_trials=200,
      )
      return meta_learning.MetaLearningDesigner(
          problem=problem,
          tuned_designer_factory=_eagle_designer_factory,
          meta_designer_factory=_eagle_designer_factory,
          tuning_hyperparams=eagle_meta_learning.meta_eagle_search_space(),
          config=meta_config,
          seed=seed,
      )

    random_benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=_random_designer_factory, experimenter=exptr
    )

    meta_benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=_meta_eagle_eagle_designer_factory,
        experimenter=exptr,
    )
    evaluations = 1_500
    # Random designer batch size is large to expedite run time.
    comparator_runner.SimpleRegretComparisonTester(
        baseline_num_trials=2 * evaluations,
        candidate_num_trials=evaluations,
        baseline_suggestion_batch_size=2 * evaluations,
        candidate_suggestion_batch_size=5,
        baseline_num_repeats=5,
        candidate_num_repeats=1,
        alpha=0.05,
        goal=vz.ObjectiveMetricGoal.MINIMIZE,
    ).assert_benchmark_state_better_simple_regret(
        baseline_benchmark_state_factory=random_benchmark_state_factory,
        candidate_benchmark_state_factory=meta_benchmark_state_factory,
    )


if __name__ == "__main__":
  absltest.main()
