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

"""Test suite for convergence runner."""

from typing import Optional, Sequence

from vizier import algorithms as vza
from vizier import benchmarks
from vizier import pyvizier as vz
from vizier._src.algorithms.testing import convergence_runner
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest


class DummyDesigner(vza.Designer):
  """Dummy designer to control convergence."""

  def __init__(self, problem: vz.ProblemStatement, convegence_value: float):
    self.problem = problem
    self.convegence_value = convegence_value
    self.non_convegence_value = 1.0
    self.num_trial_to_converge = 5
    self._current_trial_id = 0
    self.convergence_suggestion = vz.TrialSuggestion(
        parameters={
            param.name: self.convegence_value
            for param in self.problem.search_space.parameters
        })
    self.non_convergence_suggestion = vz.TrialSuggestion(
        parameters={
            param.name: self.non_convegence_value
            for param in self.problem.search_space.parameters
        })

  def update(self, delta: vza.CompletedTrials) -> None:
    pass

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    self._current_trial_id += 1
    if self._current_trial_id < self.num_trial_to_converge:
      return [self.non_convergence_suggestion]
    else:
      return [self.convergence_suggestion]


class ConvergenceRunnerTest(absltest.TestCase):
  """Test suite for convergence runner."""

  def setUp(self):
    super(ConvergenceRunnerTest, self).setUp()
    self.problem = bbob.DefaultBBOBProblemStatement(3)
    self.experimenter = benchmarks.NumpyExperimenter(bbob.Sphere, self.problem)
    self.optimum_trial = vz.Trial()
    for param_config in self.problem.search_space.parameters:
      self.optimum_trial.parameters[param_config.name] = 0.0

  def test_successful_convergence(self):
    dummy_designer_factory = lambda problem: DummyDesigner(problem, 0.0)
    benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        experimenter=self.experimenter, designer_factory=dummy_designer_factory)

    convergence_runner.assert_benchmark_state_converges(
        benchmark_state_factory=benchmark_state_factory,
        optimum_trial=self.optimum_trial,
        evaluations=1000,
        alpha=0.05,
        num_repeats=5,
        success_threshold=4,
    )

  def test_failed_convergence(self):
    dummy_designer_factory = lambda problem: DummyDesigner(problem, 0.5)
    benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        experimenter=self.experimenter, designer_factory=dummy_designer_factory)

    # pylint: disable-next=g-error-prone-assert-raises
    with self.assertRaises(
        convergence_runner.FailedSimpleRegretConvergenceTestError):
      convergence_runner.assert_benchmark_state_converges(
          benchmark_state_factory=benchmark_state_factory,
          optimum_trial=self.optimum_trial,
          evaluations=1000,
          alpha=0.05,
          num_repeats=1,
          success_threshold=1,
      )


if __name__ == '__main__':
  absltest.main()
