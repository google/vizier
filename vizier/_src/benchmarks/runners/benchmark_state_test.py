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

from vizier import algorithms as vza
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.benchmarks.runners import benchmark_state

from absl.testing import absltest


class BenchmarkStateTest(absltest.TestCase):

  def test_policy_suggester_active_trials_with_init(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('x', 0.0, 1.0)
    problem.search_space.root.add_float_param('y', 0.0, 1.0)
    problem.metric_information.append(
        vz.MetricInformation(
            name='maximize_metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )

    # Set up policy supporter with one Active and Completed Trial.
    policy_supporter = pythia.InRamPolicySupporter(problem)
    trials = [
        vz.Trial(parameters={'x': 0.5, 'y': 0.6}).complete(
            vz.Measurement(metrics={'maximize_metric': 0.3})
        ),
        vz.Trial(parameters={'x': 0.1, 'y': 0.2}),
    ]
    # The Completed and Active Trials should be updated in Designer.
    policy_supporter.AddTrials(trials)

    class DummyDesigner(vza.Designer):

      def __init__(self, problem_statement):
        self._designer = random.RandomDesigner(problem_statement.search_space)
        self.completed = []
        self.all_active = []

      def suggest(self, count):
        return self._designer.suggest(count)

      def update(self, completed, all_active):
        self.completed.extend(completed.trials)
        self.all_active = all_active.trials

    designer = DummyDesigner(problem)
    suggester = benchmark_state.PolicySuggester.from_designer_factory(
        problem, lambda _, **kwargs: designer, supporter=policy_supporter
    )

    suggestions = list(suggester.suggest(batch_size=5))
    self.assertLen(designer.completed, 1)
    self.assertLen(designer.all_active, 1)
    self.assertLen(suggestions, 5)

    # Report COMPLETED Trial back to supporter directly.
    completed_trial = suggestions[0].complete(
        vz.Measurement(metrics={'maximize_metric': 0.4})
    )
    suggester.supporter.AddTrials([completed_trial])
    suggester.suggest(1)
    self.assertLen(designer.completed, 2)
    self.assertLen(designer.all_active, 5)


if __name__ == '__main__':
  absltest.main()
