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

"""Tests for scheduled designers."""

from typing import Sequence
import attrs
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import gp_bandit
from vizier._src.algorithms.designers import gp_ucb_pe
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.designers import scheduled_designer
from vizier._src.algorithms.designers import scheduled_gp_bandit
from vizier._src.algorithms.designers import scheduled_gp_ucb_pe
from vizier._src.algorithms.testing import test_runners
from vizier.testing import test_studies
from absl.testing import absltest


@attrs.define
class MockParameterizedDesigner(vza.Designer):
  """Mock parameterized designer."""

  problem: vz.ProblemStatement
  parameter1: float = 0.0
  parameter2: float = 0.0

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    """Update the designer based on completed and pending trials."""
    del completed
    del all_active

  def suggest(self, count: int = 1) -> Sequence[vz.TrialSuggestion]:
    """Suggest trials."""
    return random.RandomDesigner(self.problem.search_space).suggest(count)


class DirectDesignerStateUpdater(scheduled_designer.DesignerStateUpdater):
  """Direct designer state updater."""

  def __call__(
      self, designer: vza.Designer, params: scheduled_designer.ParamsValues
  ) -> None:
    """Update the designer state assuming the params are class attributes."""
    for param_name, param_value in params.items():
      setattr(designer, param_name, param_value)


class ScheduledDesignerTest(absltest.TestCase):

  def test_schedule_designer(self):
    expected_total_num_trials = 10
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.append(
        vz.MetricInformation(
            name="metric", goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    param1 = scheduled_designer.LinearScheduledParam(
        init_value=10.5, final_value=2.1
    )
    param2 = scheduled_designer.ExponentialScheduledParam(
        init_value=1.5, final_value=20.1, rate=1.7
    )
    mock_scheduled_designer = scheduled_designer.ScheduledDesigner(
        problem,
        designer_factory=MockParameterizedDesigner,
        designer_state_updater=DirectDesignerStateUpdater(),
        scheduled_params={"parameter1": param1, "parameter2": param2},
        expected_total_num_trials=expected_total_num_trials,
    )
    # Check initial values.
    self.assertEqual(mock_scheduled_designer.designer.parameter1, 10.5)  # pytype: disable=attribute-error
    self.assertEqual(mock_scheduled_designer.designer.parameter2, 1.5)  # pytype: disable=attribute-error
    # Check suggestions.
    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=expected_total_num_trials,
            batch_size=1,
            verbose=1,
            validate_parameters=True,
            seed=1,
        ).run_designer(mock_scheduled_designer),
        expected_total_num_trials,
    )
    # Check final values.
    self.assertAlmostEqual(mock_scheduled_designer.designer.parameter1, 2.1)  # pytype: disable=attribute-error
    self.assertAlmostEqual(mock_scheduled_designer.designer.parameter2, 20.1)  # pytype: disable=attribute-error

  def test_validate_suggested_num_trials(self):
    # Test that updating the designer with trials changes the state.
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.append(
        vz.MetricInformation(
            name="metric", goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    param1 = scheduled_designer.LinearScheduledParam(
        init_value=10.5, final_value=2.1
    )
    param2 = scheduled_designer.ExponentialScheduledParam(
        init_value=1.5, final_value=20.1, rate=1.7
    )
    mock_scheduled_designer = scheduled_designer.ScheduledDesigner(
        problem,
        designer_factory=MockParameterizedDesigner,
        designer_state_updater=DirectDesignerStateUpdater(),
        scheduled_params={"parameter1": param1, "parameter2": param2},
        expected_total_num_trials=10,
    )
    # Generate active and completed trials.
    active_trials = test_studies.flat_continuous_space_with_scaling_trials(2)
    completed_trials = [
        s.to_trial()
        for s in test_studies.flat_continuous_space_with_scaling_trials(4)
    ]
    for trial in completed_trials:
      trial.complete(vz.Measurement({"metric": 1.2}), inplace=True)
    # Update the scheduled designer.
    mock_scheduled_designer.update(
        vza.CompletedTrials(completed_trials), vza.ActiveTrials(active_trials)
    )
    # Validate that the state was updated.
    self.assertEqual(
        mock_scheduled_designer.num_incorporated_suggested_trials, 4 + 2
    )

  def test_scheduled_designer_serialization(self):
    expected_total_num_trials = 10
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.append(
        vz.MetricInformation(
            name="metric", goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    param1 = scheduled_designer.LinearScheduledParam(
        init_value=10.5, final_value=2.1
    )
    param2 = scheduled_designer.ExponentialScheduledParam(
        init_value=1.5, final_value=20.1, rate=1.7
    )
    mock_scheduled_designer = scheduled_designer.ScheduledDesigner(
        problem,
        designer_factory=MockParameterizedDesigner,
        designer_state_updater=DirectDesignerStateUpdater(),
        scheduled_params={"parameter1": param1, "parameter2": param2},
        expected_total_num_trials=expected_total_num_trials,
    )
    # Making several suggestions so the state would change.
    mock_scheduled_designer.suggest(count=1)
    mock_scheduled_designer.suggest(count=3)
    # Store the state in metadata.
    state = mock_scheduled_designer.dump()
    # Create a new designer and load state.
    new_mock_scheduled_designer = scheduled_designer.ScheduledDesigner(
        problem,
        designer_factory=MockParameterizedDesigner,
        designer_state_updater=DirectDesignerStateUpdater(),
        scheduled_params={"parameter1": param1, "parameter2": param2},
        expected_total_num_trials=expected_total_num_trials,
    )
    new_mock_scheduled_designer.load(state)
    self.assertEqual(
        new_mock_scheduled_designer._num_incorporated_suggested_trials, 4
    )


class ScheduledGpBanditTest(absltest.TestCase):

  def test_scheduled_gp_bandit(self):
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.append(
        vz.MetricInformation(
            name="metric", goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )

    def _gp_bandit_factory(problem):
      return gp_bandit.VizierGPBandit(problem)

    scheduled_desinger = scheduled_gp_bandit.ScheduledGPBanditFactory(
        gp_bandit_factory=_gp_bandit_factory,
        expected_total_num_trials=2,
        init_ucb_coefficient=4.0,
        final_ucb_coefficient=1.0,
        decay_ucb_coefficient=1.2,
    )(problem)

    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=2,
            batch_size=1,
            verbose=1,
            validate_parameters=True,
            seed=1,
        ).run_designer(scheduled_desinger),
        2,
    )


class ScheduledGpUcbPeTest(absltest.TestCase):

  def test_scheduled_gp_ucb_pe(self):
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.append(
        vz.MetricInformation(
            name="metric", goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )

    def _gp_ucb_pe_factory(
        problem: vz.ProblemStatement,
    ) -> gp_ucb_pe.VizierGPUCBPEBandit:
      return gp_ucb_pe.VizierGPUCBPEBandit(problem)

    scheduled_desinger = scheduled_gp_ucb_pe.ScheduledGPUCBPEFactory(
        gp_ucb_pe_factory=_gp_ucb_pe_factory,
        expected_total_num_trials=3,
        init_ucb_coefficient=4.0,
        final_ucb_coefficient=1.0,
        decay_ucb_coefficient=1.2,
        init_explore_region_ucb_coefficient=1.0,
        final_explore_region_ucb_coefficient=0.5,
        decay_explore_region_ucb_coefficient=1.2,
        init_ucb_overwrite_probability=0.25,
        final_ucb_overwrite_probability=0.0,
        decay_ucb_overwrite_probability=1.0,
    )(problem)

    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=3,
            batch_size=1,
            verbose=1,
            validate_parameters=True,
            seed=1,
        ).run_designer(scheduled_desinger),
        3,
    )


if __name__ == "__main__":
  absltest.main()
