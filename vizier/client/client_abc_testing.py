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

"""Tests for client."""

import abc
import functools
from typing import Callable, Optional

from absl import logging
from vizier import pyvizier as vz
from vizier.client import client_abc

from absl.testing import parameterized

# Aliases are defined, so when you are developing a new client, you can
# swap it with your subclass. It makes your IDE understand which class
# you are using.
_StudyClient = client_abc.StudyInterface
_TrialClient = client_abc.TrialInterface


class VizierClientTestMixin(abc.ABC):

  @abc.abstractmethod
  def create_study(
      self, study_config: vz.ProblemStatement, name: str
  ) -> _StudyClient:
    """Create study given study config and study name."""
    pass


class MyMeta(type(VizierClientTestMixin), type(parameterized.TestCase)):
  pass


class TestCase(parameterized.TestCase, VizierClientTestMixin, metaclass=MyMeta):
  """Generic tests for cross-platform clients.

  This test provides basic coverage and it is not meant to be a comprehensive
  e2e test on all possible inputs to the client.

  Override `create_study` method.
  """

  def create_test_study(self, name: str) -> _StudyClient:
    """Creates a study."""
    logging.info('Creating study name=%s, testcasename=%s', name, self.id())
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('float', 0.0, 1.0)
    problem.metric_information.append(
        vz.MetricInformation(
            name='maximize_metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    study = self.create_study(problem, name)

    # TODO: Put this line back once we have __eq__ well-defined.
    # self.assertEqual(study.materialize_problem_statement(), problem)
    return study

  def test_create_or_load_study(self):
    study = self.create_test_study(self.id())
    study2 = study.from_resource_name(study.resource_name)
    self.assertEqual(study.resource_name, study2.resource_name)

    self.assertEqual(
        study.materialize_problem_statement(),
        study2.materialize_problem_statement(),
    )

  def test_delete_study(self):
    study = self.create_test_study(self.id())
    resource_name = study.resource_name
    study.delete()

    with self.assertRaises(Exception):
      study.from_resource_name(resource_name)

  def _example_trials(self) -> list[vz.Trial]:
    """Generates example trials."""
    trials = [
        # Completed trial.
        vz.Trial(
            parameters={'float': 0.5},
        ).complete(vz.Measurement({'maximize_metric': 1.0})),
        # Completed trial.
        vz.Trial(
            parameters={'float': 0.5},
        ).complete(vz.Measurement({'maximize_metric': 0.5})),
        # Requested trial, which will be made active below.
        vz.Trial(
            parameters={'float': 0.5},
            measurements=[vz.Measurement({'maximize_metric': 0.7})],
        ),
        # Requested trial.
        vz.Trial(parameters={'float': 0.5}, is_requested=True),
    ]
    for idx, t in enumerate(trials):
      t.metadata['future_id'] = str(idx + 1)  # id to be assigned
    return trials

  def create_test_study_with_trials(self, name: str) -> _StudyClient:
    study = self.create_test_study(name)
    trials = self._example_trials()
    for i, t in enumerate(trials):
      # TODO: Remove this.
      study._add_trial(t)  # pylint: disable=protected-access
      if i == 2:
        # Make sure the requested trial becomes ACTIVE.
        _ = study.suggest(count=1)
    return study

  @parameterized.parameters(list(state for state in vz.StudyState))
  def test_set_state(self, target_state: vz.StudyState):
    study = self.create_test_study_with_trials(self.id())
    # TODO: Check that the study moved to the target state.
    # This test is currently a placeholder.
    try:
      study.set_state(target_state)
    except NotImplementedError:
      logging.exception(
          'Set study state for %s is not implemented in %s',
          target_state,
          type(self),
      )

  def test_list_trials(self):
    study = self.create_test_study_with_trials(self.id())
    self.assertLen(list(study.trials()), 4)

  def test_trials_iter_and_get_are_equal(self):
    study = self.create_test_study_with_trials(self.id())
    all_trials = study.trials()
    self.assertEqual(
        [t.id for t in all_trials], [t.id for t in all_trials.get()]
    )

  def test_optimal_trials_on_empty(self):
    study = self.create_test_study(self.id())
    self.assertEmpty(list(study.optimal_trials()))

  def test_optimal_trials_notempty(self):
    study = self.create_test_study_with_trials(self.id())
    optimal_trials = list(study.optimal_trials())
    self.assertLen(optimal_trials, 1)
    self.assertEqual(
        optimal_trials[0].materialize().status, vz.TrialStatus.COMPLETED
    )

  def test_suggest_same_worker(self):
    # Given the same client id, should return the same trials.
    study = self.create_test_study(self.id())
    trials1 = study.suggest(count=5, client_id='worker1')
    self.assertLen(trials1, 5)
    trials2 = study.suggest(count=2, client_id='worker1')
    self.assertTrue({t.id for t in trials2}.issubset({t.id for t in trials1}))

  def test_suggest_requested(self):
    # Given the same client id, should return the same trials.
    study = self.create_test_study(self.id())
    requested_parameters = {'float': 0.11112}
    requested_trial = study.request(vz.TrialSuggestion(requested_parameters))
    self.assertCountEqual(
        requested_trial.parameters.items(), requested_parameters.items()
    )
    trials = study.suggest(count=5, client_id='worker1')
    self.assertLen(trials, 5)
    self.assertCountEqual(
        trials[0].parameters.items(), requested_parameters.items()
    )

  def test_suggest_different_workers(self):
    # Given different client ids, should generate new suggestions.
    study = self.create_test_study(self.id())
    worker1_trials = study.suggest(count=5, client_id='worker1')
    self.assertLen(worker1_trials, 5)
    worker1_trial_ids = {t.id for t in worker1_trials}
    worker2_trials = study.suggest(count=2, client_id='worker2')
    self.assertLen(worker2_trials, 2)
    worker2_trial_ids = {t.id for t in worker2_trials}
    self.assertEmpty(
        worker1_trial_ids.intersection(worker2_trial_ids),
        msg=(
            f'worker1_trial_ids={worker1_trial_ids}\n'
            f'worker2_trial_ids={worker2_trial_ids}'
        ),
    )

  def test_suggest_with_immutable_study(self):
    # Given immutable study, suggestions should be empty.
    study = self.create_test_study(self.id())
    study.set_state(vz.StudyState.ABORTED)
    trials = study.suggest(count=5, client_id='worker1')
    self.assertEmpty(trials)

  def test_get_trial(self):
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(2).materialize()
    self.assertEqual(trial.metadata['future_id'], '2')
    self.assertEqual(trial.id, 2)

  def test_get_trials(self):
    study = self.create_test_study_with_trials(self.id())
    completed = study.trials(vz.TrialFilter(status={vz.TrialStatus.COMPLETED}))
    self.assertLen(list(completed), 2)

  def test_parameters(self):
    study = self.create_test_study_with_trials(self.id())
    self.assertEqual(
        study.get_trial(1).parameters, study.get_trial(1).parameters
    )

  def test_study_update_metadata(self):
    """Checks for correct merge behavior."""
    study = self.create_test_study(self.id())
    delta_metadata = vz.Metadata({'bar': 'bar_v'}, foo='foo_v')
    study.update_metadata(delta_metadata)
    self.assertEqual(
        study.materialize_problem_statement().metadata.get('bar'), 'bar_v'
    )

    delta_metadata_2 = vz.Metadata({'bar': 'bar_w'})
    study.update_metadata(delta_metadata_2)

    problem_statement = study.materialize_problem_statement()
    self.assertEqual(problem_statement.metadata.get('bar'), 'bar_w')
    self.assertEqual(problem_statement.metadata.get('foo'), 'foo_v')

  ###########################################
  ## Tests on Trial class start here.
  ###########################################

  def test_delete_trial(self):
    study = self.create_test_study_with_trials(self.id())
    study.get_trial(2).delete()
    with self.assertRaises(Exception):
      study.get_trial(2)

  def test_complete_trial_no_measurements(self):
    study = self.create_test_study_with_trials(self.id())
    with self.assertRaises(Exception):
      study.get_trial(4).complete()

  def test_complete_trial_auto_selection(self):
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(3)
    measurement = trial.complete()
    self.assertIn(measurement, trial.materialize().measurements)

  def test_complete_trial_no_measurements_infeasible(self):
    study = self.create_test_study_with_trials(self.id())
    # Delete trial 3 so it's not suggested.
    trial = study.get_trial(3)
    trial.delete()
    # Ask Vizier to suggest the trial so it becomes ACTIVE.
    trial = study.suggest(count=1)[0]
    self.assertEqual(trial.id, 4)
    self.assertIsNone(trial.complete(infeasible_reason='just because'))
    self.assertTrue(trial.materialize().infeasible)

  def test_complete_trial_manual_measurement(self):
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(3)
    final_measurement = vz.Measurement(metrics={'maximize_metric': 0.1})
    self.assertEqual(trial.complete(final_measurement), final_measurement)
    self.assertEqual(trial.materialize().status, vz.TrialStatus.COMPLETED)

  def test_should_trial_stop(self):
    # TODO: Improve coverage.
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(3)
    self.assertIsInstance(trial.check_early_stopping(), bool)

  def test_trial_stop(self):
    """Checks for correct stopping behavior."""
    study = self.create_test_study_with_trials(self.id())
    active_trial = study.suggest(count=1)[0]
    active_trial.stop()
    self.assertEqual(active_trial.materialize().status, vz.TrialStatus.STOPPING)

    # COMPLETED, STOPPING
    noop_trials = [study.get_trial(2), active_trial]
    original_statuses = [trial.materialize().status for trial in noop_trials]
    for trial, status in zip(noop_trials, original_statuses):
      trial.stop()  # Should do nothing.
      self.assertEqual(trial.materialize().status, status)

    requested_trial = study.get_trial(4)
    with self.assertRaises(Exception):
      requested_trial.stop()

  def test_intermediate_measurement(self):
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(3)
    before = trial.materialize()
    trial.add_measurement(
        vz.Measurement(steps=2, metrics={'maximize_metric': 0.2})
    )
    after = trial.materialize()
    self.assertLen(after.measurements, len(before.measurements) + 1)

  def test_intermediate_measurement_infeasible(self):
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(3)
    trial.complete(infeasible_reason='just because')
    before = trial.materialize()
    trial.add_measurement(
        vz.Measurement(steps=2, metrics={'maximize_metric': 0.2})
    )
    after = trial.materialize()
    # Can't add measurements to infeasible trials, but doing so isn't an error.
    self.assertEqual(before, after)

  def test_intermediate_measurement_completed(self):
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(3)
    trial.complete(vz.Measurement(steps=1, metrics={'maximize_metric': 0.5}))
    # Can't add measurements to completed trials; doing so is an error.
    with self.assertRaises(Exception):
      trial.add_measurement(
          vz.Measurement(steps=2, metrics={'maximize_metric': 0.2})
      )

  def test_study_property(self):
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(2)
    self.assertEqual(
        trial.materialize(), trial.study.get_trial(2).materialize()
    )

  def assertPassesE2ETuning(
      self,
      *,
      study_factory: Optional[
          Callable[[vz.ProblemStatement], _StudyClient]
      ] = None,
      batch_size: int = 2,
      num_iterations: int = 5,
      multi_objective: bool = False,
  ):
    """Runs an e2e test.

    This test simulates a hyperparameter tuning scenario, where the model
    hyperparameters are being tuned, and automated trial early stopping is
    used to stop trials early.

    Args:
      study_factory: If not specified, uses the default factory of this class.
      batch_size:
      num_iterations:
      multi_objective:
    """
    study_factory = study_factory or functools.partial(
        self.create_study, study_id=self.id()
    )

    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param(
        'learning_rate', min_value=0.0, max_value=1.0, default_value=0.5
    )
    problem.search_space.root.add_int_param(
        'num_layers', min_value=1, max_value=5
    )
    problem.metric_information = [
        vz.MetricInformation(
            name='accuracy', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    ]
    if multi_objective:
      problem.metric_information.append(
          vz.MetricInformation(
              name='latency', goal=vz.ObjectiveMetricGoal.MINIMIZE
          )
      )

    study: _StudyClient = study_factory(problem)

    def learning_curve_simulator(learning_rate: float) -> list[float]:
      return [learning_rate * step for step in range(10)]

    for _ in range(num_iterations):
      suggestions = study.suggest(count=batch_size)
      for trial in suggestions:
        trial: _TrialClient = trial
        assert isinstance(trial, _TrialClient), type(trial)
        learning_rate = trial.parameters['learning_rate']
        num_layers = trial.parameters['num_layers']
        possible_curve = learning_curve_simulator(learning_rate)
        evaluated_curve = []
        for i, obj in enumerate(possible_curve, start=1):
          if i > 1 and trial.check_early_stopping():
            break
          evaluated_curve.append(obj)
          trial.add_measurement(
              vz.Measurement(
                  steps=i,
                  elapsed_secs=0.1 * i,
                  metrics={'accuracy': obj, 'latency': 0.5 * num_layers},
              )
          )
        trial.complete()
        stored_curve = [
            m.metrics['accuracy'].value
            for m in trial.materialize().measurements
        ]

        # See if curve was stored correctly.
        self.assertEqual(evaluated_curve, stored_curve)

        # See if final_measurement is defaulted to end of curve.
        final_measurement = trial.materialize().final_measurement
        assert final_measurement is not None
        final_accuracy = final_measurement.metrics['accuracy'].value
        self.assertEqual(evaluated_curve[-1], final_accuracy)

  def test_trial_update_metadata(self):
    """Checks for correct merge behavior."""
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(3)
    delta_metadata = vz.Metadata({'bar': 'bar_v'}, foo='foo_v')
    trial.update_metadata(delta_metadata)
    self.assertEqual(trial.materialize().metadata.get('bar'), 'bar_v')
    delta_metadata_2 = vz.Metadata({'bar': 'bar_w'})
    trial.update_metadata(delta_metadata_2)
    self.assertEqual(trial.materialize().metadata.get('bar'), 'bar_w')
    self.assertEqual(trial.materialize().metadata.get('foo'), 'foo_v')
