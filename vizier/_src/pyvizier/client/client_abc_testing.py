"""Tests for client."""

import abc
import functools
from typing import Callable, Optional

from absl import logging
from vizier import pyvizier as vz
from vizier._src.pyvizier.client import client_abc

from absl.testing import parameterized

# Aliases are defined, so when you are developing a new client, you can
# swap it with your subclass. It makes your IDE understand which class
# you are using.
_StudyClient = client_abc.StudyInterface
_TrialClient = client_abc.TrialInterface


class VizierClientTestMixin(metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def create_study(self, study_config: vz.ProblemStatement,
                   name: str) -> _StudyClient:
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
    problem.search_space.select_root().add_float_param('float', 0.0, 1.0)
    problem.metric_information.append(
        vz.MetricInformation(
            name='maximize_metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE))
    study = self.create_study(problem, name)

    # TODO: Put this line back once we have __eq__ well-defined.
    # self.assertEqual(study.materialize_problem_statement(), problem)
    return study

  def test_create_or_load_study(self):
    study = self.create_test_study(self.id())
    study2 = study.from_resource_name(study.resource_name)
    self.assertEqual(study.resource_name, study2.resource_name)

    self.assertEqual(study.materialize_problem_statement(),
                     study2.materialize_problem_statement())

  def test_delete_study(self):
    study = self.create_test_study(self.id())
    resource_name = study.resource_name
    study.delete()

    with self.assertRaises(Exception):
      study.from_resource_name(resource_name)

  def _example_trials(self) -> list[vz.Trial]:
    trials = [
        vz.Trial().complete(vz.Measurement({'maximize_metric': 1.0})),
        vz.Trial().complete(vz.Measurement({'maximize_metric': 0.5})),
        vz.Trial(measurements=[vz.Measurement({'maximize_metric': 0.7})]),
        vz.Trial(is_requested=True)
    ]
    for idx, t in enumerate(trials):
      t.metadata['future_id'] = str(idx + 1)  # id to be assigned
    return trials

  def create_test_study_with_trials(self, name: str) -> _StudyClient:
    study = self.create_test_study(name)
    for t in self._example_trials():
      # TODO: Remove this.
      study._add_trial(t)  # pylint: disable=protected-access
    return study

  def test_list_trials(self):
    study = self.create_test_study_with_trials(self.id())
    self.assertLen(list(study.trials()), 4)

  def test_optimal_trials_on_empty(self):
    study = self.create_test_study(self.id())
    self.assertEmpty(study.optimal_trials())

  def test_optimal_trials_notempty(self):
    study = self.create_test_study_with_trials(self.id())
    optimal_trials = list(study.optimal_trials())
    self.assertLen(optimal_trials, 1)
    self.assertEqual(optimal_trials[0].materialize().status,
                     vz.TrialStatus.COMPLETED)

  def test_suggest_same_worker(self):
    # Given the same client id, should return the same trials.
    study = self.create_test_study(self.id())
    trials1 = study.suggest(count=5, client_id='worker1')
    self.assertLen(trials1, 5)
    trials2 = study.suggest(count=2, client_id='worker1')
    self.assertTrue({t.id for t in trials2}.issubset({t.id for t in trials1}))

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
        msg=(f'worker1_trial_ids={worker1_trial_ids}\n'
             f'worker2_trial_ids={worker2_trial_ids}'))

  def test_get_trial(self):
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(2).materialize()
    self.assertEqual(trial.metadata['future_id'], '2')
    self.assertEqual(trial.id, 2)

  def test_get_trials(self):
    study = self.create_test_study_with_trials(self.id())
    completed = study.trials(vz.TrialFilter(status={vz.TrialStatus.COMPLETED}))
    self.assertLen(list(completed), 2)

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
    trial = study.get_trial(4)
    self.assertIsNone(trial.complete(infeasible_reason='just because'))
    self.assertTrue(trial.materialize().infeasible)

  def test_complete_trial_manual_measurement(self):
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(2)
    final_measurement = vz.Measurement(metrics={'maximize_metric': .1})
    self.assertEqual(trial.complete(final_measurement), final_measurement)
    self.assertEqual(trial.materialize().status, vz.TrialStatus.COMPLETED)

  def test_should_trial_stop(self):
    # TODO: Improve coverage.
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(2)
    self.assertIsInstance(trial.check_early_stopping(), bool)

  def test_intermediate_measurement(self):
    study = self.create_test_study_with_trials(self.id())
    trial = study.get_trial(2)
    before = trial.materialize()
    trial.add_measurement(
        vz.Measurement(steps=2, metrics={'maximize_metric': 0.2}))
    after = trial.materialize()
    self.assertLen(after.measurements, len(before.measurements) + 1)

  def assertPassesE2ETuning(
      self,
      *,
      study_factory: Optional[Callable[[vz.ProblemStatement],
                                       _StudyClient]] = None,
      batch_size: int = 2,
      num_iterations: int = 5,
      multi_objective: bool = False):
    """Runs an e2e test.

    Args:
      study_factory: If not specified, uses the default factory of this class.
      batch_size:
      num_iterations:
      multi_objective:
    """
    study_factory = study_factory or functools.partial(
        self.create_study, study_id=self.id())

    problem = vz.ProblemStatement()
    problem.search_space.select_root().add_float_param(
        'learning_rate', min_value=0.0, max_value=1.0, default_value=0.5)
    problem.search_space.select_root().add_int_param(
        'num_layers', min_value=1, max_value=5)
    problem.metric_information = [
        vz.MetricInformation(
            name='accuracy', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    ]
    if multi_objective:
      problem.metric_information.append(
          vz.MetricInformation(
              name='latency', goal=vz.ObjectiveMetricGoal.MINIMIZE))

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
        curve = learning_curve_simulator(learning_rate)
        for i in range(len(curve)):
          if i > 1 and trial.check_early_stopping():
            break
          trial.add_measurement(
              vz.Measurement(
                  steps=i,
                  elapsed_secs=0.1 * i,
                  metrics={
                      'accuracy': curve[i],
                      'latency': 0.5 * num_layers
                  }))
        trial.complete()
        stored_curve = [
            m.metrics['accuracy'].value
            for m in trial.materialize().measurements
        ]

        # See if curve was stored correctly.
        self.assertEqual(curve, stored_curve)

        # See if final_measurement is defaulted to end of curve.
        final_accuracy = (
            trial.materialize().final_measurement.metrics['accuracy'].value)
        self.assertEqual(curve[-1], final_accuracy)
