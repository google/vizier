"""Tests for vizier.service.vizier_server."""
from vizier.service import resources
from vizier.service import study_pb2
from vizier.service import test_util
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2

from google.longrunning import operations_pb2

from absl.testing import absltest
from absl.testing import parameterized


class VizierServerTest(parameterized.TestCase):

  def setUp(self):
    self.vs = vizier_server.VizierService()
    self.owner_id = 'my_username'
    self.study_id = '0123123'
    self.client_id = 'client_0'
    super().setUp()

  @parameterized.named_parameters(
      ('Maximize', study_pb2.StudySpec.MetricSpec.MAXIMIZE),
      ('Minimize', study_pb2.StudySpec.MetricSpec.MINIMIZE))
  def test_single_objective_list_optimal(self, max_or_min_proto):
    metric_id = 'accuracy'
    max_or_min_study = test_util.generate_study(
        self.owner_id,
        self.study_id,
        study_spec=study_pb2.StudySpec(metrics=[
            study_pb2.StudySpec.MetricSpec(
                metric_id=metric_id, goal=max_or_min_proto)
        ]))

    self.vs.datastore.create_study(max_or_min_study)

    lowest_trial_final_measurement = study_pb2.Measurement(
        metrics=[study_pb2.Measurement.Metric(metric_id=metric_id, value=-1.0)])

    middle_trial_final_measurement = study_pb2.Measurement(
        metrics=[study_pb2.Measurement.Metric(metric_id=metric_id, value=0.0)])

    highest_trial_final_measurement = study_pb2.Measurement(
        metrics=[study_pb2.Measurement.Metric(metric_id=metric_id, value=1.0)])

    lowest_trial = test_util.generate_trials(
        trial_id_list=[1],
        owner_id=self.owner_id,
        study_id=self.study_id,
        final_measurement=lowest_trial_final_measurement,
        state=study_pb2.Trial.State.SUCCEEDED)[0]

    middle_trial = test_util.generate_trials(
        trial_id_list=[2],
        owner_id=self.owner_id,
        study_id=self.study_id,
        final_measurement=middle_trial_final_measurement,
        state=study_pb2.Trial.State.SUCCEEDED)[0]

    highest_trial = test_util.generate_trials(
        trial_id_list=[3],
        owner_id=self.owner_id,
        study_id=self.study_id,
        final_measurement=highest_trial_final_measurement,
        state=study_pb2.Trial.State.SUCCEEDED)[0]

    another_highest_trial = test_util.generate_trials(
        trial_id_list=[4],
        owner_id=self.owner_id,
        study_id=self.study_id,
        final_measurement=highest_trial_final_measurement,
        state=study_pb2.Trial.State.SUCCEEDED)[0]

    trials_to_be_ignored = test_util.generate_trials([5, 6, 7], self.owner_id,
                                                     self.study_id)

    all_trials = [
        lowest_trial, middle_trial, highest_trial, another_highest_trial
    ] + trials_to_be_ignored

    for trial in all_trials:
      self.vs.datastore.create_trial(trial)

    optimal_trial_list = self.vs.ListOptimalTrials(
        request=vizier_service_pb2.ListOptimalTrialsRequest(
            parent=max_or_min_study.name)).optimal_trials

    if max_or_min_study == study_pb2.StudySpec.MetricSpec.MAXIMIZE:
      self.assertLen(optimal_trial_list, 2)
    elif max_or_min_study == study_pb2.StudySpec.MetricSpec.MINIMIZE:
      self.assertEqual(optimal_trial_list[0], lowest_trial)

  def test_multiobjective_list_optimal(self):
    metric_id_1 = 'x1'
    metric_id_2 = 'x2'
    study = test_util.generate_study(
        self.owner_id,
        self.study_id,
        study_spec=study_pb2.StudySpec(metrics=[
            study_pb2.StudySpec.MetricSpec(
                metric_id=metric_id_1,
                goal=study_pb2.StudySpec.MetricSpec.MAXIMIZE),
            study_pb2.StudySpec.MetricSpec(
                metric_id=metric_id_2,
                goal=study_pb2.StudySpec.MetricSpec.MINIMIZE),
        ]))
    self.vs.datastore.create_study(study)

    low_x1 = study_pb2.Measurement.Metric(metric_id=metric_id_1, value=-1.0)
    low_x2 = study_pb2.Measurement.Metric(metric_id=metric_id_2, value=-1.0)
    high_x1 = study_pb2.Measurement.Metric(metric_id=metric_id_1, value=1.0)
    high_x2 = study_pb2.Measurement.Metric(metric_id=metric_id_2, value=1.0)

    low_low = study_pb2.Measurement(metrics=[low_x1, low_x2])
    low_high = study_pb2.Measurement(metrics=[low_x1, high_x2])
    high_low = study_pb2.Measurement(metrics=[high_x1, low_x2])
    high_high = study_pb2.Measurement(metrics=[high_x1, high_x2])

    incomplete_measurement = study_pb2.Measurement(metrics=[low_x1])

    ll_trial = test_util.generate_trials(
        trial_id_list=[1],
        owner_id=self.owner_id,
        study_id=self.study_id,
        final_measurement=low_low,
        state=study_pb2.Trial.State.SUCCEEDED)[0]
    lh_trial = test_util.generate_trials(
        trial_id_list=[2],
        owner_id=self.owner_id,
        study_id=self.study_id,
        final_measurement=low_high,
        state=study_pb2.Trial.State.SUCCEEDED)[0]
    hl_trial = test_util.generate_trials(
        trial_id_list=[3],
        owner_id=self.owner_id,
        study_id=self.study_id,
        final_measurement=high_low,
        state=study_pb2.Trial.State.SUCCEEDED)[0]
    hh_trial = test_util.generate_trials(
        trial_id_list=[4],
        owner_id=self.owner_id,
        study_id=self.study_id,
        final_measurement=high_high,
        state=study_pb2.Trial.State.SUCCEEDED)[0]

    missing_measurement_trials = test_util.generate_trials(
        trial_id_list=[5, 6, 7],
        owner_id=self.owner_id,
        study_id=self.study_id,
        final_measurement=incomplete_measurement,
        state=study_pb2.Trial.State.SUCCEEDED)

    all_trials = [ll_trial, lh_trial, hl_trial, hh_trial
                 ] + missing_measurement_trials
    for trial in all_trials:
      self.vs.datastore.create_trial(trial)

    optimal_trial_list = self.vs.ListOptimalTrials(
        request=vizier_service_pb2.ListOptimalTrialsRequest(
            parent=study.name)).optimal_trials

    self.assertLen(optimal_trial_list, 1)
    self.assertEqual(optimal_trial_list[0], hl_trial)

  def test_suggest_trials(self):
    example_study_spec = test_util.generate_all_four_parameter_specs(
        algorithm=study_pb2.StudySpec.Algorithm.RANDOM_SEARCH)
    example_study = test_util.generate_study(
        self.owner_id, self.study_id, study_spec=example_study_spec)
    self.vs.datastore.create_study(example_study)

    request = vizier_service_pb2.SuggestTrialsRequest(
        parent=resources.StudyResource(self.owner_id, self.study_id).name,
        suggestion_count=2,
        client_id=self.client_id)
    operation = self.vs.SuggestTrials(request)

    # Check if operation was stored in database.
    get_operation_request = operations_pb2.GetOperationRequest(
        name=resources.SuggestionOperationResource(self.owner_id,
                                                   self.client_id, 1).name)
    get_operation_output = self.vs.GetOperation(get_operation_request)
    self.assertEqual(operation, get_operation_output)

    # Check operation contents.
    suggest_trials_response = vizier_service_pb2.SuggestTrialsResponse.FromString(
        operation.response.value)
    for trial in suggest_trials_response.trials:
      self.assertLen(trial.parameters, 4)

  @parameterized.named_parameters(('IncludeFinalMeasurement', True),
                                  ('NoFinalMeasurement', False))
  def test_complete_trial(self, include_final_measurement: bool):
    metric_id = 'x'
    study = test_util.generate_study(
        self.owner_id,
        self.study_id,
        study_spec=study_pb2.StudySpec(metrics=[
            study_pb2.StudySpec.MetricSpec(
                metric_id=metric_id,
                goal=study_pb2.StudySpec.MetricSpec.MAXIMIZE)
        ]))
    self.vs.datastore.create_study(study)

    trial = test_util.generate_trials(
        trial_id_list=[1],
        owner_id=self.owner_id,
        study_id=self.study_id,
        state=study_pb2.Trial.State.ACTIVE)[0]
    self.vs.datastore.create_trial(trial)
    complete_trial_request = vizier_service_pb2.CompleteTrialRequest(
        name=trial.name)

    if include_final_measurement:
      final_measurement = study_pb2.Measurement(metrics=[
          study_pb2.Measurement.Metric(metric_id=metric_id, value=-1.0)
      ])
      complete_trial_request.final_measurement.CopyFrom(final_measurement)
      self.vs.CompleteTrial(complete_trial_request)
    else:
      with self.assertRaises(ValueError):
        # trial and request both do not contain measurements.
        self.vs.CompleteTrial(complete_trial_request)

  def test_early_stopping(self):
    example_study_spec = test_util.generate_all_four_parameter_specs(
        algorithm=study_pb2.StudySpec.Algorithm.RANDOM_SEARCH)
    example_study = test_util.generate_study(
        self.owner_id, self.study_id, study_spec=example_study_spec)
    self.vs.datastore.create_study(example_study)

    trial = test_util.generate_trials(
        trial_id_list=[1],
        owner_id=self.owner_id,
        study_id=self.study_id,
        state=study_pb2.Trial.State.ACTIVE)[0]
    self.vs.datastore.create_trial(trial)

    request = vizier_service_pb2.CheckTrialEarlyStoppingStateRequest(
        trial_name=trial.name)

    # TODO: Write more comprehensive test.
    self.vs.CheckTrialEarlyStoppingState(request)


if __name__ == '__main__':
  absltest.main()
