"""Tests for vizier.service.vizier_client."""
from concurrent import futures
import datetime
import time
from typing import List
from absl import logging
import grpc
import portpicker

from vizier.service import pyvizier
from vizier.service import resources
from vizier.service import study_pb2
from vizier.service import vizier_client
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc

from google.protobuf import json_format
from absl.testing import absltest
from absl.testing import parameterized


class VizierClientTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Setup Vizier Service and some pre-stored data.
    self.early_stop_recycle_period = datetime.timedelta(seconds=1)
    self.servicer = vizier_server.VizierService(
        early_stop_recycle_period=self.early_stop_recycle_period)
    self.owner_id = 'my_username'
    self.study_id = '1231232'
    self.study_name = resources.StudyResource(self.owner_id, self.study_id).name
    self.client_id = 'my_client'

    double_value_spec = study_pb2.StudySpec.ParameterSpec.DoubleValueSpec(
        min_value=-1.0, max_value=1.0)
    double_parameter_spec = study_pb2.StudySpec.ParameterSpec(
        parameter_id='double', double_value_spec=double_value_spec)

    self.example_study = study_pb2.Study(
        name=self.study_name,
        study_spec=study_pb2.StudySpec(
            algorithm=study_pb2.StudySpec.Algorithm.RANDOM_SEARCH,
            parameters=[double_parameter_spec]))

    self.active_trial = study_pb2.Trial(
        name=resources.TrialResource(self.owner_id, self.study_id, 1).name,
        id='1',
        state=study_pb2.Trial.State.ACTIVE)

    self.servicer.datastore.create_study(self.example_study)
    self.servicer.datastore.create_trial(self.active_trial)

    # Setup local networking.
    self.port = portpicker.pick_unused_port()
    self.address = f'localhost:{self.port}'

    # Setup server.
    self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

    vizier_service_pb2_grpc.add_VizierServiceServicer_to_server(
        self.servicer, self.server)
    self.server.add_secure_port(self.address, grpc.local_server_credentials())
    self.server.start()

    # Setup connection to server.
    self.client = vizier_client.VizierClient(
        service_endpoint=self.address,
        study_name=self.study_name,
        client_id=self.client_id)

  def test_create_or_load_study(self):
    study_config = pyvizier.StudyConfig()
    study_display_name = 'example_display_name'

    client = vizier_client.create_or_load_study(
        service_endpoint=self.address,
        owner_id=self.owner_id,
        study_display_name=study_display_name,
        study_config=study_config,
        client_id='a_client')
    study = self.servicer.datastore.load_study(client.study_name)
    self.assertEqual(study.study_spec, study_config.to_proto())
    self.assertIsNotNone(study.name)

    another_client = vizier_client.create_or_load_study(
        service_endpoint=self.address,
        owner_id=self.owner_id,
        study_display_name=study_display_name,
        study_config=study_config,
        client_id='another_client')
    self.assertEqual(client.study_name, another_client.study_name)

  def test_list_studies(self):
    study_list_json = self.client.list_studies()
    self.assertLen(study_list_json, 1)

  def test_delete_study(self):
    self.client.delete_study(study_name=self.example_study.name)
    empty_list_json = self.client.list_studies()
    self.assertEmpty(empty_list_json)

  def test_list_trials(self):
    trial_list_json = self.client.list_trials()
    self.assertLen(trial_list_json, 1)

  def test_get_trial(self):
    active_trial_json = self.client.get_trial(trial_id='1')
    self.assertEqual(active_trial_json,
                     json_format.MessageToDict(self.active_trial))

  @parameterized.named_parameters(('Infeasible', 'infeasible test'),
                                  ('Complete', None))
  def test_complete_trial(self, infeasibility_reason):
    final_measurement = pyvizier.Measurement(
        metrics={'metric': pyvizier.Metric(value=0.1)})
    trial_json = self.client.complete_trial(
        trial_id='1',
        final_measurement=final_measurement,
        infeasibility_reason=infeasibility_reason)
    self.assertEqual(trial_json, json_format.MessageToDict(self.active_trial))

    trial = json_format.ParseDict(trial_json, study_pb2.Trial())
    if infeasibility_reason is not None:
      self.assertEqual(trial.state, study_pb2.Trial.State.INFEASIBLE)

  def test_should_trial_stop(self):
    # Only trial 1 was ACTIVE, so RandomPolicy will signal to stop it.
    should_stop = self.client.should_trial_stop(trial_id='1')
    self.assertTrue(should_stop)
    self.active_trial.state = study_pb2.Trial.State.STOPPING

    # The op will become recycled after the time period and early stopping will
    # be recomputed again. But RandomPolicy will consider the trial non-ACTIVE
    # and simply return False.
    time.sleep(self.early_stop_recycle_period.total_seconds())
    should_stop_again = self.client.should_trial_stop(trial_id='1')
    self.assertFalse(should_stop_again)

  def test_intermediate_measurement(self):
    self.client.report_intermediate_objective_value(
        step=5,
        elapsed_secs=3.0,
        metric_list=[{
            'example_metric': 5
        }],
        trial_id='1')

  def test_get_suggestions(self):
    suggestion_count = 2
    suggestions_list = self.client.get_suggestions(
        suggestion_count=suggestion_count)
    self.assertLen(suggestions_list, suggestion_count)
    logging.info('Suggestions List: %s', suggestions_list)

  def test_e2e_tuning(self):
    # Runs end-to-end tuning via back-and-forth communication to server.
    def learning_curve_generator(learning_rate: float) -> List[float]:
      return [learning_rate * step for step in range(10)]

    study_config = pyvizier.StudyConfig()
    study_config.search_space.select_root().add_float_param(
        'learning_rate', min_value=0.0, max_value=1.0, default_value=0.5)
    study_config.metric_information = [
        pyvizier.MetricInformation(
            name='accuracy', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ]
    study_config.algorithm = pyvizier.Algorithm.RANDOM_SEARCH

    cifar10_client = vizier_client.create_or_load_study(
        service_endpoint=self.address,
        owner_id=self.owner_id,
        study_display_name='cifar10',
        study_config=study_config,
        client_id=self.client_id)

    for t in range(1, 101):
      suggestion_json = cifar10_client.get_suggestions(suggestion_count=1)[0]
      learning_rate = suggestion_json['parameters'][0]['value']
      curve = learning_curve_generator(learning_rate)
      for i in range(len(curve)):
        cifar10_client.report_intermediate_objective_value(
            step=i,
            elapsed_secs=0.1 * i,
            metric_list=[{
                'accuracy': curve[i]
            }],
            trial_id=str(t))
      cifar10_client.complete_trial(trial_id=str(t))

      # Recover the trial from database.
      study_resource = resources.StudyResource.from_name(
          cifar10_client.study_name)
      trial_name = resources.TrialResource(study_resource.owner_id,
                                           study_resource.study_id, str(t)).name
      stored_trial = self.servicer.datastore.get_trial(trial_name)
      stored_curve = [m.metrics[0].value for m in stored_trial.measurements]

      # See if curve was stored correctly.
      self.assertEqual(curve, stored_curve)

      # See if final_measurement is defaulted to end of curve.
      final_accuracy = stored_trial.final_measurement.metrics[0].value
      self.assertEqual(curve[-1], final_accuracy)


if __name__ == '__main__':
  absltest.main()
