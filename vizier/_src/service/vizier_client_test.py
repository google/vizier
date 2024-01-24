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

"""Tests for vizier.service.vizier_client."""

# TODO: Rewrite using RPC calls and remove all direct lookups
# into the datastore.
# TODO: Cover delete_trial, delete_study, get_study_config, and
# add_trial, OR turn it into a private module

from typing import List

from absl import logging
from vizier._src.service import constants
from vizier._src.service import resources
from vizier._src.service import study_pb2
from vizier._src.service import vizier_client
from vizier._src.service import vizier_server
from vizier._src.service import vizier_service_pb2_grpc
from vizier.service import pyvizier as vz

from absl.testing import absltest
from absl.testing import parameterized


class VizierClientTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Setup Vizier Server and some pre-stored data.
    self.local_server = vizier_server.DefaultVizierServer(
        database_url=constants.SQL_MEMORY_URL
    )
    self.servicer = self.local_server._servicer
    self.owner_id = 'my_username'
    self.study_id = '1231232'
    self.study_resource_name = resources.StudyResource(
        self.owner_id, self.study_id
    ).name

    # Setup connection to server.
    vizier_client.environment_variables.server_endpoint = (
        self.local_server.endpoint
    )
    self.client = vizier_client.VizierClient(
        study_resource_name=self.study_resource_name, client_id='my_client'
    )

    # Store initial data in the vizier service.
    double_value_spec = study_pb2.StudySpec.ParameterSpec.DoubleValueSpec(
        min_value=-1.0, max_value=1.0
    )
    double_parameter_spec = study_pb2.StudySpec.ParameterSpec(
        parameter_id='double', double_value_spec=double_value_spec
    )
    metric_spec = study_pb2.StudySpec.MetricSpec(
        metric_id='example_metric',
        goal=study_pb2.StudySpec.MetricSpec.GoalType.MAXIMIZE,
    )
    self.example_study = study_pb2.Study(
        name=self.study_resource_name,
        study_spec=study_pb2.StudySpec(
            algorithm='RANDOM_SEARCH',
            parameters=[double_parameter_spec],
            metrics=[metric_spec],
        ),
    )
    self.active_trial = study_pb2.Trial(
        name=resources.TrialResource(self.owner_id, self.study_id, 1).name,
        id='1',
        state=study_pb2.Trial.State.ACTIVE,
    )
    self.servicer.datastore.create_study(self.example_study)
    self.servicer.datastore.create_trial(self.active_trial)

  def test_create_or_load_study(self):
    study_config = vz.StudyConfig()
    study_id = 'example_display_name'

    client = vizier_client.create_or_load_study(
        owner_id=self.owner_id,
        client_id='a_client',
        study_id=study_id,
        study_config=study_config,
    )
    study = self.servicer.datastore.load_study(client.study_resource_name)
    self.assertEqual(study.study_spec, study_config.to_proto())
    self.assertIsNotNone(study.name)

    another_client = vizier_client.create_or_load_study(
        owner_id=self.owner_id,
        client_id='another_client',
        study_id=study_id,
        study_config=study_config,
    )
    self.assertEqual(
        client.study_resource_name, another_client.study_resource_name
    )

  def test_list_studies(self):
    study_list_json = self.client.list_studies()
    self.assertLen(study_list_json, 1)

  @parameterized.parameters(
      (vz.StudyState.ABORTED,),
      (vz.StudyState.ACTIVE,),
  )
  def test_set_and_get_study_state(self, state):
    self.client.set_study_state(state)
    self.assertEqual(self.client.get_study_state(), state)

  def test_delete_study(self):
    self.client.delete_study()
    empty_list_json = self.client.list_studies()
    self.assertEmpty(empty_list_json)

  def test_list_trials(self):
    trial_list = self.client.list_trials()
    self.assertLen(trial_list, 1)

  def test_list_optimal_trials(self):
    for i in range(2, 10):
      metric = study_pb2.Measurement.Metric(
          metric_id='example_metric', value=0.2 * i
      )
      completed_trial = study_pb2.Trial(
          name=resources.TrialResource(self.owner_id, self.study_id, i).name,
          id=str(i),
          state=study_pb2.Trial.State.SUCCEEDED,
          final_measurement=study_pb2.Measurement(metrics=[metric]),
      )

      self.servicer.datastore.create_trial(completed_trial)

    trial_list = self.client.list_optimal_trials()
    self.assertLen(trial_list, 1)

  def test_get_trial(self):
    active_trial = self.client.get_trial(trial_id=1)
    self.assertEqual(
        active_trial, vz.TrialConverter.from_proto(self.active_trial)
    )

  @parameterized.named_parameters(
      ('Infeasible', 'infeasible_reason'), ('Complete', None)
  )
  def test_complete_trial(self, infeasibility_reason):
    final_measurement = vz.Measurement(metrics={'metric': vz.Metric(value=0.1)})
    output_trial = self.client.complete_trial(
        trial_id=1,
        final_measurement=final_measurement,
        infeasibility_reason=infeasibility_reason,
    )

    self.assertEqual(output_trial.status, vz.TrialStatus.COMPLETED)
    self.assertEqual(output_trial.infeasibility_reason, infeasibility_reason)

    # See if the rest of the contents were maintained.
    completed_trial = vz.TrialConverter.from_proto(
        self.servicer.datastore.get_trial(self.active_trial.name)
    )
    self.assertEqual(output_trial, completed_trial)

  def test_should_trial_stop(self):
    # Only trial 1 was ACTIVE, so RandomPolicy will signal to stop it.
    should_stop = self.client.should_trial_stop(trial_id=1)
    self.assertTrue(should_stop)
    self.client.stop_trial(trial_id=1)

    # The op will become recycled after the time period and early stopping will
    # be recomputed again. But RandomPolicy will consider the trial non-ACTIVE
    # and simply return False.
    self.local_server.wait_for_early_stop_recycle_period()
    should_stop_again = self.client.should_trial_stop(trial_id=1)
    self.assertFalse(should_stop_again)

  def test_intermediate_measurement(self):
    updated_trial = self.client.report_intermediate_objective_value(
        step=5,
        elapsed_secs=3.0,
        metric_list=[{'example_metric': 5}],
        trial_id=1,
    )
    self.assertLen(updated_trial.measurements, 1)
    self.assertEqual(updated_trial.measurements[0].steps, 5)
    self.assertEqual(updated_trial.measurements[0].elapsed_secs, 3.0)
    self.assertEqual(
        updated_trial.measurements[0].metrics['example_metric'],
        vz.Metric(value=5.0),
    )
    self.assertEqual(updated_trial.id, 1)

  def test_get_suggestions(self):
    suggestion_count = 2
    suggestions_list = self.client.get_suggestions(
        suggestion_count=suggestion_count
    )
    self.assertLen(suggestions_list, suggestion_count)
    logging.info('Suggestions List: %s', suggestions_list)

  # Only test algorithms which don't depend on external libraries (except for
  # numpy).
  @parameterized.parameters(
      dict(algorithm='DEFAULT'),
      dict(algorithm=vz.Algorithm.ALGORITHM_UNSPECIFIED),
      dict(algorithm=vz.Algorithm.RANDOM_SEARCH),
      dict(algorithm=vz.Algorithm.QUASI_RANDOM_SEARCH),
      dict(algorithm=vz.Algorithm.GRID_SEARCH),
      dict(algorithm=vz.Algorithm.NSGA2, multiobj=True),
      dict(algorithm=vz.Algorithm.GAUSSIAN_PROCESS_BANDIT, multiobj=True),
      dict(algorithm=vz.Algorithm.GAUSSIAN_PROCESS_BANDIT, multiobj=False),
      dict(algorithm=vz.Algorithm.GP_UCB_PE),
  )
  def test_e2e_tuning(
      self,
      *,
      algorithm,
      num_iterations: int = 10,
      batch_size: int = 1,
      multiobj: bool = False,
  ):
    # Runs end-to-end tuning via back-and-forth communication to server.
    def learning_curve_generator(learning_rate: float) -> List[float]:
      return [learning_rate * step for step in range(10)]

    study_config = vz.StudyConfig()
    study_config.search_space.root.add_float_param(
        'learning_rate', min_value=0.0, max_value=1.0, default_value=0.5
    )
    study_config.search_space.root.add_int_param(
        'num_layers', min_value=1, max_value=5
    )
    study_config.metric_information = [
        vz.MetricInformation(
            name='accuracy', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    ]
    if multiobj:
      study_config.metric_information.append(
          vz.MetricInformation(
              name='latency', goal=vz.ObjectiveMetricGoal.MINIMIZE
          )
      )
    study_config.algorithm = algorithm

    cifar10_client = vizier_client.create_or_load_study(
        owner_id=self.owner_id,
        study_id='cifar10',
        study_config=study_config,
        client_id='cifar10_client',
    )

    for _ in range(num_iterations):
      suggestions = cifar10_client.get_suggestions(suggestion_count=batch_size)
      for trial in suggestions:
        learning_rate = trial.parameters.get_value('learning_rate')
        num_layers = trial.parameters.get_value('num_layers')
        curve = learning_curve_generator(learning_rate)
        for i in range(len(curve)):
          cifar10_client.report_intermediate_objective_value(
              step=i,
              elapsed_secs=0.1 * i,
              metric_list=[{'accuracy': curve[i], 'latency': 0.5 * num_layers}],
              trial_id=trial.id,
          )
        cifar10_client.complete_trial(trial_id=trial.id)

        # Recover the trial from database.
        study_resource = resources.StudyResource.from_name(
            cifar10_client.study_resource_name
        )
        trial_name = resources.TrialResource(
            study_resource.owner_id, study_resource.study_id, trial.id
        ).name
        stored_trial = self.servicer.datastore.get_trial(trial_name)
        stored_curve = [m.metrics[0].value for m in stored_trial.measurements]

        # See if curve was stored correctly.
        self.assertEqual(curve, stored_curve)

        # See if final_measurement is defaulted to end of curve.
        final_accuracy = stored_trial.final_measurement.metrics[0].value
        self.assertEqual(curve[-1], final_accuracy)

  def test_update_metadata(self):
    # Only a smoke test, same as in `service_policy_supporter_test.py`.
    on_study_metadata = vz.Metadata()
    on_study_metadata.ns('bar')['foo'] = '.bar.foo.1'
    on_trial1_metadata = vz.Metadata()
    on_trial1_metadata.ns('bax')['nerf'] = '1.bar.nerf.2'
    metadata_delta = vz.MetadataDelta(
        on_study=on_study_metadata, on_trials={1: on_trial1_metadata}
    )
    self.client.update_metadata(metadata_delta)

  def test_unset_endpoint_client(self):
    study_id = 'dummy_study'
    study_config = vz.StudyConfig()
    study_resource_name = resources.StudyResource(self.owner_id, study_id).name

    vizier_client.environment_variables.server_endpoint = constants.NO_ENDPOINT
    vizier_client.environment_variables.servicer_use_sql_ram()
    # Check if servicer is stored in client.
    local_client1 = vizier_client.create_or_load_study(
        owner_id=self.owner_id,
        client_id='local_client1',
        study_id=study_id,
        study_config=study_config,
    )
    self.assertIsInstance(
        local_client1._service, vizier_service_pb2_grpc.VizierServiceServicer
    )

    # Check if the local server is shared.
    local_client2 = vizier_client.VizierClient(
        study_resource_name=study_resource_name, client_id='local_client2'
    )
    self.assertEqual(local_client1._service, local_client2._service)

    # Same server still exists globally in cache after clients are deleted.
    del local_client1
    del local_client2
    local_client3 = vizier_client.VizierClient(
        study_resource_name=study_resource_name, client_id='local_client3'
    )
    self.assertLen(local_client3.list_studies(), 1)


if __name__ == '__main__':
  absltest.main()
