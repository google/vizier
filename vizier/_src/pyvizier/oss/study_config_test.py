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

"""Tests for vizier.pyvizier.oss.study_config."""
import datetime

from vizier._src.service import constants
from vizier._src.service import key_value_pb2
from vizier._src.service import study_pb2
from vizier.service import pyvizier as vz

from google.protobuf import struct_pb2
from vizier._src.pyvizier.oss import compare
from absl.testing import absltest
from absl.testing import parameterized


class StudyConfigTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.pconfigs = [
        study_pb2.StudySpec.ParameterSpec(
            parameter_id='learning_rate',
            double_value_spec=study_pb2.StudySpec.ParameterSpec.DoubleValueSpec(
                min_value=0.00001, max_value=1.0),
            scale_type=study_pb2.StudySpec.ParameterSpec.ScaleType
            .UNIT_LINEAR_SCALE),
        study_pb2.StudySpec.ParameterSpec(
            parameter_id='optimizer',
            categorical_value_spec=study_pb2.StudySpec.ParameterSpec
            .CategoricalValueSpec(values=['adagrad', 'adam', 'experimental'])),
    ]

  def testCreationFromAndToProtoStudy(self):
    expected_automated_stopping_config = (
        study_pb2.StudySpec.DefaultEarlyStoppingSpec())

    study_config_proto = study_pb2.StudySpec(
        algorithm='QUASI_RANDOM_SEARCH',
        metrics=[
            study_pb2.StudySpec.MetricSpec(
                metric_id='pr-auc',
                goal=study_pb2.StudySpec.MetricSpec.GoalType.MAXIMIZE,
            )
        ],
        default_stopping_spec=expected_automated_stopping_config,
        metadata=[
            key_value_pb2.KeyValue(
                key='foo',
                ns=vz.Namespace(['ns_bar']).encode(),
                value='val',
            )
        ],
    )

    study_config_proto.parameters.extend(self.pconfigs)
    # Test all proprties.
    sc = vz.StudyConfig.from_proto(study_config_proto)
    expected = vz.MetricsConfig(
        [
            vz.MetricInformation(
                name='pr-auc', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    self.assertEqual(sc.metric_information, expected)
    self.assertEqual(sc.algorithm, 'QUASI_RANDOM_SEARCH')
    self.assertIsNone(sc.pythia_endpoint)
    self.assertEqual(sc.single_objective_metric_name, 'pr-auc')
    self.assertTrue(sc.is_single_objective)
    assert sc.automated_stopping_config is not None
    compare.assertProto2Equal(self, expected_automated_stopping_config,
                              sc.automated_stopping_config.to_proto())
    compare.assertProto2Equal(self, study_config_proto, sc.to_proto())
    _ = vz.StudyConfig.from_problem(sc.to_problem())  # smoke test.

  @parameterized.parameters([
      ('my custom algorithm'),
      ('QUASI_RANDOM_SEARCH'),
  ])
  def testCreationFromAndToProtoStudyStringAlgorithm(self, algorithm):
    study_config_proto = study_pb2.StudySpec(
        metrics=[
            study_pb2.StudySpec.MetricSpec(
                metric_id='pr-auc',
                goal=study_pb2.StudySpec.MetricSpec.GoalType.MAXIMIZE)
        ],
        algorithm=algorithm)
    # Test the algorithm when pythia endpoint is not specified.
    # This can be used when a pythia service is injected directly to the Vizier
    # service class.
    sc = vz.StudyConfig.from_proto(study_config_proto)
    self.assertEqual(sc.algorithm, algorithm)
    self.assertIsNone(sc.pythia_endpoint)

    compare.assertProto2Equal(self, study_config_proto, sc.to_proto())
    _ = vz.StudyConfig.from_problem(sc.to_problem())  # smoke test.

  @parameterized.parameters([
      'my custom algorithm',
      'QUASI_RANDOM_SEARCH',
  ])
  def testCreationFromAndToProtoStudyStringAlgorithmPythiaEndpoint(
      self, algorithm
  ):
    study_config_proto = study_pb2.StudySpec(
        metrics=[
            study_pb2.StudySpec.MetricSpec(
                metric_id='pr-auc',
                goal=study_pb2.StudySpec.MetricSpec.GoalType.MAXIMIZE,
            )
        ],
        algorithm=algorithm,
        metadata=[
            key_value_pb2.KeyValue(
                key=constants.PYTHIA_ENDPOINT_KEY,
                ns=vz.Namespace([constants.PYTHIA_ENDPOINT_NAMESPACE]).encode(),
                value='localhost:8888',
            )
        ],
    )
    sc = vz.StudyConfig.from_proto(study_config_proto)
    self.assertEqual(sc.algorithm, algorithm)
    self.assertEqual(sc.pythia_endpoint, 'localhost:8888')

    compare.assertProto2Equal(self, study_config_proto, sc.to_proto())
    _ = vz.StudyConfig.from_problem(sc.to_problem())  # smoke test.

  def testCreationFromAndToProtoMultiObjectiveStudy(self):
    study_config_proto = study_pb2.StudySpec(
        metrics=[
            study_pb2.StudySpec.MetricSpec(
                metric_id='pr-auc',
                goal=study_pb2.StudySpec.MetricSpec.GoalType.MAXIMIZE),
            study_pb2.StudySpec.MetricSpec(
                metric_id='loss',
                goal=study_pb2.StudySpec.MetricSpec.GoalType.MINIMIZE)
        ],)
    study_config_proto.parameters.extend(self.pconfigs)
    # Test all proprties.
    sc = vz.StudyConfig.from_proto(study_config_proto)

    expected = vz.MetricsConfig([
        vz.MetricInformation(name='loss', goal=vz.ObjectiveMetricGoal.MINIMIZE),
        vz.MetricInformation(
            name='pr-auc', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        ),
    ])
    self.assertEqual(sc.metric_information, expected)
    self.assertIsNone(sc.single_objective_metric_name)
    self.assertFalse(sc.is_single_objective)

    round_trip_proto = sc.to_proto()
    compare.assertProto2SameElements(self, study_config_proto, round_trip_proto)

    _ = vz.StudyConfig.from_problem(sc.to_problem())  # smoke test.

  def testCreationFromAndToProtoSafeStudy(self):
    expected_automated_stopping_config = (
        study_pb2.StudySpec.DefaultEarlyStoppingSpec())

    study_config_proto = study_pb2.StudySpec(
        algorithm='QUASI_RANDOM_SEARCH',
        metrics=[
            study_pb2.StudySpec.MetricSpec(
                metric_id='pr-auc',
                goal=study_pb2.StudySpec.MetricSpec.GoalType.MAXIMIZE),
            study_pb2.StudySpec.MetricSpec(
                metric_id='privacy-safety',
                goal=study_pb2.StudySpec.MetricSpec.GoalType.MINIMIZE,
                safety_config=study_pb2.StudySpec.MetricSpec.SafetyMetricConfig(
                    safety_threshold=0.2, desired_min_safe_trials_fraction=0.8))
        ],
        default_stopping_spec=expected_automated_stopping_config)

    study_config_proto.parameters.extend(self.pconfigs)
    # Test all proprties.
    sc = vz.StudyConfig.from_proto(study_config_proto)
    expected = vz.MetricsConfig([
        vz.MetricInformation(
            name='pr-auc', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        ),
        vz.MetricInformation(
            name='privacy-safety',
            goal=vz.ObjectiveMetricGoal.MINIMIZE,
            safety_threshold=0.2,
            desired_min_safe_trials_fraction=0.8,
        ),
    ])
    self.assertEqual(sc.metric_information, expected)
    self.assertEqual(sc.algorithm, 'QUASI_RANDOM_SEARCH')
    self.assertIsNone(sc.pythia_endpoint)
    self.assertEqual(sc.single_objective_metric_name, 'pr-auc')
    self.assertTrue(sc.is_single_objective)

    assert sc.automated_stopping_config is not None
    compare.assertProto2Equal(self, expected_automated_stopping_config,
                              sc.automated_stopping_config.to_proto())
    compare.assertProto2Equal(self, study_config_proto, sc.to_proto())
    _ = vz.StudyConfig.from_problem(sc.to_problem())  # smoke test.

  def testCreationFromProtoNoGoalRaises(self):
    study_config_proto = study_pb2.StudySpec()

    sc = vz.StudyConfig.from_proto(study_config_proto)
    self.assertEmpty(sc.metric_information)

  def testMetadata(self):
    empty_trial = study_pb2.Trial(id=str(1))
    sc = vz.StudyConfig()
    sc.metric_information.append(
        vz.MetricInformation(
            name='pr-auc', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    sc.metadata.abs_ns(['ns'])['key'] = 'ns-value'
    sc.metadata.abs_ns()['key'] = 'value'
    sc.metadata['proto'] = empty_trial

    proto = sc.to_proto()
    compare.assertProto2Contains(
        self, """metadata {
        key: "key"
        ns: ":ns"
        value: "ns-value"
      }
      metadata {
        key: "key"
        value: "value"
      }
      metadata {
        key: "proto"
        proto {
          [type.googleapis.com/vizier.Trial] {
            id: '1'
          }
        }
      }
    """, proto)
    from_proto = sc.from_proto(proto).metadata
    self.assertCountEqual(sc.metadata, from_proto)

  def testCreation(self):
    sc = vz.StudyConfig()
    sc.algorithm = vz.Algorithm.RANDOM_SEARCH
    sc.metric_information.append(
        vz.MetricInformation(
            name='pr-auc', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    root = sc.search_space.root
    root.add_float_param(
        'learning_rate', 0.00001, 1.0, scale_type=vz.ScaleType.LINEAR
    )
    root.add_categorical_param('optimizer', ['adagrad', 'adam', 'experimental'])

    sc.automated_stopping_config = (
        vz.AutomatedStoppingConfig.default_stopping_spec()
    )

    # Test all proprties.
    self.assertEqual(sc.algorithm, 'RANDOM_SEARCH')
    expected = vz.MetricsConfig(
        [
            vz.MetricInformation(
                name='pr-auc', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    self.assertEqual(sc.metric_information, expected)
    self.assertEqual(sc.single_objective_metric_name, 'pr-auc')
    self.assertTrue(sc.is_single_objective)

    expected = study_pb2.StudySpec(
        algorithm='RANDOM_SEARCH',
        metrics=[
            study_pb2.StudySpec.MetricSpec(
                metric_id='pr-auc',
                goal=study_pb2.StudySpec.MetricSpec.GoalType.MAXIMIZE)
        ],
        default_stopping_spec=study_pb2.StudySpec.DefaultEarlyStoppingSpec(),
        observation_noise=study_pb2.StudySpec.ObservationNoise
        .OBSERVATION_NOISE_UNSPECIFIED,
    )
    expected.parameters.extend(self.pconfigs)
    compare.assertProto2Equal(self, expected, sc.to_proto())

  @absltest.skip('???')
  def testTrialToDict(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root
    root.add_float_param('learning_rate', 0.01, 3.0)
    root.add_int_param('units', 10, 1000, scale_type=vz.ScaleType.LOG)
    root.add_discrete_param('batch_size', [8, 16, 32])
    root.add_discrete_param(
        'floating_point_param', [8., 16., 32.], auto_cast=False)
    root.add_categorical_param('activation', ['tanh', 'relu'])
    root.add_bool_param('synchronous')

    trial_proto = study_pb2.Trial(id=str(1))
    trial_proto.parameters.add(
        parameter_id='activation', value=struct_pb2.Value(string_value='relu'))
    trial_proto.parameters.add(
        parameter_id='synchronus', value=struct_pb2.Value(string_value='true'))
    trial_proto.parameters.add(
        parameter_id='batch_size', value=struct_pb2.Value(number_value=32))
    trial_proto.parameters.add(
        parameter_id='floating_point_param',
        value=struct_pb2.Value(number_value=32))
    trial_proto.parameters.add(
        parameter_id='learning_rate', value=struct_pb2.Value(number_value=0.5))
    trial_proto.parameters.add(
        parameter_id='units', value=struct_pb2.Value(number_value=50))

    parameters = py_study_config.trial_parameters(trial_proto)
    expected = {
        'learning_rate': 0.5,
        'units': 50,
        'activation': 'relu',
        'batch_size': 32,
        'floating_point_param': 32.,
        'synchronous': True
    }
    self.assertEqual(expected, parameters)
    self.assertIsInstance(parameters['batch_size'], int)
    self.assertIsInstance(parameters['floating_point_param'], float)

  def testPyTrialToDict(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root
    root.add_float_param('learning_rate', 0.01, 3.0)
    root.add_int_param('units', 10, 1000, scale_type=vz.ScaleType.LOG)
    root.add_discrete_param('batch_size', [8, 16, 32])
    root.add_discrete_param(
        'floating_point_param', [8., 16., 32.], auto_cast=False)
    root.add_categorical_param('activation', ['tanh', 'relu'])
    root.add_bool_param('synchronous')

    pytrial = vz.Trial(id=1)
    pytrial.parameters = {
        'activation': vz.ParameterValue(value='relu'),
        'synchronous': vz.ParameterValue(value=True),
        'batch_size': vz.ParameterValue(value=32),
        'floating_point_param': vz.ParameterValue(value=32.0),
        'learning_rate': vz.ParameterValue(value=0.5),
        'units': vz.ParameterValue(value=50),
    }
    parameters = py_study_config._pytrial_parameters(pytrial)
    expected = {
        'learning_rate': 0.5,
        'units': 50,
        'activation': 'relu',
        'batch_size': 32,
        'floating_point_param': 32.,
        'synchronous': True
    }
    self.assertEqual(expected, parameters)
    self.assertIsInstance(parameters['batch_size'], int)
    self.assertIsInstance(parameters['floating_point_param'], float)

  def testTrialToDictWithoutExternalType(self):
    """Test conversion when external types are not specified."""
    proto = study_pb2.StudySpec()
    proto.parameters.add(
        parameter_id='learning_rate',
        double_value_spec=study_pb2.StudySpec.ParameterSpec.DoubleValueSpec(
            min_value=1e-4, max_value=0.1),
        scale_type=study_pb2.StudySpec.ParameterSpec.ScaleType.UNIT_LOG_SCALE)
    proto.parameters.add(
        parameter_id='batch_size',
        discrete_value_spec=study_pb2.StudySpec.ParameterSpec.DiscreteValueSpec(
            values=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]))
    proto.parameters.add(
        parameter_id='training_steps',
        discrete_value_spec=study_pb2.StudySpec.ParameterSpec.DiscreteValueSpec(
            values=[1000.0, 10000.0]))
    proto.observation_noise = study_pb2.StudySpec.ObservationNoise.HIGH
    proto.metrics.add(
        metric_id='loss', goal=study_pb2.StudySpec.MetricSpec.MINIMIZE)

    trial_proto = study_pb2.Trial()
    trial_proto.id = str(1)
    trial_proto.parameters.add(
        parameter_id='batch_size', value=struct_pb2.Value(number_value=128.0))
    trial_proto.parameters.add(
        parameter_id='learning_rate',
        value=struct_pb2.Value(number_value=1.2137854406366652E-4))
    trial_proto.parameters.add(
        parameter_id='training_steps',
        value=struct_pb2.Value(number_value=10000.0))

    py_study_config = vz.StudyConfig.from_proto(proto)
    self.assertEqual(
        py_study_config.observation_noise, vz.ObservationNoise.HIGH
    )
    parameters = py_study_config.trial_parameters(trial_proto)
    self.assertEqual(
        py_study_config.observation_noise, vz.ObservationNoise.HIGH
    )
    expected = {
        'batch_size': 128,
        'learning_rate': 1.2137854406366652E-4,
        'training_steps': 10000.0
    }
    self.assertEqual(expected, parameters)
    self.assertIsInstance(parameters['learning_rate'], float)
    self.assertIsInstance(parameters['batch_size'], float)
    self.assertIsInstance(parameters['training_steps'], float)

  @absltest.skip('???')
  def testTrialToDictMultidimensional(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root
    for index in (0, 1):
      root.add_float_param('learning_rate', 0.01, 3.0, index=index)
      root.add_int_param(
          'units', 10, 1000, scale_type=vz.ScaleType.LOG, index=index
      )
      root.add_categorical_param('activation', ['tanh', 'relu'], index=index)
      root.add_bool_param('synchronous', index=index)
      root.add_discrete_param('batch_size', [8, 16, 32], index=index)
    root.add_discrete_param(
        'floating_point_param', [8., 16., 32.], auto_cast=False)

    trial_proto = study_pb2.Trial()
    trial_proto.id = str(2)
    trial_proto.parameters.add(
        parameter_id='learning_rate[0]',
        value=struct_pb2.Value(number_value=0.5))
    trial_proto.parameters.add(
        parameter_id='learning_rate[1]',
        value=struct_pb2.Value(number_value=0.1))
    trial_proto.parameters.add(
        parameter_id='units[0]', value=struct_pb2.Value(number_value=50))
    trial_proto.parameters.add(
        parameter_id='units[1]', value=struct_pb2.Value(number_value=200))
    trial_proto.parameters.add(
        parameter_id='activation[0]',
        value=struct_pb2.Value(string_value='relu'))
    trial_proto.parameters.add(
        parameter_id='activation[1]',
        value=struct_pb2.Value(string_value='relu'))
    trial_proto.parameters.add(
        parameter_id='synchronus[0]',
        value=struct_pb2.Value(string_value='true'))
    trial_proto.parameters.add(
        parameter_id='synchronus[1]',
        value=struct_pb2.Value(string_value='false'))
    trial_proto.parameters.add(
        parameter_id='batch_size[0]', value=struct_pb2.Value(number_value=32.0))
    trial_proto.parameters.add(
        parameter_id='batch_size[1]', value=struct_pb2.Value(number_value=8.0))
    trial_proto.parameters.add(
        parameter_id='floating_point_param',
        value=struct_pb2.Value(number_value=16.0))
    parameters = py_study_config.trial_parameters(trial_proto)
    expected = {
        'learning_rate': [0.5, 0.1],
        'units': [50, 200],
        'activation': ['relu', 'relu'],
        'batch_size': [32, 8],
        'synchronous': [True, False],
        'floating_point_param': 16.,
    }
    self.assertEqual(expected, parameters)

  def testPyTrialToDictMultidimensional(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root
    for index in (0, 1):
      root.add_float_param('learning_rate', 0.01, 3.0, index=index)
      root.add_int_param(
          'units', 10, 1000, scale_type=vz.ScaleType.LOG, index=index
      )
      root.add_categorical_param('activation', ['tanh', 'relu'], index=index)
      root.add_bool_param('synchronous', index=index)
      root.add_discrete_param('batch_size', [8, 16, 32], index=index)
    root.add_discrete_param(
        'floating_point_param', [8., 16., 32.], auto_cast=False)

    pytrial = vz.Trial(id=2)
    pytrial.parameters = {
        'learning_rate[0]': vz.ParameterValue(value=0.5),
        'learning_rate[1]': vz.ParameterValue(value=0.1),
        'units[0]': vz.ParameterValue(value=50),
        'units[1]': vz.ParameterValue(value=200),
        'activation[0]': vz.ParameterValue(value='relu'),
        'activation[1]': vz.ParameterValue(value='relu'),
        'synchronous[0]': vz.ParameterValue(value=True),
        'synchronous[1]': vz.ParameterValue(value=False),
        'batch_size[0]': vz.ParameterValue(value=32.0),
        'batch_size[1]': vz.ParameterValue(value=8.0),
        'floating_point_param': vz.ParameterValue(value=16.0),
    }
    parameters = py_study_config._pytrial_parameters(pytrial)
    expected = {
        'learning_rate': [0.5, 0.1],
        'units': [50, 200],
        'activation': ['relu', 'relu'],
        'batch_size': [32, 8],
        'synchronous': [True, False],
        'floating_point_param': 16.,
    }
    self.assertEqual(expected, parameters)

  def testGinConfigMultiDimensional(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root
    block_categories = [
        'block_3x3', 'block_4x4', 'block_1x3_3x1', 'block_1x3_3x1_dw',
        'block_identity'
    ]
    for index in range(5):
      root.add_categorical_param(
          '_gin.ambient_net_exp_from_vec.block_type',
          feasible_values=block_categories,
          index=index)

    trial_proto = study_pb2.Trial()
    trial_proto.id = str(2)
    trial_proto.parameters.add(
        parameter_id='_gin.ambient_net_exp_from_vec.block_type[0]',
        value=struct_pb2.Value(string_value='block_3x3'))
    trial_proto.parameters.add(
        parameter_id='_gin.ambient_net_exp_from_vec.block_type[1]',
        value=struct_pb2.Value(string_value='block_4x4'))
    trial_proto.parameters.add(
        parameter_id='_gin.ambient_net_exp_from_vec.block_type[2]',
        value=struct_pb2.Value(string_value='block_1x3_3x1_dw'))
    trial_proto.parameters.add(
        parameter_id='_gin.ambient_net_exp_from_vec.block_type[3]',
        value=struct_pb2.Value(string_value='block_identity'))
    trial_proto.parameters.add(
        parameter_id='_gin.ambient_net_exp_from_vec.block_type[4]',
        value=struct_pb2.Value(string_value='block_1x3_3x1'))

    parameters = py_study_config.trial_parameters(trial_proto)
    expected = {
        '_gin.ambient_net_exp_from_vec.block_type': [
            'block_3x3', 'block_4x4', 'block_1x3_3x1_dw', 'block_identity',
            'block_1x3_3x1'
        ],
    }
    self.assertEqual(expected, parameters)

  @absltest.skip('???')
  def testTrialToDictConditional(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root

    model_type = root.add_categorical_param('model_type', ['dnn', 'linear'])
    dnn = model_type.select_values(['dnn'])
    dnn.add_float_param('learning_rate', 0.01, 3.0)
    dnn.add_int_param('units', 1, 50, index=0)
    dnn.add_int_param('units', 1, 80, index=1)
    dnn.add_categorical_param('activation', ['tanh', 'relu'])
    model_type.select_values(['linear'
                             ]).add_float_param('learning_rate', 0.01, 1.0)

    dnn_trial = study_pb2.Trial()
    dnn_trial.id = str(1)
    dnn_trial.parameters.add(
        parameter_id='model_type', value=struct_pb2.Value(string_value='dnn'))
    dnn_trial.parameters.add(
        parameter_id='learning_rate', value=struct_pb2.Value(number_value=2.1))
    dnn_trial.parameters.add(
        parameter_id='unts[0]', value=struct_pb2.Value(number_value=49))
    dnn_trial.parameters.add(
        parameter_id='unts[1]', value=struct_pb2.Value(number_value=79))
    dnn_trial.parameters.add(
        parameter_id='activation', value=struct_pb2.Value(string_value='relu'))

    parameters = py_study_config.trial_parameters(dnn_trial)
    expected = {
        'model_type': 'dnn',
        'learning_rate': 2.1,
        'units': [49, 79],
        'activation': 'relu',
    }
    self.assertEqual(expected, parameters)

  def testPyTrialToDictConditional(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root

    model_type = root.add_categorical_param('model_type', ['dnn', 'linear'])
    dnn = model_type.select_values(['dnn'])
    dnn.add_float_param('learning_rate', 0.01, 3.0)
    dnn.add_int_param('units', 1, 50, index=0)
    dnn.add_int_param('units', 1, 80, index=1)
    dnn.add_categorical_param('activation', ['tanh', 'relu'])
    model_type.select_values(['linear'
                             ]).add_float_param('learning_rate', 0.01, 1.0)

    pytrial = vz.Trial(
        id=1,
        parameters={
            'model_type': vz.ParameterValue(value='dnn'),
            'learning_rate': vz.ParameterValue(value=2.1),
            'units[0]': vz.ParameterValue(value=49),
            'units[1]': vz.ParameterValue(value=79),
            'activation': vz.ParameterValue(value='relu'),
        },
    )
    parameters = py_study_config._pytrial_parameters(pytrial)
    expected = {
        'model_type': 'dnn',
        'learning_rate': 2.1,
        'units': [49, 79],
        'activation': 'relu',
    }
    self.assertEqual(expected, parameters)

  def testTrialToDictRaisesDuplicateParameters(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    trial_proto = study_pb2.Trial()
    trial_proto.id = str(1)
    trial_proto.parameters.add(
        parameter_id='activation', value=struct_pb2.Value(string_value='relu'))
    trial_proto.parameters.add(
        parameter_id='activation', value=struct_pb2.Value(string_value='tanh'))
    trial_proto.parameters.add(
        parameter_id='units', value=struct_pb2.Value(number_value=50))

    with self.assertRaisesRegex(ValueError, 'Invalid trial proto'):
      py_study_config.trial_parameters(trial_proto)

  def testTrialToDictRaisesInvalidTrial(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root
    root.add_float_param('learning_rate', 0.01, 3.0)

    trial_proto = study_pb2.Trial()
    trial_proto.id = str(1)
    trial_proto.state = study_pb2.Trial.State.ACTIVE
    trial_proto.parameters.add(
        parameter_id='foo', value=struct_pb2.Value(number_value=0.5))
    with self.assertRaisesRegex(ValueError,
                                'Invalid trial for this search space'):
      py_study_config.trial_parameters(trial_proto)

  def testTrialToDictWithFinalMetricsSingleObjective(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root
    root.add_float_param('learning_rate', 0.01, 3.0)

    trial_proto = study_pb2.Trial()
    trial_proto.id = str(1)
    trial_proto.state = study_pb2.Trial.State.SUCCEEDED
    trial_proto.parameters.add(
        parameter_id='learning_rate', value=struct_pb2.Value(number_value=0.5))
    trial_proto.final_measurement.step_count = 101
    trial_proto.final_measurement.elapsed_duration.seconds = 67
    trial_proto.final_measurement.metrics.add(metric_id='loss', value=56.8)
    trial_proto.final_measurement.metrics.add(metric_id='objective', value=77.7)

    parameters = py_study_config.trial_parameters(trial_proto)
    self.assertEqual({'learning_rate': 0.5}, parameters)
    metrics = py_study_config.trial_metrics(trial_proto)
    self.assertEqual({'objective': 77.7}, metrics)
    metrics = py_study_config.trial_metrics(
        trial_proto, include_all_metrics=True)
    self.assertEqual({'objective': 77.7, 'loss': 56.8}, metrics)

  def testPyTrialToDictWithFinalMetricsSingleObjective(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root
    root.add_float_param('learning_rate', 0.01, 3.0)

    pytrial = vz.Trial(
        id=1,
        completion_time=datetime.datetime(
            year=2021, month=12, day=2, hour=7, minute=31
        ),
        parameters={'learning_rate': vz.ParameterValue(0.5)},
        final_measurement=vz.Measurement(
            metrics={
                'loss': vz.Metric(value=56.8),
                'objective': vz.Metric(value=77.7),
            },
            elapsed_secs=67,
            steps=101,
        ),
    )
    parameters = py_study_config._pytrial_parameters(pytrial)
    self.assertEqual({'learning_rate': 0.5}, parameters)
    metrics = py_study_config._pytrial_metrics(pytrial)
    self.assertEqual({'objective': 77.7}, metrics)
    metrics = py_study_config._pytrial_metrics(
        pytrial, include_all_metrics=True)
    self.assertEqual({'objective': 77.7, 'loss': 56.8}, metrics)

  def testTrialToDictWithFinalMetricsNotCompleted(self):
    # Throw a Trial that has inconsistent field values.
    # (ACTIVE but has final measurement).
    # Pyvizier fixes the state.
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root
    root.add_float_param('learning_rate', 0.01, 3.0)

    trial_proto = study_pb2.Trial()
    trial_proto.id = str(1)
    trial_proto.state = study_pb2.Trial.State.ACTIVE
    trial_proto.parameters.add(
        parameter_id='learning_rate', value=struct_pb2.Value(number_value=0.5))
    trial_proto.final_measurement.step_count = 101
    trial_proto.final_measurement.elapsed_duration.seconds = 67
    trial_proto.final_measurement.metrics.add(metric_id='loss', value=56.8)
    trial_proto.final_measurement.metrics.add(metric_id='objective', value=77.7)

    parameters = py_study_config.trial_parameters(trial_proto)
    self.assertEqual({'learning_rate': 0.5}, parameters)
    self.assertLen(
        py_study_config.trial_metrics(trial_proto, include_all_metrics=True), 2)

  def testTrialToDictWithFinalMetricsInfeasible(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root
    root.add_float_param('learning_rate', 0.01, 3.0)

    trial_proto = study_pb2.Trial()
    trial_proto.id = str(1)
    trial_proto.state = study_pb2.Trial.State.INFEASIBLE
    trial_proto.parameters.add(
        parameter_id='learning_rate', value=struct_pb2.Value(number_value=0.5))
    trial_proto.final_measurement.step_count = 101
    trial_proto.final_measurement.elapsed_duration.seconds = 67
    trial_proto.final_measurement.metrics.add(metric_id='loss', value=56.8)
    trial_proto.final_measurement.metrics.add(metric_id='objective', value=77.7)

    parameters = py_study_config.trial_parameters(trial_proto)
    self.assertEqual({'learning_rate': 0.5}, parameters)
    self.assertEmpty(py_study_config.trial_metrics(trial_proto))
    self.assertEmpty(
        py_study_config.trial_metrics(trial_proto, include_all_metrics=True))

  def testPyTrialToDictWithFinalMetricsInfeasible(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    root = py_study_config.search_space.root
    root.add_float_param('learning_rate', 0.01, 3.0)

    pytrial = vz.Trial(
        id=1,
        infeasibility_reason='just because',
        completion_time=datetime.datetime(
            year=2021, month=12, day=2, hour=7, minute=31
        ),
        parameters={'learning_rate': vz.ParameterValue(0.5)},
        final_measurement=vz.Measurement(
            metrics={
                'loss': vz.Metric(value=56.8),
                'other': vz.Metric(value=77.7),
            },
            elapsed_secs=67,
            steps=101,
        ),
    )
    parameters = py_study_config._pytrial_parameters(pytrial)
    self.assertEqual({'learning_rate': 0.5}, parameters)
    self.assertEmpty(py_study_config._pytrial_metrics(pytrial))
    self.assertEmpty(
        py_study_config._pytrial_metrics(pytrial, include_all_metrics=True))

  def testTrialToDictWithFinalMetricsMultiObjective(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            ),
            vz.MetricInformation(
                name='objective2', goal=vz.ObjectiveMetricGoal.MINIMIZE
            ),
        ]
    )
    root = py_study_config.search_space.root
    root.add_float_param('learning_rate', 0.01, 3.0)

    trial_proto = study_pb2.Trial(id=str(1))
    trial_proto.state = study_pb2.Trial.State.SUCCEEDED

    trial_proto.parameters.add(
        parameter_id='learning_rate', value=struct_pb2.Value(number_value=0.5))
    trial_proto.final_measurement.step_count = 101
    trial_proto.final_measurement.elapsed_duration.seconds = 67
    trial_proto.final_measurement.metrics.add(metric_id='loss', value=56.8)
    trial_proto.final_measurement.metrics.add(metric_id='objective', value=77.7)
    trial_proto.final_measurement.metrics.add(
        metric_id='objective2', value=-0.2)

    parameters = py_study_config.trial_parameters(trial_proto)
    self.assertEqual({'learning_rate': 0.5}, parameters)
    metrics = py_study_config.trial_metrics(trial_proto)
    self.assertEqual({'objective': 77.7, 'objective2': -0.2}, metrics)
    metrics = py_study_config.trial_metrics(
        trial_proto, include_all_metrics=True)
    self.assertEqual({
        'objective': 77.7,
        'objective2': -0.2,
        'loss': 56.8
    }, metrics)

  def testPyTrialToDictWithFinalMetricsMultiObjective(self):
    py_study_config = vz.StudyConfig(
        metric_information=[
            vz.MetricInformation(
                name='objective', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            ),
            vz.MetricInformation(
                name='objective2', goal=vz.ObjectiveMetricGoal.MINIMIZE
            ),
        ]
    )
    root = py_study_config.search_space.root
    root.add_float_param('learning_rate', 0.01, 3.0)

    pytrial = vz.Trial(
        id=1,
        completion_time=datetime.datetime(
            year=2021, month=12, day=2, hour=7, minute=31
        ),
        parameters={'learning_rate': vz.ParameterValue(0.5)},
        final_measurement=vz.Measurement(
            metrics={
                'loss': vz.Metric(value=56.8),
                'objective': vz.Metric(value=77.7),
                'objective2': vz.Metric(value=-0.2),
            },
            elapsed_secs=67,
            steps=101,
        ),
    )
    parameters = py_study_config._pytrial_parameters(pytrial)
    self.assertEqual({'learning_rate': 0.5}, parameters)
    metrics = py_study_config._pytrial_metrics(pytrial)
    self.assertEqual({'objective': 77.7, 'objective2': -0.2}, metrics)
    metrics = py_study_config._pytrial_metrics(
        pytrial, include_all_metrics=True)
    self.assertEqual({
        'objective': 77.7,
        'objective2': -0.2,
        'loss': 56.8
    }, metrics)

  def testSearchSpacesNotShared(self):
    sc1 = vz.StudyConfig()
    sc1.search_space.root.add_float_param('x', 1, 2)
    sc2 = vz.StudyConfig()
    sc2.search_space.root.add_float_param('x', 1, 2)
    self.assertLen(sc1.search_space.parameters, 1)
    self.assertLen(sc2.search_space.parameters, 1)

  def testHasConditionalParametersFlatSpace(self):
    sc = vz.StudyConfig()
    sc.search_space.root.add_float_param('x', 1, 2)
    self.assertFalse(sc.search_space.is_conditional)

  def testHasConditionalParameters(self):
    sc = vz.StudyConfig()
    root = sc.search_space.root
    model_type = root.add_categorical_param('model_type', ['linear', 'dnn'])
    _ = model_type.select_values(['dnn']).add_float_param(
        'learning_rate',
        0.1,
        1.0,
        default_value=0.001,
        scale_type=vz.ScaleType.LOG,
    )
    self.assertTrue(sc.search_space.is_conditional)


if __name__ == '__main__':
  absltest.main()
