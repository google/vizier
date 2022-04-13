"""Tests for vizier.pyvizier.oss.study_config."""

import datetime

from vizier.service import pyvizier
from vizier.service import study_pb2

from google.protobuf import struct_pb2
from vizier._src.pyvizier.oss import compare
from absl.testing import absltest


class StudyConfigTest(absltest.TestCase):

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

  def testCreationFromAndToProtoModernStudy(self):
    expected_automated_stopping_config = study_pb2.StudySpec.DecayCurveAutomatedStoppingSpec(
        use_elapsed_duration=False)

    study_config_proto = study_pb2.StudySpec(
        metrics=[
            study_pb2.StudySpec.MetricSpec(
                metric_id='pr-auc',
                goal=study_pb2.StudySpec.MetricSpec.GoalType.MAXIMIZE)
        ],
        decay_curve_stopping_spec=expected_automated_stopping_config)

    study_config_proto.parameters.extend(self.pconfigs)
    # Test all proprties.
    sc = pyvizier.StudyConfig.from_proto(study_config_proto)
    expected = pyvizier.MetricsConfig([
        pyvizier.MetricInformation(
            name='pr-auc', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    self.assertEqual(sc.metric_information, expected)
    self.assertEqual(sc.single_objective_metric_name, 'pr-auc')
    self.assertTrue(sc.is_single_objective)

    compare.assertProto2Equal(self, expected_automated_stopping_config,
                              sc.automated_stopping_config.to_proto())
    compare.assertProto2Equal(self, study_config_proto, sc.to_proto())

  def testCreationFromAndToProtoModernMultiObjectiveStudy(self):
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
    sc = pyvizier.StudyConfig.from_proto(study_config_proto)

    expected = pyvizier.MetricsConfig([
        pyvizier.MetricInformation(
            name='loss', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE),
        pyvizier.MetricInformation(
            name='pr-auc', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE),
    ])
    self.assertEqual(sc.metric_information, expected)
    self.assertIsNone(sc.single_objective_metric_name)
    self.assertFalse(sc.is_single_objective)
    compare.assertProto2SameElements(self, study_config_proto, sc.to_proto())

  def testCreationFromProtoNoGoalRaises(self):
    study_config_proto = study_pb2.StudySpec()

    sc = pyvizier.StudyConfig.from_proto(study_config_proto)
    self.assertEmpty(sc.metric_information)

  def testMetadata(self):
    empty_trial = study_pb2.Trial(id=str(1))
    sc = pyvizier.StudyConfig()
    sc.metric_information.append(
        pyvizier.MetricInformation(
            name='pr-auc', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE))
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
    sc = pyvizier.StudyConfig()
    sc.algorithm = pyvizier.Algorithm.RANDOM_SEARCH
    sc.metric_information.append(
        pyvizier.MetricInformation(
            name='pr-auc', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE))
    root = sc.search_space.select_root()
    root.add_float_param(
        'learning_rate', 0.00001, 1.0, scale_type=pyvizier.ScaleType.LINEAR)
    root.add_categorical_param('optimizer', ['adagrad', 'adam', 'experimental'])

    sc.automated_stopping_config = pyvizier.AutomatedStoppingConfig.decay_curve_stopping_config(
        use_steps=True)

    # Test all proprties.
    self.assertEqual(sc.algorithm.value, study_pb2.StudySpec.RANDOM_SEARCH)
    expected = pyvizier.MetricsConfig([
        pyvizier.MetricInformation(
            name='pr-auc', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    self.assertEqual(sc.metric_information, expected)
    self.assertEqual(sc.single_objective_metric_name, 'pr-auc')
    self.assertTrue(sc.is_single_objective)

    expected = study_pb2.StudySpec(
        algorithm=study_pb2.StudySpec.RANDOM_SEARCH,
        metrics=[
            study_pb2.StudySpec.MetricSpec(
                metric_id='pr-auc',
                goal=study_pb2.StudySpec.MetricSpec.GoalType.MAXIMIZE)
        ],
        decay_curve_stopping_spec=study_pb2.StudySpec
        .DecayCurveAutomatedStoppingSpec(use_elapsed_duration=False),
        observation_noise=study_pb2.StudySpec.ObservationNoise
        .OBSERVATION_NOISE_UNSPECIFIED,
    )
    expected.parameters.extend(self.pconfigs)
    compare.assertProto2Equal(self, expected, sc.to_proto())

  @absltest.skip('???')
  def testTrialToDict(self):
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()
    root.add_float_param('learning_rate', 0.01, 3.0)
    root.add_int_param('units', 10, 1000, scale_type=pyvizier.ScaleType.LOG)
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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()
    root.add_float_param('learning_rate', 0.01, 3.0)
    root.add_int_param('units', 10, 1000, scale_type=pyvizier.ScaleType.LOG)
    root.add_discrete_param('batch_size', [8, 16, 32])
    root.add_discrete_param(
        'floating_point_param', [8., 16., 32.], auto_cast=False)
    root.add_categorical_param('activation', ['tanh', 'relu'])
    root.add_bool_param('synchronous')

    pytrial = pyvizier.Trial(id=1)
    pytrial.parameters = {
        'activation': pyvizier.ParameterValue(value='relu'),
        'synchronous': pyvizier.ParameterValue(value=True),
        'batch_size': pyvizier.ParameterValue(value=32),
        'floating_point_param': pyvizier.ParameterValue(value=32.),
        'learning_rate': pyvizier.ParameterValue(value=0.5),
        'units': pyvizier.ParameterValue(value=50)
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

    py_study_config = pyvizier.StudyConfig.from_proto(proto)
    self.assertEqual(py_study_config.observation_noise,
                     pyvizier.ObservationNoise.HIGH)
    parameters = py_study_config.trial_parameters(trial_proto)
    self.assertEqual(py_study_config.observation_noise,
                     pyvizier.ObservationNoise.HIGH)
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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()
    for index in (0, 1):
      root.add_float_param('learning_rate', 0.01, 3.0, index=index)
      root.add_int_param(
          'units', 10, 1000, scale_type=pyvizier.ScaleType.LOG, index=index)
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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()
    for index in (0, 1):
      root.add_float_param('learning_rate', 0.01, 3.0, index=index)
      root.add_int_param(
          'units', 10, 1000, scale_type=pyvizier.ScaleType.LOG, index=index)
      root.add_categorical_param('activation', ['tanh', 'relu'], index=index)
      root.add_bool_param('synchronous', index=index)
      root.add_discrete_param('batch_size', [8, 16, 32], index=index)
    root.add_discrete_param(
        'floating_point_param', [8., 16., 32.], auto_cast=False)

    pytrial = pyvizier.Trial(id=2)
    pytrial.parameters = {
        'learning_rate[0]': pyvizier.ParameterValue(value=0.5),
        'learning_rate[1]': pyvizier.ParameterValue(value=0.1),
        'units[0]': pyvizier.ParameterValue(value=50),
        'units[1]': pyvizier.ParameterValue(value=200),
        'activation[0]': pyvizier.ParameterValue(value='relu'),
        'activation[1]': pyvizier.ParameterValue(value='relu'),
        'synchronous[0]': pyvizier.ParameterValue(value=True),
        'synchronous[1]': pyvizier.ParameterValue(value=False),
        'batch_size[0]': pyvizier.ParameterValue(value=32.0),
        'batch_size[1]': pyvizier.ParameterValue(value=8.0),
        'floating_point_param': pyvizier.ParameterValue(value=16.)
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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()
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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()

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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()

    model_type = root.add_categorical_param('model_type', ['dnn', 'linear'])
    dnn = model_type.select_values(['dnn'])
    dnn.add_float_param('learning_rate', 0.01, 3.0)
    dnn.add_int_param('units', 1, 50, index=0)
    dnn.add_int_param('units', 1, 80, index=1)
    dnn.add_categorical_param('activation', ['tanh', 'relu'])
    model_type.select_values(['linear'
                             ]).add_float_param('learning_rate', 0.01, 1.0)

    pytrial = pyvizier.Trial(
        id=1,
        parameters={
            'model_type': pyvizier.ParameterValue(value='dnn'),
            'learning_rate': pyvizier.ParameterValue(value=2.1),
            'units[0]': pyvizier.ParameterValue(value=49),
            'units[1]': pyvizier.ParameterValue(value=79),
            'activation': pyvizier.ParameterValue(value='relu'),
        })
    parameters = py_study_config._pytrial_parameters(pytrial)
    expected = {
        'model_type': 'dnn',
        'learning_rate': 2.1,
        'units': [49, 79],
        'activation': 'relu',
    }
    self.assertEqual(expected, parameters)

  def testTrialToDictRaisesDuplicateParameters(self):
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()
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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()
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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()
    root.add_float_param('learning_rate', 0.01, 3.0)

    pytrial = pyvizier.Trial(
        id=1,
        completion_time=datetime.datetime(
            year=2021, month=12, day=2, hour=7, minute=31),
        parameters={'learning_rate': pyvizier.ParameterValue(0.5)},
        final_measurement=pyvizier.Measurement(
            metrics={
                'loss': pyvizier.Metric(value=56.8),
                'objective': pyvizier.Metric(value=77.7)
            },
            elapsed_secs=67,
            steps=101))
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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()
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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()
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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE)
    ])
    root = py_study_config.search_space.select_root()
    root.add_float_param('learning_rate', 0.01, 3.0)

    pytrial = pyvizier.Trial(
        id=1,
        infeasibility_reason='just because',
        completion_time=datetime.datetime(
            year=2021, month=12, day=2, hour=7, minute=31),
        parameters={'learning_rate': pyvizier.ParameterValue(0.5)},
        final_measurement=pyvizier.Measurement(
            metrics={
                'loss': pyvizier.Metric(value=56.8),
                'other': pyvizier.Metric(value=77.7)
            },
            elapsed_secs=67,
            steps=101))
    parameters = py_study_config._pytrial_parameters(pytrial)
    self.assertEqual({'learning_rate': 0.5}, parameters)
    self.assertEmpty(py_study_config._pytrial_metrics(pytrial))
    self.assertEmpty(
        py_study_config._pytrial_metrics(pytrial, include_all_metrics=True))

  def testTrialToDictWithFinalMetricsMultiObjective(self):
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE),
        pyvizier.MetricInformation(
            name='objective2', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE)
    ])
    root = py_study_config.search_space.select_root()
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
    py_study_config = pyvizier.StudyConfig(metric_information=[
        pyvizier.MetricInformation(
            name='objective', goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE),
        pyvizier.MetricInformation(
            name='objective2', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE)
    ])
    root = py_study_config.search_space.select_root()
    root.add_float_param('learning_rate', 0.01, 3.0)

    pytrial = pyvizier.Trial(
        id=1,
        completion_time=datetime.datetime(
            year=2021, month=12, day=2, hour=7, minute=31),
        parameters={'learning_rate': pyvizier.ParameterValue(0.5)},
        final_measurement=pyvizier.Measurement(
            metrics={
                'loss': pyvizier.Metric(value=56.8),
                'objective': pyvizier.Metric(value=77.7),
                'objective2': pyvizier.Metric(value=-0.2)
            },
            elapsed_secs=67,
            steps=101))
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
    sc1 = pyvizier.StudyConfig()
    sc1.search_space.select_root().add_float_param('x', 1, 2)
    sc2 = pyvizier.StudyConfig()
    sc2.search_space.select_root().add_float_param('x', 1, 2)
    self.assertLen(sc1.search_space.parameters, 1)
    self.assertLen(sc2.search_space.parameters, 1)

  def testHasConditionalParametersFlatSpace(self):
    sc = pyvizier.StudyConfig()
    sc.search_space.select_root().add_float_param('x', 1, 2)
    self.assertFalse(sc.search_space.is_conditional)

  def testHasConditionalParameters(self):
    sc = pyvizier.StudyConfig()
    root = sc.search_space.select_root()
    model_type = root.add_categorical_param('model_type', ['linear', 'dnn'])
    _ = model_type.select_values(['dnn']).add_float_param(
        'learning_rate',
        0.1,
        1.0,
        default_value=0.001,
        scale_type=pyvizier.ScaleType.LOG)
    self.assertTrue(sc.search_space.is_conditional)


if __name__ == '__main__':
  absltest.main()
