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

"""Tests for proto_converters."""

from absl import logging

import attr

from vizier._src.pyvizier.oss import proto_converters
from vizier._src.pyvizier.pythia import study
from vizier._src.pyvizier.shared import parameter_config as pc
from vizier._src.pyvizier.shared import trial
from vizier._src.service import study_pb2

from google.protobuf import struct_pb2
from google.protobuf import wrappers_pb2
from vizier._src.pyvizier.oss import compare
from absl.testing import absltest
from absl.testing import parameterized

Metric = trial.Metric
Measurement = trial.Measurement


class StudyStateConverterTest(absltest.TestCase):

  def testReversibility(self):
    for original_pyvizier_state in study.StudyState:
      proto_state = proto_converters.StudyStateConverter.to_proto(
          original_pyvizier_state
      )
      py_state = proto_converters.StudyStateConverter.from_proto(proto_state)
      self.assertEqual(original_pyvizier_state, py_state)

  def testStudyStateNotSet(self):
    self.assertEqual(
        proto_converters.StudyStateConverter.from_proto(
            study_pb2.Study.State.STATE_UNSPECIFIED
        ),
        study.StudyState.ACTIVE,
    )


class MeasurementConverterTest(absltest.TestCase):

  def testMeasurementProtoWithEmptyNamedMetric(self):
    proto = study_pb2.Measurement()
    proto.metrics.add(metric_id='', value=0.8)
    measurement = proto_converters.MeasurementConverter.from_proto(proto)
    self.assertEqual(measurement.metrics[''], Metric(value=0.8))

  def testMeasurementCreation(self):
    measurement = Measurement(
        metrics={
            '': Metric(
                value=0
            ),  # The empty metric always exists in Measurement.
            'pr-auc': Metric(value=0.8),
            'latency': Metric(value=32),
        },
        elapsed_secs=12,
        steps=12,
    )
    proto = proto_converters.MeasurementConverter.to_proto(measurement)
    self.assertEqual(
        attr.asdict(proto_converters.MeasurementConverter.from_proto(proto)),
        attr.asdict(measurement),
    )


ParameterValue = trial.ParameterValue


class ParameterValueConverterTest(parameterized.TestCase):

  def testToDoubleProto(self):
    value = ParameterValue(True)
    compare.assertProto2Equal(
        self,
        proto_converters.ParameterValueConverter.to_proto(value, 'aa'),
        study_pb2.Trial.Parameter(
            parameter_id='aa', value=struct_pb2.Value(number_value=1.0)
        ),
    )

  def testToDiscreteProto(self):
    value = ParameterValue(True)
    compare.assertProto2Equal(
        self,
        proto_converters.ParameterValueConverter.to_proto(value, 'aa'),
        study_pb2.Trial.Parameter(
            parameter_id='aa', value=struct_pb2.Value(number_value=1.0)
        ),
    )

  def testToStringProto(self):
    value = ParameterValue('category')
    compare.assertProto2Equal(
        self,
        proto_converters.ParameterValueConverter.to_proto(value, 'aa'),
        study_pb2.Trial.Parameter(
            parameter_id='aa', value=struct_pb2.Value(string_value='category')
        ),
    )

  def testToIntegerProto(self):
    value = ParameterValue(True)
    compare.assertProto2Equal(
        self,
        proto_converters.ParameterValueConverter.to_proto(value, 'aa'),
        study_pb2.Trial.Parameter(
            parameter_id='aa', value=struct_pb2.Value(number_value=1.0)
        ),
    )


class TrialConverterTest(absltest.TestCase):

  def testFromProtoCompleted(self):
    proto = study_pb2.Trial(id=str(1))
    proto.state = study_pb2.Trial.State.SUCCEEDED
    proto.parameters.add(
        parameter_id='float', value=struct_pb2.Value(number_value=1.0)
    )
    proto.parameters.add(
        parameter_id='int', value=struct_pb2.Value(number_value=2)
    )
    proto.parameters.add(
        parameter_id='str', value=struct_pb2.Value(string_value='3')
    )
    proto.final_measurement.metrics.add(metric_id='pr-auc', value=0.8)
    proto.final_measurement.metrics.add(metric_id='latency', value=32)

    proto.start_time.seconds = 1586649600
    proto.end_time.seconds = 1586649600 + 10

    proto.measurements.add(step_count=10)
    proto.measurements[-1].elapsed_duration.seconds = 15
    proto.measurements[-1].metrics.add(metric_id='pr-auc', value=0.7)
    proto.measurements[-1].metrics.add(metric_id='latency', value=42)

    proto.measurements.add(step_count=20)
    proto.measurements[-1].elapsed_duration.seconds = 30
    proto.measurements[-1].metrics.add(metric_id='pr-auc', value=0.75)
    proto.measurements[-1].metrics.add(metric_id='latency', value=37)

    test = proto_converters.TrialConverter.from_proto(proto=proto)
    self.assertEqual(test.id, 1)
    self.assertEqual(test.status, trial.TrialStatus.COMPLETED)
    self.assertTrue(test.is_completed)
    self.assertFalse(test.infeasible)
    self.assertIsNone(test.infeasibility_reason)
    self.assertLen(test.parameters, 3)
    self.assertEqual(test.parameters['float'].value, 1.0)
    self.assertEqual(test.parameters['int'].value, 2)
    self.assertEqual(test.parameters['str'].value, '3')

    # Final measurement
    assert test.final_measurement is not None
    self.assertLen(test.final_measurement.metrics, 2)
    self.assertEqual(test.final_measurement.metrics['pr-auc'].value, 0.8)
    self.assertEqual(test.final_measurement.metrics['latency'].value, 32)

    # Intermediate measurement
    self.assertEqual(
        test.measurements[0],
        trial.Measurement(
            metrics={'pr-auc': 0.7, 'latency': 42}, steps=10, elapsed_secs=15
        ),
    )
    self.assertEqual(
        test.measurements[1],
        trial.Measurement(
            metrics={'pr-auc': 0.75, 'latency': 37}, steps=20, elapsed_secs=30
        ),
    )

    self.assertEqual(test.id, 1)

    self.assertIsNotNone(test.creation_time)
    self.assertIsNotNone(test.completion_time)
    assert test.duration is not None
    self.assertEqual(test.duration.total_seconds(), 10)

    self.assertFalse(test.infeasible)

  def testFromProtoPending(self):
    proto = study_pb2.Trial(id=str(2))
    proto.state = study_pb2.Trial.State.ACTIVE
    proto.start_time.seconds = 1586649600
    test = proto_converters.TrialConverter.from_proto(proto=proto)
    self.assertEqual(test.status, trial.TrialStatus.ACTIVE)
    self.assertFalse(test.is_completed)
    self.assertFalse(test.infeasible)
    self.assertIsNone(test.infeasibility_reason)
    self.assertIsNotNone(test.creation_time)
    self.assertIsNone(test.completion_time)
    self.assertIsNone(test.duration)
    self.assertEmpty(test.metadata)

  def testFromProtoInfeasible(self):
    proto = study_pb2.Trial(id=str(1))
    proto.state = study_pb2.Trial.State.INFEASIBLE
    proto.parameters.add(
        parameter_id='float', value=struct_pb2.Value(number_value=1.0)
    )
    proto.parameters.add(
        parameter_id='int', value=struct_pb2.Value(number_value=2)
    )
    proto.parameters.add(
        parameter_id='str', value=struct_pb2.Value(string_value='3')
    )
    proto.start_time.seconds = 1586649600
    proto.end_time.seconds = 1586649600 + 10
    proto.infeasible_reason = 'A reason'

    test = proto_converters.TrialConverter.from_proto(proto=proto)
    self.assertEqual(test.status, trial.TrialStatus.COMPLETED)
    self.assertTrue(test.is_completed)
    self.assertTrue(test.infeasible)
    self.assertEqual(test.infeasibility_reason, 'A reason')

  def testFromProtoInvalidTrial(self):
    proto = study_pb2.Trial(id=str(2))
    proto.parameters.add(
        parameter_id='float', value=struct_pb2.Value(number_value=1.0)
    )
    proto.parameters.add(
        parameter_id='float', value=struct_pb2.Value(number_value=2.0)
    )
    proto.state = study_pb2.Trial.State.ACTIVE
    proto.start_time.seconds = 1586649600
    with self.assertRaisesRegex(ValueError, 'Invalid trial proto'):
      proto_converters.TrialConverter.from_proto(proto=proto)

  def testFromProtoMetadata(self):
    proto = study_pb2.Trial(id=str(1))
    proto.state = study_pb2.Trial.ACTIVE
    proto.parameters.add(
        parameter_id='float', value=struct_pb2.Value(number_value=1.0)
    )
    proto.metadata.add(key='key0', ns='x', value='namespace=x0')
    proto.metadata.add(key='key1', ns='x', value='namespace=x1')
    proto.metadata.add(key='key1', ns='', value='gets overwritten')
    proto.metadata.add(key='key1', value='second value takes priority')
    logging.info('PROTO:: %s', proto)
    added1 = proto.metadata.add(key='proto')
    added1.proto.Pack(study_pb2.Trial(id=str(999)))
    added2 = proto.metadata.add(key='proto', ns='t')
    added2.proto.Pack(study_pb2.Trial(id=str(991)))
    test = proto_converters.TrialConverter.from_proto(proto=proto)
    logging.info('TEST:: %s', repr(test.metadata))
    logging.info('TEST-x:: %s', repr(test.metadata.ns('x')))
    logging.info('TEST-t:: %s', repr(test.metadata.ns('t')))
    logging.info('TEST-namespaces: %s', test.metadata.namespaces())
    logging.info('test.ns("x"):: %s', test.metadata.ns('x'))
    self.assertEqual(test.metadata['key1'], 'second value takes priority')
    self.assertEqual(test.metadata.abs_ns(['x'])['key0'], 'namespace=x0')
    self.assertEqual(test.metadata.abs_ns(['x'])['key1'], 'namespace=x1')
    self.assertEqual(
        test.metadata.get_proto('proto', cls=study_pb2.Trial),
        study_pb2.Trial(id=str(999)),
    )
    self.assertEqual(
        test.metadata.abs_ns(['t']).get_proto('proto', cls=study_pb2.Trial),
        study_pb2.Trial(id=str(991)),
    )


class TrialConverterToProtoTest(absltest.TestCase):
  """Tests for TrialConverter.to_proto()."""

  def _GetSingleObjectiveBaseTrial(self):
    proto = study_pb2.Trial(
        name='owners/my_username/studies/cifar_10',
        id=str(2),
        client_id='worker0',
    )
    proto.parameters.add(
        parameter_id='activation', value=struct_pb2.Value(string_value='relu')
    )
    proto.parameters.add(
        parameter_id='synchronus', value=struct_pb2.Value(string_value='true')
    )
    proto.parameters.add(
        parameter_id='batch_size', value=struct_pb2.Value(number_value=32)
    )
    proto.parameters.add(
        parameter_id='floating_point_param',
        value=struct_pb2.Value(number_value=32.0),
    )
    proto.parameters.add(
        parameter_id='learning_rate', value=struct_pb2.Value(number_value=0.5)
    )
    proto.parameters.add(
        parameter_id='units', value=struct_pb2.Value(number_value=50)
    )
    proto.start_time.seconds = 1630505100
    return proto

  def testParameterBackToBackConversion(self):
    proto = self._GetSingleObjectiveBaseTrial()
    proto.state = study_pb2.Trial.State.ACTIVE
    pytrial = proto_converters.TrialConverter.from_proto(proto)
    got = proto_converters.TrialConverter.to_proto(pytrial)
    compare.assertProto2Equal(self, proto, got)

  def testFinalMeasurementBackToBackConversion(self):
    proto = study_pb2.Trial(id=str(1), state=study_pb2.Trial.State.SUCCEEDED)
    proto.start_time.seconds = 12456
    proto.end_time.seconds = 12456 + 10
    proto.parameters.add(
        parameter_id='learning_rate', value=struct_pb2.Value(number_value=0.5)
    )
    proto.final_measurement.step_count = 101
    proto.final_measurement.elapsed_duration.seconds = 67
    proto.final_measurement.metrics.add(metric_id='loss', value=56.8)
    proto.final_measurement.metrics.add(metric_id='objective', value=77.7)
    proto.final_measurement.metrics.add(metric_id='objective2', value=-0.2)

    pytrial = proto_converters.TrialConverter.from_proto(proto)
    got = proto_converters.TrialConverter.to_proto(pytrial)
    compare.assertProto2SameElements(self, proto, got, number_matters=True)

  def testMeasurementBackToBackConversion(self):
    proto = study_pb2.Trial(
        id=str(2), state=study_pb2.Trial.State.ACTIVE, client_id='worker0'
    )
    proto.start_time.seconds = 1630505100
    proto.measurements.add(step_count=123)
    proto.measurements[-1].elapsed_duration.seconds = 22
    proto.measurements[-1].metrics.add(metric_id='objective', value=0.4321)
    proto.measurements[-1].metrics.add(metric_id='loss', value=0.001)

    proto.measurements.add(step_count=789)
    proto.measurements[-1].elapsed_duration.seconds = 55
    proto.measurements[-1].metrics.add(metric_id='objective', value=0.21)
    proto.measurements[-1].metrics.add(metric_id='loss', value=0.0001)

    pytrial = proto_converters.TrialConverter.from_proto(proto)
    got = proto_converters.TrialConverter.to_proto(pytrial)
    compare.assertProto2SameElements(self, proto, got, number_matters=True)


class ParameterConfigConverterToProtoTest(absltest.TestCase):
  """Tests for ParameterConfigConverter.to_proto()."""

  def testDiscreteConfigToProto(self):
    feasible_values = (-1, 3, 2)
    parameter_config = pc.ParameterConfig.factory(
        'name',
        feasible_values=feasible_values,
        scale_type=pc.ScaleType.LOG,
        default_value=2,
    )

    proto = proto_converters.ParameterConfigConverter.to_proto(parameter_config)
    self.assertEqual(proto.parameter_id, 'name')
    self.assertEqual(proto.discrete_value_spec.values, [-1.0, 2.0, 3.0])
    self.assertEqual(proto.discrete_value_spec.default_value.value, 2)
    self.assertEqual(
        proto.scale_type,
        study_pb2.StudySpec.ParameterSpec.ScaleType.UNIT_LOG_SCALE,
    )


class ParameterConfigConverterFromProtoTest(absltest.TestCase):
  """Tests for ParameterConfigConverter.from_proto()."""

  def testCreatesFromGoodProto(self):
    proto = study_pb2.StudySpec.ParameterSpec(
        parameter_id='name',
        discrete_value_spec=study_pb2.StudySpec.ParameterSpec.DiscreteValueSpec(
            values=[1.0, 2.0, 3.0],
            default_value=wrappers_pb2.DoubleValue(value=2.0),
        ),
    )
    parameter_config = proto_converters.ParameterConfigConverter.from_proto(
        proto
    )
    self.assertEqual(parameter_config.name, proto.parameter_id)
    self.assertEqual(parameter_config.type, pc.ParameterType.DISCRETE)
    self.assertEqual(parameter_config.bounds, (1.0, 3.0))
    self.assertEqual(parameter_config.feasible_values, [1.0, 2.0, 3.0])
    self.assertEqual(parameter_config.default_value, 2.0)


if __name__ == '__main__':
  absltest.main()
