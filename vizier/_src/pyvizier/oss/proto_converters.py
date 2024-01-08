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

"""Converters for OSS Vizier's protos from/to PyVizier's classes."""

import datetime
import logging
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from absl import logging
from vizier._src.pythia import policy
from vizier._src.pyvizier.oss import metadata_util
from vizier._src.pyvizier.pythia import study
from vizier._src.pyvizier.shared import base_study_config
from vizier._src.pyvizier.shared import common
from vizier._src.pyvizier.shared import parameter_config
from vizier._src.pyvizier.shared import trial
from vizier._src.service import pythia_service_pb2
from vizier._src.service import study_pb2
from vizier._src.service import vizier_service_pb2


ScaleType = parameter_config.ScaleType
_ScaleTypePb2 = study_pb2.StudySpec.ParameterSpec.ScaleType
ParameterType = parameter_config.ParameterType
MonotypeParameterSequence = parameter_config.MonotypeParameterSequence


class StudyStateConverter:
  """Proto converter for Study states."""

  _pyvizier_to_proto = {
      study.StudyState.ACTIVE: study_pb2.Study.State.ACTIVE,
      study.StudyState.ABORTED: study_pb2.Study.State.INACTIVE,
      study.StudyState.COMPLETED: study_pb2.Study.State.COMPLETED,
  }
  _proto_to_pyvizier = {v: k for k, v in _pyvizier_to_proto.items()}

  @classmethod
  def to_proto(cls, state: study.StudyState) -> study_pb2.Study.State:
    if state in cls._pyvizier_to_proto:
      return cls._pyvizier_to_proto[state]
    return study_pb2.Study.State.STATE_UNSPECIFIED

  @classmethod
  def from_proto(cls, proto: study_pb2.Study.State) -> study.StudyState:
    if proto in cls._proto_to_pyvizier:
      return cls._proto_to_pyvizier[proto]
    elif proto == study_pb2.Study.State.STATE_UNSPECIFIED:
      # OSS Vizier server treats STATE_UNSPECIFIED as ACTIVE.
      return study.StudyState.ACTIVE
    else:
      raise ValueError(
          'Proto Study state {} has no equivalent in PyVizier.'.format(
              study_pb2.Study.State.Name(proto)
          )
      )


class _ScaleTypeMap:
  """Proto converter for scale type."""

  _pyvizier_to_proto = {
      ScaleType.LINEAR: _ScaleTypePb2.UNIT_LINEAR_SCALE,
      ScaleType.LOG: _ScaleTypePb2.UNIT_LOG_SCALE,
      ScaleType.REVERSE_LOG: _ScaleTypePb2.UNIT_REVERSE_LOG_SCALE,
  }
  _proto_to_pyvizier = {v: k for k, v in _pyvizier_to_proto.items()}

  @classmethod
  def to_proto(cls, pyvizier: ScaleType) -> _ScaleTypePb2:
    return cls._pyvizier_to_proto[pyvizier]

  @classmethod
  def from_proto(cls, proto: _ScaleTypePb2) -> ScaleType:
    return cls._proto_to_pyvizier[proto]


class ParameterConfigConverter:
  """Converter for ParameterConfig."""

  @classmethod
  def _set_bounds(
      cls,
      proto: study_pb2.StudySpec.ParameterSpec,
      lower: float,
      upper: float,
      parameter_type: ParameterType,
  ):
    """Sets the proto's min_value and max_value fields."""
    if parameter_type == ParameterType.INTEGER:
      proto.integer_value_spec.min_value = lower
      proto.integer_value_spec.max_value = upper
    elif parameter_type == ParameterType.DOUBLE:
      proto.double_value_spec.min_value = lower
      proto.double_value_spec.max_value = upper

  @classmethod
  def _set_feasible_points(
      cls,
      proto: study_pb2.StudySpec.ParameterSpec,
      feasible_points: Sequence[float],
  ):
    """Sets the proto's feasible_points field."""
    feasible_points = sorted(feasible_points)
    proto.discrete_value_spec.ClearField('values')
    proto.discrete_value_spec.values.extend(feasible_points)

  @classmethod
  def _set_categories(
      cls, proto: study_pb2.StudySpec.ParameterSpec, categories: Sequence[str]
  ):
    """Sets the protos' categories field."""
    proto.categorical_value_spec.ClearField('values')
    proto.categorical_value_spec.values.extend(categories)

  @classmethod
  def _set_default_value(
      cls,
      proto: study_pb2.StudySpec.ParameterSpec,
      default_value: Union[float, int, str],
  ):
    """Sets the protos' default_value field."""
    which_pv_spec = proto.WhichOneof('parameter_value_spec')
    getattr(proto, which_pv_spec).default_value.value = default_value

  @classmethod
  def _matching_parent_values(
      cls, proto: study_pb2.StudySpec.ParameterSpec.ConditionalParameterSpec
  ) -> MonotypeParameterSequence:
    """Returns the matching parent values, if set."""
    oneof_name = proto.WhichOneof('parent_value_condition')
    if not oneof_name:
      return []
    if oneof_name in (
        'parent_discrete_values',
        'parent_int_values',
        'parent_categorical_values',
    ):
      return list(getattr(getattr(proto, oneof_name), 'values'))
    raise ValueError('Unknown matching_parent_vals: {}'.format(oneof_name))

  @classmethod
  def from_proto(
      cls,
      proto: study_pb2.StudySpec.ParameterSpec,
      *,
      strict_validation: bool = False,
  ) -> parameter_config.ParameterConfig:
    """Creates a ParameterConfig.

    Args:
      proto:
      strict_validation: If True, raise ValueError to enforce that
        from_proto(proto).to_proto == proto.

    Returns:
      ParameterConfig object

    Raises:
      ValueError: See the "strict_validtion" arg documentation.
    """
    feasible_values = []
    oneof_name = proto.WhichOneof('parameter_value_spec')
    if oneof_name == 'integer_value_spec':
      bounds = (
          int(proto.integer_value_spec.min_value),
          int(proto.integer_value_spec.max_value),
      )
    elif oneof_name == 'double_value_spec':
      bounds = (
          proto.double_value_spec.min_value,
          proto.double_value_spec.max_value,
      )
    elif oneof_name == 'discrete_value_spec':
      bounds = None
      feasible_values = proto.discrete_value_spec.values
    elif oneof_name == 'categorical_value_spec':
      bounds = None
      feasible_values = proto.categorical_value_spec.values

    default_value = None
    if getattr(proto, oneof_name).default_value.value:
      default_value = getattr(proto, oneof_name).default_value.value

    if proto.conditional_parameter_specs:
      children = []
      for conditional_ps in proto.conditional_parameter_specs:
        parent_values = cls._matching_parent_values(conditional_ps)
        children.append(
            (parent_values, cls.from_proto(conditional_ps.parameter_spec))
        )
    else:
      children = None

    scale_type = None
    if proto.scale_type:
      scale_type = _ScaleTypeMap.from_proto(proto.scale_type)

    try:
      config = parameter_config.ParameterConfig.factory(
          name=proto.parameter_id,
          feasible_values=feasible_values,
          bounds=bounds,
          children=children,
          scale_type=scale_type,
          default_value=default_value,
      )
    except ValueError as e:
      raise ValueError(
          'The provided proto was misconfigured. {}'.format(proto)
      ) from e

    if strict_validation and cls.to_proto(config) != proto:
      raise ValueError(
          'The provided proto was misconfigured. Expected: {} Given: {}'.format(
              cls.to_proto(config), proto
          )
      )
    return config

  @classmethod
  def _set_child_parameter_configs(
      cls,
      parent_proto: study_pb2.StudySpec.ParameterSpec,
      pc: parameter_config.ParameterConfig,
  ):
    """Sets the parent_proto's conditional_parameter_specs field.

    Args:
      parent_proto: Modified in place.
      pc: Parent ParameterConfig to copy children from.

    Raises:
      ValueError: If the child configs are invalid
    """
    children: List[
        Tuple[MonotypeParameterSequence, parameter_config.ParameterConfig]
    ] = []
    for child in pc.child_parameter_configs:
      children.append((child.matching_parent_values, child))
    if not children:
      return

    parent_proto.ClearField('conditional_parameter_specs')
    for child_pair in children:
      if len(child_pair) != 2:
        raise ValueError(
            """Each element in children must be a tuple of
            (Sequence of valid parent values,  ParameterConfig)"""
        )

    logging.debug(
        '_set_child_parameter_configs: parent_proto=%s, children=%s',
        parent_proto,
        children,
    )
    for unsorted_parent_values, child in children:
      parent_values = sorted(unsorted_parent_values)
      child_proto = cls.to_proto(child.clone_without_children)
      conditional_parameter_spec = (
          study_pb2.StudySpec.ParameterSpec.ConditionalParameterSpec(
              parameter_spec=child_proto
          )
      )

      if parent_proto.HasField('discrete_value_spec'):
        conditional_parameter_spec.parent_discrete_values.values[:] = (
            parent_values
        )
      elif parent_proto.HasField('categorical_value_spec'):
        conditional_parameter_spec.parent_categorical_values.values[:] = (
            parent_values
        )
      elif parent_proto.HasField('integer_value_spec'):
        conditional_parameter_spec.parent_int_values.values[:] = parent_values
      else:
        raise ValueError('DOUBLE type cannot have child parameters')
      if child.child_parameter_configs:
        cls._set_child_parameter_configs(child_proto, child)
      parent_proto.conditional_parameter_specs.extend(
          [conditional_parameter_spec]
      )

  @classmethod
  def to_proto(
      cls, pc: parameter_config.ParameterConfig
  ) -> study_pb2.StudySpec.ParameterSpec:
    """Returns a ParameterConfig Proto."""
    proto = study_pb2.StudySpec.ParameterSpec(parameter_id=pc.name)
    if pc.type == ParameterType.DISCRETE:
      cls._set_feasible_points(proto, [float(v) for v in pc.feasible_values])
    elif pc.type == ParameterType.CATEGORICAL:
      cls._set_categories(proto, pc.feasible_values)
    elif pc.type in (ParameterType.INTEGER, ParameterType.DOUBLE):
      cls._set_bounds(proto, pc.bounds[0], pc.bounds[1], pc.type)
    else:
      raise ValueError('Invalid ParameterConfig: {}'.format(pc))
    if (
        pc.scale_type is not None
        and pc.scale_type != ScaleType.UNIFORM_DISCRETE
    ):
      proto.scale_type = _ScaleTypeMap.to_proto(pc.scale_type)
    if pc.default_value is not None:
      cls._set_default_value(proto, pc.default_value)

    cls._set_child_parameter_configs(proto, pc)
    return proto


class ParameterValueConverter:
  """Converter for vz.ParameterValue."""

  @classmethod
  def from_proto(
      cls, proto: study_pb2.Trial.Parameter
  ) -> Optional[trial.ParameterValue]:
    """Returns whichever value that is populated, or None."""
    value_proto = proto.value
    oneof_name = value_proto.WhichOneof('kind')
    potential_value = getattr(value_proto, oneof_name)
    if (
        isinstance(potential_value, float)
        or isinstance(potential_value, str)
        or isinstance(potential_value, bool)
    ):
      return trial.ParameterValue(potential_value)
    else:
      return None

  @classmethod
  def to_proto(
      cls, parameter_value: trial.ParameterValue, name: str
  ) -> study_pb2.Trial.Parameter:
    """Returns Parameter Proto."""
    proto = study_pb2.Trial.Parameter(parameter_id=name)

    if isinstance(parameter_value.value, int):
      proto.value.number_value = parameter_value.value
    elif isinstance(parameter_value.value, bool):
      proto.value.bool_value = parameter_value.value
    elif isinstance(parameter_value.value, float):
      proto.value.number_value = parameter_value.value
    elif isinstance(parameter_value.value, str):
      proto.value.string_value = parameter_value.value

    return proto


class MeasurementConverter:
  """Converter for vz.Measurement."""

  @classmethod
  def from_proto(cls, proto: study_pb2.Measurement) -> trial.Measurement:
    """Creates a valid instance from proto.

    Args:
      proto: Measurement proto.

    Returns:
      A valid instance of Measurement object. Metrics with invalid values
      are automatically filtered out.
    """

    metrics = dict()

    for metric in proto.metrics:
      if (
          metric.metric_id in metrics
          and metrics[metric.metric_id].value != metric.value
      ):
        logging.log_first_n(
            logging.ERROR,
            (
                'Duplicate metric of name "%s".'
                'The newly found value %s will be used and '
                'the previously found value %s will be discarded.'
                'This always happens if the proto has an empty-named metric.'
            ),
            5,
            metric.metric_id,
            metric.value,
            metrics[metric.metric_id].value,
        )
      try:
        metrics[metric.metric_id] = trial.Metric(value=metric.value)
      except ValueError:
        pass
    return trial.Measurement(
        metrics=metrics,
        elapsed_secs=proto.elapsed_duration.seconds,
        steps=proto.step_count,
    )

  @classmethod
  def to_proto(cls, measurement: trial.Measurement) -> study_pb2.Measurement:
    """Converts to Measurement proto."""
    proto = study_pb2.Measurement()
    for name, metric in measurement.metrics.items():
      proto.metrics.add(metric_id=name, value=metric.value)

    proto.step_count = measurement.steps
    int_seconds = int(measurement.elapsed_secs)
    proto.elapsed_duration.seconds = int_seconds
    proto.elapsed_duration.nanos = int(
        1e9 * (measurement.elapsed_secs - int_seconds)
    )
    return proto


class MetricInformationConverter:
  """A converter to/from study_pb2.StudySpec.MetricInformation."""

  @classmethod
  def from_proto(
      cls, proto: study_pb2.StudySpec.MetricSpec
  ) -> base_study_config.MetricInformation:
    """Converts a MetricInformation proto to a MetricInformation object."""
    if proto.goal not in list(base_study_config.ObjectiveMetricGoal):
      raise ValueError('Unknown MetricInformation.goal: {}'.format(proto.goal))

    safety_threshold = None
    desired_min_safe_trials_fraction = None

    if proto.HasField('safety_config'):
      safety_threshold = proto.safety_config.safety_threshold
    if proto.safety_config.HasField('desired_min_safe_trials_fraction'):
      desired_min_safe_trials_fraction = (
          proto.safety_config.desired_min_safe_trials_fraction
      )

    return base_study_config.MetricInformation(
        name=proto.metric_id,
        goal=proto.goal,
        safety_threshold=safety_threshold,
        desired_min_safe_trials_fraction=desired_min_safe_trials_fraction,
        min_value=None,
        max_value=None,
    )

  @classmethod
  def to_proto(
      cls, obj: base_study_config.MetricInformation
  ) -> study_pb2.StudySpec.MetricSpec:
    """Returns this object as a proto."""

    proto = study_pb2.StudySpec.MetricSpec(
        metric_id=obj.name, goal=obj.goal.value
    )

    if obj.type == base_study_config.MetricType.SAFETY:
      proto.safety_config.safety_threshold = obj.safety_threshold
      if obj.desired_min_safe_trials_fraction is not None:
        proto.safety_config.desired_min_safe_trials_fraction = (
            obj.desired_min_safe_trials_fraction
        )
    return proto


class SearchSpaceConverter:
  """A wrapper for study_pb2.StudySpec."""

  @classmethod
  def from_proto(
      cls, proto: study_pb2.StudySpec
  ) -> parameter_config.SearchSpace:
    """Extracts a SearchSpace object from a StudyConfig proto."""
    space = parameter_config.SearchSpace()
    for pc in proto.parameters:
      space.add(ParameterConfigConverter.from_proto(pc))
    return space

  @classmethod
  def parameter_protos(
      cls, obj: parameter_config.SearchSpace
  ) -> List[study_pb2.StudySpec.ParameterSpec]:
    """Returns the search space as a List of ParameterConfig protos."""
    return [ParameterConfigConverter.to_proto(pc) for pc in obj.parameters]


class MetricsConfigConverter:
  """A wrapper for study_pb2.StudySpec.MetricSpec's."""

  @classmethod
  def from_protos(
      cls, protos: Iterable[study_pb2.StudySpec.MetricSpec]
  ) -> base_study_config.MetricsConfig:
    return base_study_config.MetricsConfig(
        [MetricInformationConverter.from_proto(m) for m in protos]
    )

  @classmethod
  def to_protos(
      cls, obj: base_study_config.MetricsConfig
  ) -> List[study_pb2.StudySpec.MetricSpec]:
    return [MetricInformationConverter.to_proto(metric) for metric in obj]


def _to_pyvizier_trial_status(
    proto_state: study_pb2.Trial.State,
) -> trial.TrialStatus:
  """from_proto conversion for Trial statuses."""
  if proto_state == study_pb2.Trial.State.REQUESTED:
    return trial.TrialStatus.REQUESTED
  elif proto_state == study_pb2.Trial.State.ACTIVE:
    return trial.TrialStatus.ACTIVE
  if proto_state == study_pb2.Trial.State.STOPPING:
    return trial.TrialStatus.STOPPING
  if proto_state == study_pb2.Trial.State.SUCCEEDED:
    return trial.TrialStatus.COMPLETED
  elif proto_state == study_pb2.Trial.State.INFEASIBLE:
    return trial.TrialStatus.COMPLETED
  else:
    return trial.TrialStatus.UNKNOWN


def _from_pyvizier_trial_status(
    status: trial.TrialStatus, infeasible: bool
) -> study_pb2.Trial.State:
  """to_proto conversion for Trial states."""
  if status == trial.TrialStatus.REQUESTED:
    return study_pb2.Trial.State.REQUESTED
  elif status == trial.TrialStatus.ACTIVE:
    return study_pb2.Trial.State.ACTIVE
  elif status == trial.TrialStatus.STOPPING:
    return study_pb2.Trial.State.STOPPING
  elif status == trial.TrialStatus.COMPLETED:
    if infeasible:
      return study_pb2.Trial.State.INFEASIBLE
    else:
      return study_pb2.Trial.State.SUCCEEDED
  else:
    return study_pb2.Trial.State.STATE_UNSPECIFIED


class TrialConverter:
  """Converter for vz.Trial."""

  @classmethod
  def from_proto(cls, proto: study_pb2.Trial) -> trial.Trial:
    """Converts from Trial proto to object.

    Args:
      proto: Trial proto.

    Returns:
      A Trial object.
    """
    parameters = {}
    for parameter in proto.parameters:
      value = ParameterValueConverter.from_proto(parameter)
      if value is not None:
        if parameter.parameter_id in parameters:
          raise ValueError(
              'Invalid trial proto contains duplicate parameter {}: {}'.format(
                  parameter.parameter_id, proto
              )
          )
        parameters[parameter.parameter_id] = value
      else:
        logging.warning(
            'A parameter without a value will be dropped: %s', parameter
        )

    final_measurement = None
    if proto.HasField('final_measurement'):
      final_measurement = MeasurementConverter.from_proto(
          proto.final_measurement
      )

    completion_time = None
    infeasibility_reason = None
    if proto.state == study_pb2.Trial.State.SUCCEEDED:
      if proto.HasField('end_time'):
        completion_ts = proto.end_time.seconds + 1e-9 * proto.end_time.nanos
        completion_time = datetime.datetime.fromtimestamp(completion_ts)
    elif proto.state == study_pb2.Trial.State.INFEASIBLE:
      infeasibility_reason = proto.infeasible_reason

    metadata = common.Metadata()
    for kv in proto.metadata:
      metadata.abs_ns(common.Namespace.decode(kv.ns))[kv.key] = (
          kv.proto if kv.HasField('proto') else kv.value
      )

    measurements = []
    for measure in proto.measurements:
      measurements.append(MeasurementConverter.from_proto(measure))

    creation_time = None
    if proto.HasField('start_time'):
      creation_ts = proto.start_time.seconds + 1e-9 * proto.start_time.nanos
      creation_time = datetime.datetime.fromtimestamp(creation_ts)
    return trial.Trial(
        id=int(proto.id),
        description=proto.name,
        assigned_worker=proto.client_id or None,
        is_requested=proto.state == proto.REQUESTED,
        stopping_reason=(
            'stopping reason not supported yet'
            if proto.state == proto.STOPPING
            else None
        ),
        parameters=parameters,
        creation_time=creation_time,
        completion_time=completion_time,
        infeasibility_reason=infeasibility_reason,
        final_measurement=final_measurement,
        measurements=measurements,
        metadata=metadata,
    )  # pytype: disable=wrong-arg-types

  @classmethod
  def from_protos(cls, protos: Iterable[study_pb2.Trial]) -> List[trial.Trial]:
    """Convenience wrapper for from_proto."""
    return [TrialConverter.from_proto(proto) for proto in protos]

  @classmethod
  def to_protos(cls, pytrials: Iterable[trial.Trial]) -> List[study_pb2.Trial]:
    return [TrialConverter.to_proto(pytrial) for pytrial in pytrials]

  @classmethod
  def to_proto(cls, pytrial: trial.Trial) -> study_pb2.Trial:
    """Converts a pyvizier Trial to a Trial proto."""
    proto = study_pb2.Trial()
    if pytrial.description is not None:
      proto.name = pytrial.description
    proto.id = str(pytrial.id)
    proto.state = _from_pyvizier_trial_status(
        pytrial.status, pytrial.infeasible
    )
    proto.client_id = pytrial.assigned_worker or ''

    for name, value in pytrial.parameters.items():
      proto.parameters.append(ParameterValueConverter.to_proto(value, name))

    # pytrial always adds an empty metric. Ideally, we should remove it if the
    # metric does not exist in the study config.
    if pytrial.final_measurement is not None:
      proto.final_measurement.CopyFrom(
          MeasurementConverter.to_proto(pytrial.final_measurement)
      )

    for measurement in pytrial.measurements:
      proto.measurements.append(MeasurementConverter.to_proto(measurement))

    if pytrial.creation_time is not None:
      creation_secs = datetime.datetime.timestamp(pytrial.creation_time)
      proto.start_time.seconds = int(creation_secs)
      proto.start_time.nanos = int(1e9 * (creation_secs - int(creation_secs)))
    if pytrial.completion_time is not None:
      completion_secs = datetime.datetime.timestamp(pytrial.completion_time)
      proto.end_time.seconds = int(completion_secs)
      proto.end_time.nanos = int(1e9 * (completion_secs - int(completion_secs)))
    if pytrial.infeasibility_reason is not None:
      proto.infeasible_reason = pytrial.infeasibility_reason
    if pytrial.metadata is not None:
      for ns in pytrial.metadata.namespaces():
        ns_string = ns.encode()
        ns_layer = pytrial.metadata.abs_ns(ns)
        for key, value in ns_layer.items():
          metadata_util.assign(proto, key=key, ns=ns_string, value=value)
    return proto


class TrialSuggestionConverter:
  """Converts vz.TrialSuggestion <--> Pythia TrialSuggestion proto."""

  @classmethod
  def from_proto(
      cls, proto: pythia_service_pb2.TrialSuggestion
  ) -> trial.TrialSuggestion:
    """Converts from TrialSuggestion proto to PyVizier TrialSuggestion."""
    parameters = {}
    for parameter in proto.parameters:
      value = ParameterValueConverter.from_proto(parameter)
      if value is None:
        raise RuntimeError('Parameter %s exists without a value.' % parameter)
      if parameter.parameter_id in parameters:
        raise ValueError(
            'Invalid trial proto contains duplicate parameter {}: {}'.format(
                parameter.parameter_id, proto
            )
        )
      parameters[parameter.parameter_id] = value

    metadata = common.Metadata()
    for kv in proto.metadata:
      metadata.abs_ns(common.Namespace.decode(kv.ns))[kv.key] = (
          kv.proto if kv.HasField('proto') else kv.value
      )

    return trial.TrialSuggestion(parameters=parameters, metadata=metadata)

  @classmethod
  def from_protos(
      cls, protos: Iterable[pythia_service_pb2.TrialSuggestion]
  ) -> List[trial.TrialSuggestion]:
    """Convenience wrapper for from_proto."""
    return [cls.from_proto(proto) for proto in protos]

  @classmethod
  def to_protos(
      cls, pytrials: Iterable[trial.TrialSuggestion]
  ) -> List[pythia_service_pb2.TrialSuggestion]:
    return [cls.to_proto(pytrial) for pytrial in pytrials]

  @classmethod
  def to_proto(
      cls, suggestion: trial.TrialSuggestion
  ) -> pythia_service_pb2.TrialSuggestion:
    """Converts a pyvizier TrialSuggestion to the corresponding proto."""
    proto = pythia_service_pb2.TrialSuggestion()

    for name, value in suggestion.parameters.items():
      proto.parameters.append(ParameterValueConverter.to_proto(value, name))

    proto.metadata.extend(
        metadata_util.make_key_value_list(suggestion.metadata)
    )
    return proto


class MetadataDeltaConverter:
  """Converts pyvizier.MetadataDelta <--> List of UnitMetadataUpdate protos."""

  @classmethod
  def to_protos(
      cls, delta: trial.MetadataDelta
  ) -> List[vizier_service_pb2.UnitMetadataUpdate]:
    """Converts pyvizier.MetadataDelta to a List of UnitMetadataUpdate protos."""
    unit_metadata_updates = metadata_util.study_metadata_to_update_list(
        delta.on_study
    )
    unit_metadata_updates.extend(
        metadata_util.trial_metadata_to_update_list(delta.on_trials)
    )
    return unit_metadata_updates

  @classmethod
  def from_protos(
      cls, protos: Iterable[vizier_service_pb2.UnitMetadataUpdate]
  ) -> trial.MetadataDelta:
    """Converts a list of UnitMetadataUpdate protos to pyvizier.MetadataDelta."""
    mdd = trial.MetadataDelta()
    for u_m_u in protos:
      key_value = u_m_u.metadatum
      namespace = common.Namespace.decode(key_value.ns)
      value = (
          key_value.proto if key_value.HasField('proto') else key_value.value
      )
      if u_m_u.HasField('trial_id'):
        mdd.on_trials[int(u_m_u.trial_id)].abs_ns(namespace)[
            key_value.key
        ] = value
      else:
        mdd.on_study.abs_ns(namespace)[key_value.key] = value
    return mdd


class ProblemStatementConverter:
  """Converts pyvizier.ProblemStatement <-> Pythia ProblemStatement proto."""

  @classmethod
  def to_proto(
      cls,
      problem_statement: base_study_config.ProblemStatement,
  ) -> pythia_service_pb2.ProblemStatement:
    """Converts PyVizier ProblemStatement to Proto version."""
    parameter_spec_protos = SearchSpaceConverter.parameter_protos(
        problem_statement.search_space
    )
    metric_information_protos = MetricsConfigConverter.to_protos(
        problem_statement.metric_information
    )
    keyvalue_protos = metadata_util.make_key_value_list(
        problem_statement.metadata
    )
    return pythia_service_pb2.ProblemStatement(
        search_space=parameter_spec_protos,
        metric_information=metric_information_protos,
        metadata=keyvalue_protos,
    )

  @classmethod
  def from_proto(
      cls, proto: pythia_service_pb2.ProblemStatement
  ) -> base_study_config.ProblemStatement:
    """Converts ProblemStatement Proto to PyVizier version."""
    study_spec = study_pb2.StudySpec(parameters=proto.search_space)
    search_space = SearchSpaceConverter.from_proto(study_spec)
    metric_information = MetricsConfigConverter.from_protos(
        proto.metric_information
    )
    metadata = metadata_util.from_key_value_list(proto.metadata)
    return base_study_config.ProblemStatement(
        search_space=search_space,
        metric_information=metric_information,
        metadata=metadata,
    )


class StudyDescriptorConverter:
  """Converts Pythia StudyDescriptorConverter <-> Pythia StudyDescriptor proto."""

  @classmethod
  def to_proto(
      cls,
      study_descriptor: study.StudyDescriptor,
  ) -> pythia_service_pb2.StudyDescriptor:
    return pythia_service_pb2.StudyDescriptor(
        config=ProblemStatementConverter.to_proto(study_descriptor.config),
        guid=study_descriptor.guid,
        max_trial_id=study_descriptor.max_trial_id,
    )

  @classmethod
  def from_proto(
      cls, proto: pythia_service_pb2.StudyDescriptor
  ) -> study.StudyDescriptor:
    return study.StudyDescriptor(
        config=ProblemStatementConverter.from_proto(proto.config),
        guid=proto.guid,
        max_trial_id=proto.max_trial_id,
    )


class SuggestConverter:
  """Converts a SuggestRequest class <--> a SuggestRequest proto."""

  @classmethod
  def to_request_proto(
      cls,
      request: policy.SuggestRequest,
  ) -> pythia_service_pb2.SuggestRequest:
    """Conversion from PyVizier to proto."""
    study_descriptor_proto = StudyDescriptorConverter.to_proto(
        request._study_descriptor  # pylint:disable=protected-access
    )
    return pythia_service_pb2.SuggestRequest(
        study_descriptor=study_descriptor_proto,
        count=request.count,
        checkpoint_dir=request.checkpoint_dir,
    )

  @classmethod
  def from_request_proto(
      cls,
      proto: pythia_service_pb2.SuggestRequest,
  ) -> policy.SuggestRequest:
    """Conversion from proto to PyVizier."""
    study_descriptor = StudyDescriptorConverter.from_proto(
        proto.study_descriptor
    )
    return policy.SuggestRequest(
        study_descriptor=study_descriptor,
        count=proto.count,
        checkpoint_dir=proto.checkpoint_dir,
    )

  @classmethod
  def to_decision_proto(
      cls,
      decision: policy.SuggestDecision,
  ) -> pythia_service_pb2.SuggestDecision:
    """Conversion from PyVizier to proto."""
    trial_suggestion_protos = TrialSuggestionConverter.to_protos(
        decision.suggestions
    )
    metadelta_protos = MetadataDeltaConverter.to_protos(decision.metadata)
    return pythia_service_pb2.SuggestDecision(
        suggestions=trial_suggestion_protos, metadata=metadelta_protos
    )

  @classmethod
  def from_decision_proto(
      cls,
      proto: pythia_service_pb2.SuggestDecision,
  ) -> policy.SuggestDecision:
    """Conversion from proto to PyVizier."""
    suggestions = TrialSuggestionConverter.from_protos(proto.suggestions)
    metadata = MetadataDeltaConverter.from_protos(proto.metadata)
    return policy.SuggestDecision(suggestions=suggestions, metadata=metadata)


class EarlyStopConverter:
  """Converts Pythia EarlyStopping <-> Pythia EarlyStopping protos."""

  @classmethod
  def to_request_proto(
      cls,
      request: policy.EarlyStopRequest,
  ) -> pythia_service_pb2.EarlyStopRequest:
    """Conversion from PyVizier to proto."""
    return pythia_service_pb2.EarlyStopRequest(
        study_descriptor=StudyDescriptorConverter.to_proto(
            request._study_descriptor  # pylint:disable=protected-access
        ),
        trial_ids=request.trial_ids,
        checkpoint_dir=request.checkpoint_dir,
    )

  @classmethod
  def from_request_proto(
      cls,
      proto: pythia_service_pb2.EarlyStopRequest,
  ) -> policy.EarlyStopRequest:
    """Conversion from proto to PyVizier."""
    study_descriptor = StudyDescriptorConverter.from_proto(
        proto.study_descriptor
    )
    return policy.EarlyStopRequest(
        study_descriptor=study_descriptor,
        trial_ids=proto.trial_ids,
        checkpoint_dir=proto.checkpoint_dir,
    )

  @classmethod
  def to_decisions_proto(
      cls,
      decisions: policy.EarlyStopDecisions,
  ) -> pythia_service_pb2.EarlyStopDecisions:
    """Conversion from PyVizier to proto."""
    decision_protos = []
    for decision in decisions.decisions:
      predicted_final_measurement_proto = study_pb2.Measurement()
      if decision.predicted_final_measurement:
        predicted_final_measurement_proto = MeasurementConverter.to_proto(
            decision.predicted_final_measurement
        )
      decision_proto = pythia_service_pb2.EarlyStopDecision(
          id=decision.id,
          reason=decision.reason,
          should_stop=decision.should_stop,
          predicted_final_measurement=predicted_final_measurement_proto,
      )
      decision_protos.append(decision_proto)
    key_value_protos = MetadataDeltaConverter.to_protos(decisions.metadata)
    return pythia_service_pb2.EarlyStopDecisions(
        decisions=decision_protos, metadata=key_value_protos
    )

  @classmethod
  def from_decisions_proto(
      cls,
      proto: pythia_service_pb2.EarlyStopDecisions,
  ) -> policy.EarlyStopDecisions:
    """Conversion from proto to PyVizier."""
    decisions = []
    for decision_proto in proto.decisions:
      decision = policy.EarlyStopDecision(
          id=decision_proto.id,
          reason=decision_proto.reason,
          should_stop=decision_proto.should_stop,
          predicted_final_measurement=MeasurementConverter.from_proto(
              decision_proto.predicted_final_measurement
          ),
      )
      decisions.append(decision)
    metadata = MetadataDeltaConverter.from_protos(proto.metadata)
    return policy.EarlyStopDecisions(decisions=decisions, metadata=metadata)
