"""Converters for OSS Vizier's protos from/to PyVizier's classes."""
import datetime
import logging
from typing import List, Optional, Sequence, Tuple, Union
from absl import logging

from vizier._src.pyvizier.oss import metadata_util
from vizier._src.pyvizier.shared import common
from vizier._src.pyvizier.shared import parameter_config
from vizier._src.pyvizier.shared import trial
from vizier.service import study_pb2

ScaleType = parameter_config.ScaleType
_ScaleTypePb2 = study_pb2.StudySpec.ParameterSpec.ScaleType
ParameterType = parameter_config.ParameterType
MonotypeParameterSequence = parameter_config.MonotypeParameterSequence


class _ScaleTypeMap:
  """Proto converter for scale type."""
  _pyvizier_to_proto = {
      parameter_config.ScaleType.LINEAR:
          _ScaleTypePb2.UNIT_LINEAR_SCALE,
      parameter_config.ScaleType.LOG:
          _ScaleTypePb2.UNIT_LOG_SCALE,
      parameter_config.ScaleType.REVERSE_LOG:
          _ScaleTypePb2.UNIT_REVERSE_LOG_SCALE,
  }
  _proto_to_pyvizier = {v: k for k, v in _pyvizier_to_proto.items()}

  @classmethod
  def to_proto(cls, pyvizier: parameter_config.ScaleType) -> _ScaleTypePb2:
    return cls._pyvizier_to_proto[pyvizier]

  @classmethod
  def from_proto(cls, proto: _ScaleTypePb2) -> parameter_config.ScaleType:
    return cls._proto_to_pyvizier[proto]


class ParameterConfigConverter:
  """Converter for ParameterConfig."""

  @classmethod
  def _set_bounds(cls, proto: study_pb2.StudySpec.ParameterSpec, lower: float,
                  upper: float, parameter_type: ParameterType):
    """Sets the proto's min_value and max_value fields."""
    if parameter_type == ParameterType.INTEGER:
      proto.integer_value_spec.min_value = lower
      proto.integer_value_spec.max_value = upper
    elif parameter_type == ParameterType.DOUBLE:
      proto.double_value_spec.min_value = lower
      proto.double_value_spec.max_value = upper

  @classmethod
  def _set_feasible_points(cls, proto: study_pb2.StudySpec.ParameterSpec,
                           feasible_points: Sequence[float]):
    """Sets the proto's feasible_points field."""
    feasible_points = sorted(feasible_points)
    proto.discrete_value_spec.ClearField('values')
    proto.discrete_value_spec.values.extend(feasible_points)

  @classmethod
  def _set_categories(cls, proto: study_pb2.StudySpec.ParameterSpec,
                      categories: Sequence[str]):
    """Sets the protos' categories field."""
    proto.categorical_value_spec.ClearField('values')
    proto.categorical_value_spec.values.extend(categories)

  @classmethod
  def _set_default_value(cls, proto: study_pb2.StudySpec.ParameterSpec,
                         default_value: Union[float, int, str]):
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
    if oneof_name in ('parent_discrete_values', 'parent_int_values',
                      'parent_categorical_values'):
      return list(getattr(getattr(proto, oneof_name), 'values'))
    raise ValueError('Unknown matching_parent_vals: {}'.format(oneof_name))

  @classmethod
  def from_proto(
      cls,
      proto: study_pb2.StudySpec.ParameterSpec,
      *,
      strict_validation: bool = False) -> parameter_config.ParameterConfig:
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
      bounds = (int(proto.integer_value_spec.min_value),
                int(proto.integer_value_spec.max_value))
    elif oneof_name == 'double_value_spec':
      bounds = (proto.double_value_spec.min_value,
                proto.double_value_spec.max_value)
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
            (parent_values, cls.from_proto(conditional_ps.parameter_spec)))
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
          default_value=default_value)
    except ValueError as e:
      raise ValueError(
          'The provided proto was misconfigured. {}'.format(proto)) from e

    if strict_validation and cls.to_proto(config) != proto:
      raise ValueError(
          'The provided proto was misconfigured. Expected: {} Given: {}'.format(
              cls.to_proto(config), proto))
    return config

  @classmethod
  def _set_child_parameter_configs(
      cls, parent_proto: study_pb2.StudySpec.ParameterSpec,
      pc: parameter_config.ParameterConfig):
    """Sets the parent_proto's conditional_parameter_specs field.

    Args:
      parent_proto: Modified in place.
      pc: Parent ParameterConfig to copy children from.

    Raises:
      ValueError: If the child configs are invalid
    """
    children: List[Tuple[MonotypeParameterSequence,
                         parameter_config.ParameterConfig]] = []
    for child in pc.child_parameter_configs:
      children.append((child.matching_parent_values, child))
    if not children:
      return

    parent_proto.ClearField('conditional_parameter_specs')
    for child_pair in children:
      if len(child_pair) != 2:
        raise ValueError("""Each element in children must be a tuple of
            (Sequence of valid parent values,  ParameterConfig)""")

    logging.debug('_set_child_parameter_configs: parent_proto=%s, children=%s',
                  parent_proto, children)
    for unsorted_parent_values, child in children:
      parent_values = sorted(unsorted_parent_values)
      child_proto = cls.to_proto(child.clone_without_children)
      conditional_parameter_spec = study_pb2.StudySpec.ParameterSpec.ConditionalParameterSpec(
          parameter_spec=child_proto)

      if parent_proto.HasField('discrete_value_spec'):
        conditional_parameter_spec.parent_discrete_values.values[:] = parent_values
      elif parent_proto.HasField('categorical_value_spec'):
        conditional_parameter_spec.parent_categorical_values.values[:] = parent_values
      elif parent_proto.HasField('integer_value_spec'):
        conditional_parameter_spec.parent_int_values.values[:] = parent_values
      else:
        raise ValueError('DOUBLE type cannot have child parameters')
      if child.child_parameter_configs:
        cls._set_child_parameter_configs(child_proto, child)
      parent_proto.conditional_parameter_specs.extend(
          [conditional_parameter_spec])

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
    if pc.scale_type is not None and pc.scale_type != ScaleType.UNIFORM_DISCRETE:
      proto.scale_type = _ScaleTypeMap.to_proto(pc.scale_type)
    if pc.default_value is not None:
      cls._set_default_value(proto, pc.default_value)

    cls._set_child_parameter_configs(proto, pc)
    return proto


class ParameterValueConverter:
  """Converter for trial.ParameterValue."""

  @classmethod
  def from_proto(
      cls, proto: study_pb2.Trial.Parameter) -> Optional[trial.ParameterValue]:
    """Returns whichever value that is populated, or None."""
    value_proto = proto.value
    oneof_name = value_proto.WhichOneof('kind')
    potential_value = getattr(value_proto, oneof_name)
    if isinstance(potential_value, float) or isinstance(
        potential_value, str) or isinstance(potential_value, bool):
      return trial.ParameterValue(potential_value)
    else:
      return None

  @classmethod
  def to_proto(cls, parameter_value: trial.ParameterValue,
               name: str) -> study_pb2.Trial.Parameter:
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
  """Converter for trial.MeasurementConverter."""

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
      if metric.metric_id in metrics and metrics[
          metric.metric_id].value != metric.value:
        logging.log_first_n(
            logging.ERROR, 'Duplicate metric of name "%s".'
            'The newly found value %s will be used and '
            'the previously found value %s will be discarded.'
            'This always happens if the proto has an empty-named metric.', 5,
            metric.metric_id, metric.value, metrics[metric.metric_id].value)
      try:
        metrics[metric.metric_id] = trial.Metric(value=metric.value)
      except ValueError:
        pass
    return trial.Measurement(
        metrics=metrics,
        elapsed_secs=proto.elapsed_duration.seconds,
        steps=proto.step_count)

  @classmethod
  def to_proto(cls, measurement: trial.Measurement) -> study_pb2.Measurement:
    """Converts to Measurement proto."""
    proto = study_pb2.Measurement()
    for name, metric in measurement.metrics.items():
      proto.metrics.add(metric_id=name, value=metric.value)

    proto.step_count = measurement.steps
    int_seconds = int(measurement.elapsed_secs)
    proto.elapsed_duration.seconds = int_seconds
    proto.elapsed_duration.nanos = int(1e9 *
                                       (measurement.elapsed_secs - int_seconds))
    return proto


def _to_pyvizier_trial_status(
    proto_state: study_pb2.Trial.State) -> trial.TrialStatus:
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


def _from_pyvizier_trial_status(status: trial.TrialStatus,
                                infeasible: bool) -> study_pb2.Trial.State:
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
  """Converter for trial.TrialConverter."""

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
          raise ValueError('Invalid trial proto contains duplicate parameter {}'
                           ': {}'.format(parameter.parameter_id, proto))
        parameters[parameter.parameter_id] = value
      else:
        logging.warning('A parameter without a value will be dropped: %s',
                        parameter)

    final_measurement = None
    if proto.HasField('final_measurement'):
      final_measurement = MeasurementConverter.from_proto(
          proto.final_measurement)

    completion_time = None
    infeasibility_reason = None
    if proto.state == study_pb2.Trial.State.SUCCEEDED:
      if proto.HasField('end_time'):
        completion_ts = proto.end_time.seconds + 1e-9 * proto.end_time.nanos
        completion_time = datetime.datetime.fromtimestamp(completion_ts)
    elif proto.state == study_pb2.Trial.State.INFEASIBLE:
      infeasibility_reason = proto.infeasible_reason

    metadata = trial.Metadata()
    for kv in proto.metadata:
      metadata.abs_ns(common.Namespace.decode(kv.ns))[kv.key] = (
          kv.proto if kv.HasField('proto') else kv.value)

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
        stopping_reason=('stopping reason not supported yet'
                         if proto.state == proto.STOPPING else None),
        parameters=parameters,
        creation_time=creation_time,
        completion_time=completion_time,
        infeasibility_reason=infeasibility_reason,
        final_measurement=final_measurement,
        measurements=measurements,
        metadata=metadata)  # pytype: disable=wrong-arg-types

  @classmethod
  def from_protos(cls, protos: Sequence[study_pb2.Trial]) -> List[trial.Trial]:
    """Convenience wrapper for from_proto."""
    return [TrialConverter.from_proto(proto) for proto in protos]

  @classmethod
  def to_protos(cls, pytrials: Sequence[trial.Trial]) -> List[study_pb2.Trial]:
    return [TrialConverter.to_proto(pytrial) for pytrial in pytrials]

  @classmethod
  def to_proto(cls, pytrial: trial.Trial) -> study_pb2.Trial:
    """Converts a pyvizier Trial to a Trial proto."""
    proto = study_pb2.Trial()
    if pytrial.description is not None:
      proto.name = pytrial.description
    proto.id = str(pytrial.id)
    proto.state = _from_pyvizier_trial_status(pytrial.status,
                                              pytrial.infeasible)
    proto.client_id = pytrial.assigned_worker or ''

    for name, value in pytrial.parameters.items():
      proto.parameters.append(ParameterValueConverter.to_proto(value, name))

    # pytrial always adds an empty metric. Ideally, we should remove it if the
    # metric does not exist in the study config.
    if pytrial.final_measurement is not None:
      proto.final_measurement.CopyFrom(
          MeasurementConverter.to_proto(pytrial.final_measurement))

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
