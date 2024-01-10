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

"""PyVizier module. See VizierConverter class."""

import base64
import datetime
import json
import lzma
import numbers
from typing import Any, Dict, List, Optional, Sequence, Literal

from absl import logging
import attr
import pyglove as pg
from vizier import pyvizier as vz
from vizier._src.pyglove import constants


def _to_json_str_compressed(value: Any) -> str:
  """Serialize (maybe) symbolic object to compressed JSON value."""
  return base64.b64encode(
      lzma.compress(json.dumps(
          pg.to_json(value)).encode('utf-8'))).decode('ascii')


def _parameter_with_external_type(
    val: vz.ParameterValueTypes,
    external_type: vz.ExternalType) -> vz.ParameterValueTypes:
  """Converts a parameter value to proper external type."""
  if external_type == vz.ExternalType.BOOLEAN:
    # We output strings 'True' or 'False', not booleans themselves.
    # because BOOLEAN is interally CATEGORICAL.
    return val
  elif external_type == vz.ExternalType.INTEGER:
    return int(val)
  elif external_type == vz.ExternalType.FLOAT:
    return float(val)
  else:
    return val


def _make_decision_point(
    parameter_config: vz.ParameterConfig) -> pg.geno.DecisionPoint:
  """Make a decision point (DNASpec) out from a parameter config."""

  # NOTE(daiyip): We set the name of each decision point instead of its
  # location with parameter name.
  #
  # Why? For conditional space, the ID of a decision point is a
  # path of locations from the root to the leaf node. For example, if there
  # are two parameters - a parent with location 'a' and a child with location
  # 'b', the ID for the child will be 'a.b'. However, for external (
  # non-PyGlove created) study, the parameter name for the child does not
  # follow this pattern. The solution is to use the ``name`` property of
  # `DNASpec`, which allows the user to access hierarchical decision
  # points by name, also DNA supports to_dict/from_dict based on the decision
  # point names instead of their IDs. Therefore, we can minimize the
  # difference between a PyGlove created study and an external study.

  name = parameter_config.name
  if parameter_config.type == vz.ParameterType.DOUBLE:
    # Create `pg.geno.Float` which does not have child spaces.
    min_value, max_value = parameter_config.bounds
    return pg.geno.Float(min_value, max_value, name=name)
  elif parameter_config.type in (vz.ParameterType.CATEGORICAL,
                                 vz.ParameterType.DISCRETE,
                                 vz.ParameterType.INTEGER):
    # Create `pg.geno.Choices` with possible child spaces.
    candidates = []
    literal_values = []
    for val in parameter_config.feasible_values:
      child_decision_points = []
      if parameter_config.child_parameter_configs:
        for child_pc in parameter_config.child_parameter_configs:
          if val in child_pc.matching_parent_values:
            child_decision_points.append(_make_decision_point(child_pc))
      candidates.append(pg.geno.Space(child_decision_points))
      literal_values.append(
          _parameter_with_external_type(val, parameter_config.external_type))
    return pg.geno.Choices(
        1, candidates, literal_values=literal_values, name=name)
  else:
    raise ValueError(
        f'Parameter Config Type {parameter_config.type!r} is not supported.')


def _to_dna_spec(search_space: vz.SearchSpace) -> pg.DNASpec:
  return pg.geno.Space(
      [_make_decision_point(pc) for pc in search_space.parameters])


def _to_search_space(dna_spec: pg.DNASpec) -> vz.SearchSpace:
  """Converts a DNASpec to Vizier search space.

  Args:
    dna_spec:

  Returns:
    Vizier search space.

  Raises:
    NotImplementedError: If no part of the spec can be converted to a Vizier
      parameter.
  """

  def _parameter_name(path: pg.KeyPath) -> str:
    # NOTE(daiyip): Vizier doesn't support empty name, thus we use a
    # special parameter name for the hyper value at root.
    return path.path if path else constants.PARAMETER_NAME_ROOT

  def _categories(spec: pg.geno.Choices) -> List[str]:
    return [spec.format_candidate(i) for i in range(len(spec.candidates))]

  def _category_value(spec: pg.geno.Choices, index: int) -> str:
    assert index < len(spec.candidates)
    return spec.format_candidate(index)

  def _add_dna_spec(root: vz.SearchSpaceSelector, path: pg.KeyPath,
                    spec: pg.DNASpec) -> None:
    """Convert a DNASpec node with parent choice to a list of parameters.

    Args:
      root: The DNA spec is added to this root.
      path: Root path of current DNA spec.
      spec: Current DNA spec.
    """
    if isinstance(spec, pg.geno.Space):
      for elem in spec.elements:
        _add_dna_spec(root, path + elem.location, elem)
    elif isinstance(spec, pg.geno.Choices):
      is_discrete = all(
          isinstance(v, numbers.Number) for v in spec.literal_values
      ) and len(set(spec.literal_values)) == len(spec.literal_values)

      for choice_idx in range(spec.num_choices):
        choice_path = path
        if spec.num_choices > 1:
          choice_path = choice_path + choice_idx

        if is_discrete:
          unique_feasible_points = sorted(set(spec.literal_values))
          root.add_discrete_param(
              name=_parameter_name(choice_path),
              # We sort the literal values since Vizier requires the feasible
              # points of a discrete parameter to be in increasing order.
              # The sorting has no impact to the trial parameter -> DNA
              # conversion since for numeric literal value, the conversion
              # is value based.
              feasible_values=unique_feasible_points,
          )
          if unique_feasible_points != spec.literal_values:
            logging.warning(
                'Candidates for parameter %r have been reordered/deduped from '
                '%s to %s to meet the sorted/distinct requirement for discrete '
                'parameter specifiction.',
                _parameter_name(choice_path),
                spec.literal_values,
                unique_feasible_points)
        else:
          new_parameter: vz.SearchSpaceSelector = root.add_categorical_param(
              name=_parameter_name(choice_path),
              feasible_values=_categories(spec))
          for candidate_idx, candidate in enumerate(spec.candidates):
            candidate_path = choice_path + pg.geno.ConditionalKey(
                candidate_idx, len(spec.candidates)
            )
            child: vz.SearchSpaceSelector = new_parameter.select_values(
                [_category_value(spec, candidate_idx)])
            _add_dna_spec(child, candidate_path, candidate)
    elif isinstance(spec, pg.geno.Float):
      root.add_float_param(
          name=_parameter_name(path),
          scale_type=get_scale_type(spec.scale),
          min_value=spec.min_value,
          max_value=spec.max_value)
    elif isinstance(spec, pg.geno.CustomDecisionPoint):
      # For CustomDecisionPoint, there is not a corresponding parameter type
      # in Vizier since its value is a variable string. In such case the
      # parameter value will be put into metadata.
      logging.info(
          'Encountered custom decision point %s, which will not be shown '
          'in Vizier dashboard.',
          _parameter_name(path),
      )
    else:
      raise NotImplementedError(
          f'Spec has unknown type. This Should never happen. Spec: {spec}')

  search_space = vz.SearchSpace()
  _add_dna_spec(search_space.root, pg.KeyPath(), dna_spec)

  if not search_space.parameters:
    raise NotImplementedError(
        'No part of the dna spec could be represented as a Vizier parameter.')
  return search_space


def get_scale_type(scale: Optional[str]) -> Optional[vz.ScaleType]:
  """Returns scale type based on scale string."""
  if scale in [None, 'linear']:
    return vz.ScaleType.LINEAR
  elif scale == 'log':
    return vz.ScaleType.LOG
  elif scale == 'rlog':
    return vz.ScaleType.REVERSE_LOG
  else:
    raise ValueError(f'Unsupported scale type: {scale!r}')


def get_pyglove_metadata(trial: vz.Trial) -> dict[str, Any]:
  """Extracts only the pyglove-related metadata into a simple dict."""
  metadata = dict()

  # NOTE(daiyip): This is to keep backward compatibility for Cloud NAS service,
  # which might loads trials from studies created in the old NAS pipeline for
  # transfer learning.
  for key, value in trial.metadata.items():
    if key in constants.TRIAL_METADATA_KEYS:
      metadata[key] = pg.from_json_str(value)

  for key, value in trial.metadata.ns(constants.METADATA_NAMESPACE).items():
    if key not in constants.TRIAL_METADATA_KEYS and value is not None:
      metadata[key] = pg.from_json_str(value)
  return metadata


def get_pyglove_study_metadata(problem: vz.ProblemStatement) -> pg.Dict:
  """Extracts only the pyglove-related metadata into a simple dict."""
  metadata = pg.Dict()
  pg_metadata = problem.metadata.ns(constants.METADATA_NAMESPACE)
  for key, value in pg_metadata.items():
    if key not in constants.STUDY_METADATA_KEYS and value is not None:
      metadata[key] = pg.from_json_str(value)
  return metadata


@attr.frozen
class VizierConverter:
  """Converts between PyGlove DNA and Vizier Trial.

  It can be initialized from a pg.DNASpec or vz.SearchSpace. It handles
  conversions between pg.DNA and vz.Trial.

  NOTE: Use a factory instead of __init__. There are two factories:
    VizierConverter.from_problem(...)
    VizierConverter.from_dna_spec(...)

  CAVEAT: The set of search spaces that can be described pg.DNASpec and
  vz.SearchSpace does not fully overlap. (e.g. DNASpec does not support
  discrete doubles and integers) As a result,
  `VizierConverter.from_dna_spec(VizierConverter.from_search_space(s).dna_spec)`
  is not the same as `VizierConverter.from_search_space(s)`.

  If VizierConverter is created from a DNA spec that can't be represented
  in Vizier search space, then `_problem` has a dummy search space and
  effectively has no purpose beyond saving dna_spec in metadata.
  In this case, `vizier_conversion_error` has a non-None value.
  """

  _dna_spec: pg.DNASpec = attr.field()
  _problem: vz.ProblemStatement = attr.field()
  _uses_external_dna_spec: bool = attr.field()

  vizier_conversion_error: Optional[Exception] = attr.field(
      default=None, kw_only=True)

  def __attrs_post_init__(self):
    # Store the dna spec in the metadata.
    self._problem.metadata.ns(constants.METADATA_NAMESPACE)[
        constants.STUDY_METADATA_KEY_DNA_SPEC] = _to_json_str_compressed(
            self._dna_spec)
    self._problem.metadata.ns(constants.METADATA_NAMESPACE)[
        constants.STUDY_METADATA_KEY_USES_EXTERNAL_DNA_SPEC
    ] = pg.to_json_str(self._uses_external_dna_spec)

  @property
  def metrics_to_optimize(self) -> Sequence[str]:
    metrics = []
    for m in self._problem.metric_information:
      if m.goal == vz.ObjectiveMetricGoal.MAXIMIZE:
        metrics.append(m.name)
      else:
        metrics.append(f'negative_{m.name}')
    return metrics

  @classmethod
  def from_problem(cls, problem: vz.ProblemStatement) -> 'VizierConverter':
    """Creates from vizier problem."""
    # TODO: Check this implementation
    json_str_compressed = problem.metadata.ns(constants.METADATA_NAMESPACE).get(
        constants.STUDY_METADATA_KEY_DNA_SPEC, None)

    if json_str_compressed is not None:
      dna_spec = restore_dna_spec(json_str_compressed)
    else:
      dna_spec = _to_dna_spec(problem.search_space)

    if bad_metrics := tuple(
        filter(lambda m: m.goal != vz.ObjectiveMetricGoal.MAXIMIZE,
               problem.metric_information)):
      raise ValueError(
          f'All goals must MAXIMIZE. Offending metrics: {bad_metrics}')
    if bad_metrics := tuple(
        filter(lambda m: m.type != vz.MetricType.OBJECTIVE,
               problem.metric_information)):
      logging.warning('All goals must be OBJECTIVE. Offending metrics: %s',
                      bad_metrics)

    uses_external_dna_spec = (
        json_str_compressed is not None
        and pg.from_json_str(
            problem.metadata.ns(constants.METADATA_NAMESPACE).get(
                constants.STUDY_METADATA_KEY_USES_EXTERNAL_DNA_SPEC, 'true'
            )
        )
    )
    return cls(dna_spec, problem, uses_external_dna_spec)

  @classmethod
  def from_dna_spec(
      cls,
      dna_spec: pg.DNASpec,
      metrics_to_maximize: Sequence[str] = (constants.REWARD_METRIC_NAME,)
  ) -> 'VizierConverter':
    """Create from dna spec."""
    problem = vz.ProblemStatement()
    for name in metrics_to_maximize:
      problem.metric_information.append(
          vz.MetricInformation(name, goal=vz.ObjectiveMetricGoal.MAXIMIZE))

    try:
      problem.search_space = _to_search_space(dna_spec)
      return cls(dna_spec, problem, True)
    except NotImplementedError as e:
      # Add a dummy parameter.
      problem.search_space.root.add_categorical_param(
          constants.DUMMY_PARAMETER_NAME,
          feasible_values=[constants.DUMMY_PARAMETER_VALUE])
      logging.info(
          'The provided DNA spec cannot be converted to a '
          'Vizier search space. Vizier algorithms cannot be used. '
          'Error was: %s', e)
      return cls(dna_spec, problem, True, vizier_conversion_error=e)

  @property
  def dna_spec(self) -> pg.DNASpec:
    return self._dna_spec

  @property
  def problem(self) -> vz.ProblemStatement:
    """Raises an error if the dna spec cannot be represented in Vizier."""
    if self.vizier_conversion_error:
      raise self.vizier_conversion_error
    return self._problem

  @property
  def uses_external_dna_spec(self) -> bool:
    """Returns True if the dna spec is provided from external."""
    return self._uses_external_dna_spec

  @property
  def problem_or_dummy(self) -> vz.ProblemStatement:
    """Returns dummy if the dna spec cannot be represented in Vizier."""
    return self._problem

  @property
  def search_space(self) -> vz.SearchSpace:
    if self.vizier_conversion_error:
      raise NotImplementedError(
          f'Vizier algorithms cannot work with the dna spec. '
          f'Error was: {self.vizier_conversion_error}')
    return self._problem.search_space

  def _process_key_value(self, key: str, value: vz.ParameterValueTypes):
    if key == constants.PARAMETER_NAME_ROOT:
      key = ''
    if self.dna_spec.hints == constants.FROM_VIZIER_STUDY_HINT:
      if not isinstance(value, str):
        value = float(value)  # Integers are always converted to doubles.
      if isinstance(self.dna_spec[key], pg.geno.Choices):
        value = repr(value)
    return key, value

  def _parameters_to_dict(self,
                          trial: vz.Trial) -> Dict[str, vz.ParameterValueTypes]:
    return dict({
        self._process_key_value(k, v)
        for k, v in trial.parameters.as_dict().items()
    })

  def to_dna(self, trial: vz.Trial) -> pg.DNA:
    """Extract DNA from vizier trial."""
    decision_dict = self._parameters_to_dict(trial)
    if len(
        decision_dict) == 1 and constants.DUMMY_PARAMETER_NAME in decision_dict:
      decision_dict = {}

    custom_decisions_str = trial.metadata.ns(constants.METADATA_NAMESPACE).get(
        constants.TRIAL_METADATA_KEY_CUSTOM_TYPE_DECISIONS, None)
    if custom_decisions_str is not None:
      custom_decisions = pg.from_json_str(custom_decisions_str)
      assert isinstance(custom_decisions, dict)
      decision_dict.update(custom_decisions)

    dna = pg.DNA.from_dict(
        decision_dict, self.dna_spec, use_ints_as_literals=True)

    # Restore DNA metadata if present
    dna_metadata = trial.metadata.ns(constants.METADATA_NAMESPACE).get(
        constants.TRIAL_METADATA_KEY_DNA_METADATA, None
    )

    if dna_metadata is None:
      # NOTE(daiyip): To be compatible with V1 pipeline for transfer learning,
      # we also try to read DNA_METADATA stored under the global (empty)
      # namespace.
      dna_metadata = trial.metadata.get(
          constants.TRIAL_METADATA_KEY_DNA_METADATA, None
      )

    if dna_metadata is not None:
      dna.rebind(
          metadata=pg.from_json_str(dna_metadata),
          skip_notification=True,
          raise_on_no_change=False,
      )
    else:
      logging.warn('DNA metadata is None for trial: %s', trial)
    return dna

  def to_trial(self, dna: pg.DNA, *,
               fallback: Literal['raise_error', 'return_dummy']) -> vz.Trial:
    """Converts DNA to vizier Trial.

    Args:
      dna:
      fallback: Decides the behavior when Vizier Search space is ill-formed from
        the DNA spec. 'raise_error': Raises an error. 'return_dummy': Returns a
        dummy trial.

    Returns:
      Vizier Trial.

    Raises:
      NotImplementedError: DNA has no valid representation as a Vizier trial.
    """
    trial = vz.Trial()
    trial.description = str(dna)
    trial.metadata.ns(constants.METADATA_NAMESPACE)[
        constants.TRIAL_METADATA_KEY_DNA_METADATA] = pg.to_json_str(
            dna.metadata)

    # Custom decision.
    def is_custom(x):
      return isinstance(x, pg.geno.CustomDecisionPoint)

    custom_decisions = dna.to_dict(filter_fn=is_custom)
    if custom_decisions:
      trial.metadata.ns(constants.METADATA_NAMESPACE)[
          constants.TRIAL_METADATA_KEY_CUSTOM_TYPE_DECISIONS] = pg.to_json_str(
              custom_decisions)

    if self.vizier_conversion_error:
      if fallback == 'return_dummy':
        trial.parameters[
            constants.DUMMY_PARAMETER_NAME] = constants.DUMMY_PARAMETER_VALUE
      else:
        raise self.vizier_conversion_error
    else:

      def is_discrete(x):
        return (
            isinstance(x, pg.geno.Choices)
            and all(isinstance(v, numbers.Number) for v in x.literal_values)
            and len(set(x.literal_values)) == len(x.literal_values)
        )

      if self._uses_external_dna_spec:
        key_type, value_type = 'id', 'choice_and_literal'
      else:
        key_type, value_type = 'name_or_id', 'literal'

      # Update (non-discrete) configured parameters from DNA.
      # We still write discrete parameter values in 'choice_and_literal' format
      # though their values will be overriden later, to preserve the order
      # of parameters.
      configured_parameters = dna.to_dict(
          key_type=key_type,
          value_type=value_type,
          filter_fn=lambda x: not is_custom(x),
      )

      # Update (discrete) configured parameters from DNA.
      # Force converting literal values to float to match DISCRETE parameter
      # expectation.
      discrete_parameters = {
          k: float(v)
          for k, v in dna.to_dict(
              key_type=key_type, value_type='literal', filter_fn=is_discrete
          ).items()
      }
      configured_parameters.update(discrete_parameters)

      if not configured_parameters:
        configured_parameters[constants.DUMMY_PARAMETER_NAME] = (
            constants.DUMMY_PARAMETER_VALUE
        )

      for name, value in configured_parameters.items():
        trial.parameters[name or constants.PARAMETER_NAME_ROOT] = (
            vz.ParameterValue(value)
        )
    return trial

  def to_tuner_measurement(
      self, vizier_measurement: Optional[vz.Measurement]
  ) -> Optional[pg.tuning.Measurement]:
    """Convert Vizier measurement to tuner Measurement."""
    if not vizier_measurement:
      return None
    tuner_measurement = pg.tuning.Measurement(
        step=int(vizier_measurement.steps),
        elapse_secs=vizier_measurement.elapsed_secs,
        reward=vizier_measurement.metrics.get_value(
            constants.REWARD_METRIC_NAME, 0.))
    tuner_measurement.metrics = {
        m: vizier_measurement.metrics.get_value(m, 0.)
        for m in vizier_measurement.metrics
    }
    return tuner_measurement

  def to_tuner_trial(self, vizier_trial: vz.Trial) -> pg.tuning.Trial:
    return pg.tuning.Trial(
        id=vizier_trial.id,
        description=vizier_trial.description,
        dna=self.to_dna(vizier_trial),
        metadata=dict(vizier_trial.metadata.ns(constants.METADATA_NAMESPACE)),
        related_links=dict(
            vizier_trial.metadata.ns(constants.METADATA_NAMESPACE).ns(
                constants.RELATED_LINKS_SUBNAMESPACE
            )
        ),
        measurements=vizier_trial.measurements,
        final_measurement=self.to_tuner_measurement(
            vizier_trial.final_measurement
        ),
        status=self._to_tuner_trial_status(vizier_trial.status),
        created_time=int(
            vizier_trial.creation_time.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()
        ),
        completed_time=int(  # pylint: disable=g-long-ternary
            vizier_trial.completion_time.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()
        )
        if vizier_trial.completion_time
        else None,
        infeasible=vizier_trial.infeasible,
    )

  def _to_tuner_trial_status(self, status: vz.TrialStatus) -> str:
    """Convert Vizier trial status to tuner trial status."""
    return 'PENDING' if status == vz.TrialStatus.ACTIVE else status.name


def restore_dna_spec(json_str_compressed: str) -> pg.DNASpec:
  """Restores DNASpec from compressed JSON str."""
  return pg.from_json(
      json.loads(lzma.decompress(base64.b64decode(json_str_compressed)))
  )
