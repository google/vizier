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

"""Defines core components for tuner integration."""

import collections
import contextlib
import datetime
import typing
from typing import Any, Optional, Sequence

from absl import logging
import attr
import pyglove as pg
from vizier import pyvizier as vz
from vizier._src.pyglove import constants
from vizier._src.pyglove import converters
from vizier.client import client_abc


def _trial_status_legacy_value(status: vz.TrialStatus) -> str:
  # PENDING was renamed to ACTIVE.
  if status == vz.TrialStatus.ACTIVE:
    return 'PENDING'
  return status.value


class VizierTrial(pg.tuning.Trial):
  """Override Trial to lazy load DNA and metadata upon access.

  When we construct a `Trial` object, it doesn't pop up DNA, measurements and
  metadata from vizier trial proto immediately. This is because that a study
  may consists of thousands of trials, if we load them at construction time, it
  would take minutes, which is not acceptable. So we made the `Trial` object
  lazily load these properties upon access, reducing the construction time into
  a few seconds.
  """

  # Here we explicitly override the __init__ method managed by PyGlove,
  # for we want to pass in `converter` and `trial` which are not managed by
  # PyGlove.
  @pg.explicit_method_override
  def __init__(
      self, converter: converters.VizierConverter, trial: vz.Trial, **kwargs
  ):
    completed_time = (
        int(trial.completion_time.timestamp()) if trial.completion_time else 0
    )
    super().__init__(
        dna=pg.DNA(None),
        id=trial.id,
        description=trial.description,
        final_measurement=converter.to_tuner_measurement(
            trial.final_measurement
        ),
        status=_trial_status_legacy_value(trial.status),
        created_time=int(trial.creation_time.timestamp()),
        completed_time=completed_time,
        infeasible=trial.infeasible,
        **kwargs,
    )
    self._converter = converter
    self._trial = trial

  @property
  def dna(self) -> pg.DNA:
    """Returns lazy loaded DNA."""
    temp_dna = self.sym_init_args.dna
    if temp_dna.value is None and not temp_dna.children:
      self.sym_init_args.dna = self._converter.to_dna(self._trial)
    return self.sym_init_args.dna

  @property
  def metadata(self) -> dict[str, Any]:
    """Returns lazy loaded metadata."""
    if not self.sym_init_args.metadata and self._trial:
      self.sym_init_args.metadata = converters.get_pyglove_metadata(self._trial)
    return self.sym_init_args.metadata

  @property
  def related_links(self) -> dict[str, str]:
    """Returns lazy loaded related links."""
    if not self.sym_init_args.related_links and self._trial:
      self.sym_init_args.related_links = dict(
          self._trial.metadata.ns(constants.METADATA_NAMESPACE).ns(
              constants.RELATED_LINKS_SUBNAMESPACE
          )
      )
    return self.sym_init_args.related_links

  @property
  def measurements(self) -> list[pg.tuning.Measurement]:
    """Returns lazy loaded measurements."""
    if not self.sym_init_args.measurements:
      self.sym_init_args.measurements = [
          self._converter.to_tuner_measurement(m)
          for m in self._trial.measurements
      ]
    return self.sym_init_args.measurements

  def format(self, *args, **kwargs):
    """Fetch lazy bound properties before print."""
    # NOTE(daiyip): `format` depends on the symbolic attributes to generate
    # the string representation. Since the following symbolic attributes are
    # lazily assigned upon property accesses, we prefetch them before calling
    # the `format`. Otherwise, the symbolic attributes are just default values
    # set at __init__ time.
    _, _, _, _ = self.dna, self.measurements, self.metadata, self.related_links
    return super().format(*args, **kwargs)


def _parse_namespace_from_key(
    encoded_key: str, default_ns: vz.Namespace
) -> tuple[vz.Namespace, str]:
  """From ':ns:key' to (ns, key)."""
  ns_and_key = tuple(vz.Namespace.decode(encoded_key))
  if not ns_and_key:
    raise ValueError(
        f'String did not parse into namespace and key: {encoded_key}'
    )
  elif len(ns_and_key) == 1:
    return (default_ns, ns_and_key[-1])
  else:
    return (vz.Namespace(ns_and_key[:-1]), ns_and_key[-1])


class Feedback(pg.tuning.Feedback):
  """Tuning feedback for a vizier trial."""

  def __init__(
      self,
      vizier_trial: client_abc.TrialInterface,
      converter: converters.VizierConverter,
  ):
    """Constructor.

    Args:
      vizier_trial: Vizier trial (cross-platform).
      converter: Vizier-Pyglove converter.
    """
    super().__init__(converter.metrics_to_optimize)
    self._converter = converter
    self._trial_client = vizier_trial
    self._trial = self._trial_client.materialize()
    self._dna_spec = converter.dna_spec
    self._discard_reward = 'reward' not in converter.metrics_to_optimize

  @property
  def id(self) -> int:
    """Gets Trial ID as ID."""
    return self._trial_client.id

  @property
  def dna(self) -> pg.DNA:
    """Gets DNA of current trial."""
    return self._converter.to_dna(self._trial)

  def get_trial(self) -> pg.tuning.Trial:
    """Gets current trial with all fields up-to-date."""
    self._trial = self._trial_client.materialize()
    return VizierTrial(self._converter, self._trial)

  @property
  def checkpoint_to_warm_start_from(self) -> Optional[str]:
    """Gets checkpoint path to warm start from. Refreshes `_trial`."""
    return None

  @contextlib.contextmanager
  def _maybe_race_condition(self, message: str):
    """Raise race condition error when error message matches with regex."""
    try:
      yield
    # TODO: once pyvizier expose common error types, we should
    # change `Exception` to a narrower error type.
    except Exception as e:  # pylint:disable=broad-except
      if message in str(e):
        raise pg.tuning.RaceConditionError(str(e)) from e
      else:
        raise

  def _add_measurement(
      self,
      reward: Optional[float],
      metrics: dict[str, float],
      step: int,
      checkpoint_path: Optional[str],
      elapse_secs: float,
  ) -> None:
    """Reports tuning measurement to the pg.tuning."""
    if reward is not None and not self._discard_reward:
      metrics |= {'reward': reward}
    with self._maybe_race_condition('Measurements can only be added to'):
      self._trial_client.add_measurement(
          vz.Measurement(metrics, elapsed_secs=elapse_secs, steps=step)
      )

  def set_metadata(self, key: str, value: Any, per_trial: bool = True) -> None:
    """Sets metadata for current trial or current sampling."""
    md = vz.Metadata()
    md.ns(constants.METADATA_NAMESPACE)[key] = pg.to_json_str(value)
    if per_trial:
      self._trial_client.update_metadata(md)
    else:
      self._trial_client.study.update_metadata(md)

  def get_metadata(self, key: str, per_trial: bool = True) -> Optional[Any]:
    """Gets metadata for current trial or current sampling.

    Args:
      key: A key to the Trial or StudyConfig metadata.  Vizier treats this as
        {namespace}:{key} where colons in {key} are escaped with a backslash,
        and {namespace} is encoded per vizier.pyvizier.Namespace.encode (i.e.
        colons are escaped and namespace components are separated by colons).
        But for the special case of the empty namespace, this simplifies to be
        just the key string.
      per_trial: True if you want to see per-trial metadata for the current
        Trial; false for study-related metadata.

    Returns:
      A metadata item, interpreted as JSON format.
    """
    abs_ns, key_in_ns = _parse_namespace_from_key(
        key, default_ns=vz.Namespace([constants.METADATA_NAMESPACE])
    )
    if per_trial:
      value = self._trial.metadata.abs_ns(abs_ns).get(key_in_ns, None)
    else:
      value = (
          self._trial_client.study.materialize_problem_statement()
          .metadata.abs_ns(abs_ns)
          .get(key_in_ns, None)
      )
    return pg.from_json_str(value) if value is not None else None

  def add_link(self, name: str, url: str) -> None:
    """Adds related link."""
    md = vz.Metadata()
    md.ns(constants.METADATA_NAMESPACE).ns(
        constants.RELATED_LINKS_SUBNAMESPACE
    )[name] = url
    self._trial_client.update_metadata(md)

  def done(
      self,
      metadata: Optional[dict[str, Any]] = None,
      related_links: Optional[dict[str, str]] = None,
  ) -> None:
    """Marks current tuning trial as done, and export final object."""
    metadata = metadata or {}
    related_links = related_links or {}
    for key, value in metadata.items():
      self.set_metadata(key, value)
    for key, value in related_links.items():
      self.add_link(key, value)
    self._trial_client.complete()
    self._trial = self._trial_client.materialize()

  def skip(self, reason: Optional[str] = None) -> None:
    """Skips current trial without providing feedback to the controller."""
    self._trial_client.complete(infeasible_reason=reason or 'skipped')

  def should_stop_early(self) -> bool:
    """Tells whether this trial should be stopped."""
    return self._trial_client.check_early_stopping()

  def end_loop(self) -> None:
    """Ends current search loop."""
    self._trial_client.study.set_state(vz.StudyState.ABORTED)


@attr.define(repr=False)
class Result(pg.tuning.Result):
  """Vizier tuner progress."""

  _converter: converters.VizierConverter = attr.field()
  _problem: vz.ProblemStatement = attr.field()
  _study: client_abc.StudyInterface = attr.field()
  _metadata: pg.Dict = attr.field()
  _best_trial: pg.tuning.Trial = attr.field()
  _trials: Sequence[pg.tuning.Trial] = attr.field()
  _num_trials_by_status: dict[str, int] = attr.field()
  _last_update_time: datetime.datetime = attr.field(
      factory=datetime.datetime.now
  )

  @classmethod
  def from_study(cls, study: client_abc.StudyInterface):
    # Recover DNA spec
    logging.info('from study..')
    problem = study.materialize_problem_statement()
    logging.info('got metadata.')
    metadata = converters.get_pyglove_study_metadata(problem)
    converter = converters.VizierConverter.from_problem(problem)

    logging.info('Getting trials..')
    best_trial = VizierTrial(converter, next(study.optimal_trials().get()))
    tuner_trials = [VizierTrial(converter, t) for t in study.trials().get()]
    logging.info('Got trials...')
    num_trials_by_status = dict(
        collections.Counter(t.status for t in tuner_trials)
    )

    return cls(
        converter,
        problem,
        study,
        metadata,
        best_trial,
        tuner_trials,
        num_trials_by_status,
    )

  @property
  def last_updated(self) -> datetime.datetime:
    """Last update time."""
    return self._last_update_time

  @property
  def is_active(self) -> bool:
    """Returns whether tuner is active."""
    state = self._study.materialize_state()
    active = state == vz.StudyState.ACTIVE
    logging.info('is_active was called. state:%s, active:%s', state, active)
    return active

  @property
  def metadata(self) -> dict[str, Any]:
    """Gets metadata for current sampling."""
    return self._metadata

  @property
  def best_trial(self) -> Optional[pg.tuning.Trial]:
    """Returns the best trial."""
    return self._best_trial

  @property
  def trials(self) -> Sequence[pg.tuning.Trial]:
    """Returns trials."""
    return self._trials

  def format(
      self,
      compact: bool = False,
      verbose: bool = True,
      root_indent: int = 0,
      **kwargs,
  ):
    # Return summary.
    status_field = pg.tuning.Trial.schema.get_field('status')
    assert status_field is not None
    possible_status = set(
        typing.cast(pg.typing.Enum, status_field.value).values
    )
    study = f'{self._study.resource_name}'
    # TODO: Add link to the study.
    status = {
        s: f'{self._num_trials_by_status[s]}/{len(self._trials)}'
        for s in possible_status
        if s in self._num_trials_by_status
    }
    json_repr = dict(study=study, status=status)
    if self._best_trial:
      json_repr['best_trial'] = dict(
          id=self._best_trial.id,
          reward=self._best_trial.final_measurement.reward,
          step=self._best_trial.final_measurement.step,
          dna=self._best_trial.dna.format(compact=True),
      )
    return pg.format(json_repr, compact, False, root_indent, **kwargs)
