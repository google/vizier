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

"""Spatio-temporal converters."""

import collections
import copy
import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from absl import logging
import numpy as np
from vizier import pyvizier
from vizier.pyvizier.converters import core


@dataclasses.dataclass
class TimedLabels:
  """Corresponds to a *single* Trial's (intermediate) measurements.

  Let M be the number of measurement a trial had, Then,
    time: (M, 1) array of timestamps (See StudyMerger.to_time_and_labels_array
     pydoc for details.)
    labels: (M, 1) arrays keyed by strings, corresponding to metrics.
  """
  times: np.ndarray
  labels: Dict[str, np.ndarray]


class TimedLabelsExtractor:
  """Extracts TimeLabels."""

  CUMMAX = 'cummax'
  CUMMAX_LASTONLY = 'cummax_lastonly'
  CUMMAX_FIRSTONLY = 'cummax_firstonly'
  RAW = 'raw'

  def __init__(self,
               metric_converters: Sequence[core.ModelOutputConverter],
               timestamp: str = 'steps',
               *,
               temporal_index_points: Sequence[float] = tuple(),
               value_extraction: str = 'cummax_lastonly'):
    """Init.

    Args:
      metric_converters:
      timestamp: One of ['steps', 'elapsed_secs', 'index'].
      temporal_index_points: If provided, extract at the provided temporal index
        points.
      value_extraction: One of ['raw', 'cummax','cummax_lastonly',
        'cummax_firstonly'].
        'cummax': take the best value up to the time. For example, a
          maximization metric raw values
          (2, 1, 0, 3, 3, 2, 4, 2, 1) will be converted to: (2, 2, 2, 3, 3, 3,
            4, 4, 4).
        'cummax_lastonly': Every time there is an improvement, record the
          measurement before that improvement. For example, a maximization
          metric raw values
          (2, 1, 0, 3, 3, 2, 4, 2, 1) will be converted to: (_, _, 2, _, _, 3,
            _, _, 4).
        'cummax_firstonly':  Every time there is an improvement, record that
          value. Discard the rest except the very last measurement, so that
          stopping algorithms don't keep running a trial that's stuck in a
          plateau. For example, a maximization metric raw values
          (2, 1, 0, 3, 3, 2, 4, 2, 1) will be converted to: (2, _, _, 3, _, _,
            4, _, 4).
    """
    self.metric_converters = metric_converters
    self.temporal_index_points = np.asarray(temporal_index_points).squeeze()
    self.value_extraction = value_extraction
    self.timestamp = timestamp

    if value_extraction not in (self.CUMMAX, self.RAW, self.CUMMAX_LASTONLY,
                                self.CUMMAX_FIRSTONLY):
      raise ValueError(
          'Bad value for value_extraction rule: {}'.format(value_extraction))
    if value_extraction in [self.CUMMAX_LASTONLY, self.CUMMAX_FIRSTONLY]:
      if len(metric_converters) > 1:
        raise ValueError(
            '{} mode supports only a single metric.'.format(value_extraction))
      if self.temporal_index_points.size > 1:
        raise ValueError(
            '{} mode does not support fixed temporal_index_points.'.format(
                value_extraction))

  def _cummax_fn(self, metric_converter: core.ModelOutputConverter) -> Any:
    if (metric_converter.metric_information.goal ==
        pyvizier.ObjectiveMetricGoal.MAXIMIZE):
      return np.maximum
    else:
      return np.minimum

  def convert(self, trials: Sequence[pyvizier.Trial]) -> List[TimedLabels]:
    """Converts each trial into TimedLabels object."""
    timedlabels = []
    if self.temporal_index_points.size == 0:
      # Default mode. Temporal index points are not specified. Take all
      # available measurements.
      for t in trials:
        times = self.to_timestamps(t.measurements)
        labels = dict()
        for mc in self.metric_converters:
          # Process one metric at a time, to keep things simple.
          if self.value_extraction == self.CUMMAX:
            labels[mc.metric_information.name] = self._cummax_fn(mc).accumulate(
                mc.convert(t.measurements), axis=0)
          elif self.value_extraction in [
              self.CUMMAX_LASTONLY, self.CUMMAX_FIRSTONLY
          ]:
            # Squeeze into 1D to make indexing easier.
            marr = self._cummax_fn(mc).accumulate(
                mc.convert(t.measurements), axis=0).reshape(-1)
            if marr.size > 0:
              # Take the first and last index and the indices
              # where there was a delta.
              if self.value_extraction == self.CUMMAX_LASTONLY:
                delta_idx = np.concatenate(
                    [marr[:-1] < marr[1:],
                     np.array([True])])
              else:
                delta_idx = np.concatenate(
                    [np.array([True]), marr[:-1] < marr[1:]])
                delta_idx[-1] = True
            else:
              # No measurements were found.
              delta_idx = np.array([], dtype=bool)
            # Filter labels and times.
            labels[mc.metric_information.name] = marr[delta_idx][:, np.newaxis]
            times = times[delta_idx]
          else:
            assert self.value_extraction == self.RAW, 'Unreachable.'
            labels[mc.metric_information.name] = mc.convert(t.measurements)
        timedlabels.append(TimedLabels(times, labels))
    elif self.temporal_index_points.size and self.value_extraction == self.RAW:
      num_empty_trials = 0
      for t in trials:
        # Apply masks to only take observations at the temporal points.
        times = self.to_timestamps(t.measurements)  # shape T x 1
        mask = np.isin(times, self.temporal_index_points).squeeze()  # shape T
        if np.sum(mask) == 0:
          num_empty_trials += 1

        # Apply the mask and extract the labels.
        measurements = np.array(t.measurements)[mask]
        times = times[mask]
        labels = dict()
        for mc in self.metric_converters:
          labels[mc.metric_information.name] = mc.convert(measurements)
        timedlabels.append(TimedLabels(times, labels))
      if num_empty_trials:
        logging.warning(
            '%s Out of %s trials had zero measurement after masking.',
            num_empty_trials, len(trials))
    else:
      assert self.value_extraction == self.CUMMAX, 'Unreachable.'
      for trial in trials:
        times = self.to_timestamps(trial.measurements)  # shape T x 1

        # mask[i] is the index of the first measurement that is taken after the
        # temporal_index_points[i].
        mask = []
        for t in self.temporal_index_points:
          greater_indices = np.where(t >= times)
          if not greater_indices:
            mask.append(0)
          else:
            mask.append(greater_indices[-1])
        mask = np.asarray(mask)

        # Extract the labels and apply the mask.
        labels = dict()
        for mc in self.metric_converters:
          labels[mc.metric_information.name] = self._cummax_fn(mc).accumulate(
              mc.convert(trial.measurements), axis=0)[mask]
        timedlabels.append(TimedLabels(times, labels))

    return timedlabels

  def to_timestamps(self,
                    measurements: Sequence[pyvizier.Measurement]) -> np.ndarray:
    """"Returns an arry of shape (len(measurements), 1)."""
    timestamps = []

    for idx, measurement in enumerate(measurements):
      if self.timestamp == 'steps':
        timestamps.append(measurement.steps)
      if self.timestamp == 'elapsed_secs':
        timestamps.append(measurement.elapsed_secs)
      if self.timestamp == 'index':
        timestamps.append(idx)
    return np.asarray(timestamps)[:, np.newaxis]

  def extract_all_timestamps(
      self, trials: Sequence[pyvizier.Trial]) -> Sequence[float]:
    """Returns a sorted list of unique temporal indices occurring in trials.

    This is a cheaper alternative to:
    ```
    ts = np.concatenate([np.asarray(tl.times).flatten() for tl in timed_labels])
    return sorted(list(set(ts)))
    ```
    because it skips the label conversions.

    Args:
      trials:

    Returns:
      Sorted list of temporal indices.
    """
    all_timestamps = np.concatenate(
        [self.to_timestamps(t.measurements).flatten() for t in trials])
    return sorted(list(set(all_timestamps)))


class SparseSpatioTemporalConverter(core.TrialToNumpyDict):
  """Optimized for when the temporal dimensions are not well-aligned.

  This converter inherits from the `TrialToNumpyDict` abstractions. Timestamp
  becomes an extra feature dimension.

  For inference, however, the user has to provide `temporal_index_points` to
  specify where to predict a Trial at.
  """

  def __init__(self,
               parameter_converters: Sequence[core.DefaultModelInputConverter],
               timed_labels_extractor: TimedLabelsExtractor):
    self.trial_converter = core.DefaultTrialConverter(
        parameter_converters, timed_labels_extractor.metric_converters)
    self.timed_labels_extractor = timed_labels_extractor

  def to_xy(
      self, trials: Sequence[pyvizier.Trial]
  ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Returned values can be used as `x` and `y` for keras.Model.fit().

    Args:
      trials:

    Returns:
      Tuple. Both have length equal to the total number of measurements.
      x values have an extra element of (None, 1). y values are the same as
      metric_converters.
    """
    timed_labels = self.timed_labels_extractor.convert(trials)
    all_features = dict()
    all_labels = dict()
    for trial, tl in zip(trials, timed_labels):
      # Let T_i be the number of temporal observations for this trial.
      # tl.times and tl.labels have shape [T_i, 1]
      features = self.to_features(trial, tl.times)
      # Add to the return values.
      for k in features:
        if k in all_features:
          all_features[k] = np.concatenate([all_features[k], features[k]],
                                           axis=0)
        else:
          all_features[k] = features[k]

      labels = tl.labels
      for k in labels:
        if k in all_labels:
          all_labels[k] = np.concatenate([all_labels[k], labels[k]], axis=0)
        else:
          all_labels[k] = labels[k]

    return all_features, all_labels

  def to_features(self, trial, temporal_index_points) -> Dict[str, np.ndarray]:
    """Converts a trial at the specified index points.

    Args:
      trial:
      temporal_index_points:

    Returns:
      Each array has shape (B,_) where _ is determined by injected
      ModelInputConverters and B = len(trials) * len(temporal_index_points).
      In addition to what ModelInputConverters return, there is an extra array
      of shape (B, 1) corresponding to the time stamps.
    """
    features = self.trial_converter.to_features(
        [trial])  # Dict values have shape [1,1]
    # Repeat the features T_i times. Dict values have shape [T_i,1]
    features = {
        k: np.tile(v, (len(temporal_index_points), 1))
        for k, v in features.items()
    }

    # Add the timestamp feature.
    features[self.timed_labels_extractor.timestamp] = temporal_index_points
    return features

  @property
  def features_shape(self) -> Dict[str, Sequence[Union[int, None]]]:
    """Returned value can be used as `input_shape` for keras.Model.build()."""
    shapes = copy.deepcopy(self.trial_converter.features_shape)
    shapes[self.timed_labels_extractor.timestamp] = (None, 1)
    return shapes

  @property
  def output_specs(self) -> Dict[str, core.NumpyArraySpec]:
    specs = copy.deepcopy(self.trial_converter.output_specs)
    name = self.timed_labels_extractor.timestamp
    # Can't use float32 max, because
    # isinstance(np.finfo(np.float32).max, float) == False
    specs[name] = core.NumpyArraySpec.from_parameter_config(
        pyvizier.ParameterConfig.factory(
            name=name, bounds=(0.0, np.finfo(float).max)),
        core.NumpyArraySpecType.default_factory)
    return specs

  @property
  def labels_shape(self) -> Dict[str, Sequence[Union[int, None]]]:
    return self.trial_converter.labels_shape

  @property
  def metric_information(self) -> Dict[str, pyvizier.MetricInformation]:
    return self.trial_converter.metric_information


class DenseSpatioTemporalConverter(core.TrialToNumpyDict):
  """Optimized for when the temporal dimensions are well-aligned.

  This converter inherits from the `TrialToNumpyDict` abstraction. Each label
  in the dict has shape [number of trials, number of temproal index points].
  To operate in this mode, temporal_index_points must be specified.

  We can also use `to_xty()` instead of `to_xy()` in an alternative workflow
  that does not depend on the `TrialToNumpyDict` interface.
  """

  def __init__(self,
               parameter_converters: Sequence[core.DefaultModelInputConverter],
               timed_labels_extractor: TimedLabelsExtractor,
               temporal_index_points: Optional[np.ndarray] = None):
    self.trial_converter = core.DefaultTrialConverter(
        parameter_converters, timed_labels_extractor.metric_converters)
    self.timed_labels_extractor = timed_labels_extractor
    if temporal_index_points is None:
      logging.info('Empty temporal_index_points.')
      self.temporal_index_points = np.zeros([0])
    else:
      self.temporal_index_points = temporal_index_points

  def _single_timedlabels_to_temporal_observations(
      self, timed_labels: TimedLabels,
      ts: Union[Sequence[float], Sequence[int]]) -> Dict[str, np.ndarray]:
    """Subroutine of _to_temporal_observations().

    Args:
      timed_labels: A single TimedLabels object.
      ts: Length T sequence of timestamps.

    Returns:
      Dict of metric names to array of shape (T).
    """

    # Reshape things into 1-D array, so we can iterate through them like
    # python list.
    ts = np.asarray(ts).reshape([-1])
    if ts.size == 0:
      # Caller asked for nothing.
      return {k: np.array([]) for k in timed_labels.labels}

    observed_times, labels_dict = timed_labels.times.reshape(
        [-1]), timed_labels.labels
    if observed_times.size == 0:
      # Nothing was observed.
      return {k: np.zeros_like(ts) * np.nan for k in timed_labels.labels}

    query_time_iter = iter(ts)
    query_time = next(query_time_iter)
    observed_time_iter = iter(enumerate(observed_times))
    idx, observed_time = next(observed_time_iter)
    this_labels = collections.defaultdict(list)

    while query_time is not None:
      if query_time == observed_time:
        # Times match. add it.
        for key in labels_dict:
          this_labels[key].append(labels_dict[key][idx, 0])
        query_time = next(query_time_iter, None)
        idx, observed_time = next(observed_time_iter, (None, None))
      elif observed_time is None or observed_time > query_time:
        # query_time is behind, or we've scanned through all observations.
        # In either case, we don't have observed data at query_time.
        # Fill with nan and advance query_time.
        query_time = next(query_time_iter, None)
        for key in labels_dict:
          this_labels[key].append(np.nan)
      else:
        # query_time is ahead.
        idx, observed_time = next(observed_time_iter, (None, None))
    return {k: np.asarray(v) for k, v in this_labels.items()}

  def _to_temporal_observations(
      self, timed_labels_sequence: Sequence[TimedLabels],
      ts: Union[Sequence[float], Sequence[int]]) -> Dict[str, np.ndarray]:
    """Returns a dict of np arrays of temporal observations.

    Filters timed_labels.labels to leave only the time indices that appear in
    ts.

    Args:
      timed_labels_sequence: Length B sequence.
      ts: Length T sequence.

    Returns:
      Dict of metric names to array of shape (B, T).
    """

    all_labels = collections.defaultdict(lambda: np.zeros(shape=(0, len(ts))))

    for tl in timed_labels_sequence:
      labels = self._single_timedlabels_to_temporal_observations(tl, ts)
      # labels have shape [T]. Add a new axis to make it (1,T) so we
      # can concat.
      labels = {k: v[np.newaxis, :] for k, v in labels.items()}
      for k in labels:
        all_labels[k] = np.concatenate([all_labels[k], labels[k]], axis=0)
    return all_labels

  def to_xy(
      self, trials: Sequence[pyvizier.Trial]
  ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Returned value can be used as `x`, `y` for keras.Model.fit()."""
    all_timed_labels = self.timed_labels_extractor.convert(trials)
    labels = self._to_temporal_observations(all_timed_labels,
                                            self.temporal_index_points)

    return self.trial_converter.to_features(trials), labels

  @property
  def features_shape(self) -> Dict[str, Sequence[Union[int, None]]]:
    """Returned value can be used as `input_shape` for keras.Model.build()."""
    return self.trial_converter.features_shape

  @property
  def output_specs(self) -> Dict[str, core.NumpyArraySpec]:
    return self.trial_converter.output_specs

  @property
  def labels_shape(self) -> Dict[str, Sequence[Union[int, None]]]:
    shapes = dict()
    for mc in self.trial_converter.metric_converters:
      shapes[mc.metric_information.name] = (None,
                                            len(self.temporal_index_points))
    return shapes

  @property
  def metric_information(self) -> Dict[str, pyvizier.MetricInformation]:
    return self.trial_converter.metric_information

  def to_features(self,
                  trials: Sequence[pyvizier.Trial]) -> Dict[str, np.ndarray]:
    return self.trial_converter.to_features(trials)

  def to_xty(
      self, trials: Sequence[pyvizier.Trial], temporal_selection: str = 'auto'
  ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
    """Returns x, t, and y.

    Args:
      trials:
      temporal_selection: 'infer', 'default', 'auto'. If 'infer', values are
        auto-inferred based on timestamps that exist in `trials`. If 'default',
        use the class-wide `self.temporal_index_points`. If 'auto', auto-infer
        only if necessary.

    Returns:
      3-tuple of (input, temporal_index_points, observations).
      input: A Dict whose values correspond to parameters.
      temporal_index_points: 1-D array of temporal index points. If provided
        as input, then it's the same as the input value.
      observations: Dict of length equal to metrics, whose values
        have shape [len(trials), len(temporal_index_points)]. May contain NaNs.
    """
    timed_labels: List[TimedLabels] = (
        self.timed_labels_extractor.convert(trials))

    if temporal_selection == 'default' or (temporal_selection == 'auto' and
                                           self.temporal_index_points.size):
      temporal_index_points = self.temporal_index_points
    elif temporal_selection == 'infer' or (temporal_selection == 'auto' and
                                           not self.temporal_index_points.size):
      temporal_index_points = sorted(
          list(
              set(
                  np.concatenate([
                      np.asarray(tl.times).flatten() for tl in timed_labels
                  ]))))
    else:
      raise ValueError('Invalid value for temporal_index_points: '
                       f'{temporal_selection}.')

    observations = self._to_temporal_observations(timed_labels,
                                                  temporal_index_points)
    inputs = self.to_features(trials)
    return inputs, np.asarray(temporal_index_points), observations
