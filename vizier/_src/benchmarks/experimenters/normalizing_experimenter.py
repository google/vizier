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

"""Experimenter that normalizes the range of each metric."""

import collections
import copy
from typing import Dict, Sequence

import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter
from vizier.pyvizier import converters


# TODO: Allow more sampling methods.
class NormalizingExperimenter(experimenter.Experimenter):
  """Normalizes an Experimenter output via (y - empirical_mean) / empirical_std."""

  def __init__(
      self,
      exptr: experimenter.Experimenter,
      num_normalization_samples: int = 100,
      noise_seed: int = 42,
  ):
    """Normalizing experimenter uses a grid to estimate a normalization constant.

    Args:
      exptr: Experimenter to be normalized.
      num_normalization_samples: Number of samples to determine normalization.
      noise_seed: Random seed for tiebreaking categorical samples.
    """
    self._exptr = exptr
    self._problem_statement = exptr.problem_statement()

    # Convert search space into hypercube, then sample.
    converter = converters.TrialToArrayConverter.from_study_config(
        study_config=self._problem_statement, scale=True, pad_oovs=False
    )
    feature_dim = sum([o.num_dimensions for o in converter.output_specs])

    normalized_samples = np.linspace(
        np.zeros(feature_dim), np.ones(feature_dim), num_normalization_samples
    )

    # Categoricals use argmax and break ties by always choosing the first
    # feasible choice. Let's randomize the tiebreaking.
    noise = np.random.RandomState(noise_seed).normal(
        scale=1e-6, size=(num_normalization_samples, feature_dim)
    )
    normalized_samples += noise

    sampled_params = converter.to_parameters(normalized_samples)
    metrics = collections.defaultdict(list)
    for parameters in sampled_params:
      trial = vz.Trial(parameters=parameters)
      exptr.evaluate([trial])
      measurement = trial.final_measurement
      for name, metric in (measurement.metrics if measurement else {}).items():
        metrics[name].append(metric.value)

    self._norm_means: Dict[str, float] = {}
    self._norm_stds: Dict[str, float] = {}
    for name, grid_values in metrics.items():
      self._norm_means[name] = np.mean(grid_values)
      self._norm_stds[name] = np.std(grid_values) + 1e-7

  def problem_statement(self) -> vz.ProblemStatement:
    return copy.deepcopy(self._problem_statement)

  def evaluate(self, suggestions: Sequence[vz.Trial]):
    self._exptr.evaluate(suggestions)
    for suggestion in suggestions:
      if suggestion.final_measurement is None:
        continue
      normalized_metrics: Dict[str, vz.Metric] = {}
      for name, metric in suggestion.final_measurement.metrics.items():
        norm_val = metric.value - self._norm_means[name]
        norm_val /= self._norm_stds[name]
        normalized_metrics[name] = vz.Metric(norm_val)
      suggestion.final_measurement.metrics = normalized_metrics

  def __repr__(self):
    return (
        'NormalizingExperimenter with normalization means'
        f' {self._norm_means} and stds {self._norm_stds} on {self._exptr}'
    )


class HyperCubeExperimenter(experimenter.Experimenter):
  """Normalizes the search space into unit hypercube using converter."""

  def __init__(self, exptr: experimenter.Experimenter):
    self._exptr = exptr
    original_problem = exptr.problem_statement()

    converter = converters.TrialToArrayConverter.from_study_config(
        study_config=original_problem, scale=True, pad_oovs=False
    )
    self._converter = converter
    feature_dim = sum([o.num_dimensions for o in converter.output_specs])

    new_space = vz.SearchSpace()  # Setup hypercube search space.
    for i in range(feature_dim):
      new_space.add(vz.ParameterConfig.factory(f'h{i}', bounds=(0.0, 1.0)))
    self._problem_statement = copy.deepcopy(original_problem)
    self._problem_statement.search_space = new_space

    # For simply converting new hypercube parameters to numpy arrays.
    self._eval_converter = converters.TrialToArrayConverter.from_study_config(
        study_config=self._problem_statement, scale=False, pad_oovs=False
    )

  def problem_statement(self) -> vz.ProblemStatement:
    return copy.deepcopy(self._problem_statement)

  def evaluate(self, suggestions: Sequence[vz.Trial]):
    hypercube_features = self._eval_converter.to_features(suggestions)
    original_params = self._converter.to_parameters(hypercube_features)

    orig_suggestions = copy.deepcopy(suggestions)
    for param_dict, orig_suggestion in zip(original_params, orig_suggestions):
      orig_suggestion.parameters = param_dict

    self._exptr.evaluate(orig_suggestions)

    for suggestion, orig_suggestion in zip(suggestions, orig_suggestions):
      suggestion.final_measurement = orig_suggestion.final_measurement
