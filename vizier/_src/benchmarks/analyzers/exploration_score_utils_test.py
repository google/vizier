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

"""Tests for exploration_score_utils."""

from typing import Tuple

import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.analyzers import exploration_score_utils

from absl.testing import absltest
from absl.testing import parameterized


def _generate_min_and_max_ent_studies() -> (
    Tuple[vz.ProblemAndTrials, vz.ProblemAndTrials]
):
  """Generates two studies with zero and large parameter entropies."""
  space = vz.SearchSpace()
  root = space.root
  root.add_float_param('continuous', -5.0, 5.0)
  root.add_int_param('integer', -5, 5)
  root.add_categorical_param(
      'categorical', [str(v) for v in np.linspace(-5, 5, 11)]
  )
  root.add_discrete_param('discrete', list(np.linspace(-5, 5, 11)))
  problem = vz.ProblemStatement(
      search_space=space,
      metric_information=[
          vz.MetricInformation('x1', goal=vz.ObjectiveMetricGoal.MAXIMIZE),
      ],
  )
  max_ent_trials = []
  min_ent_trials = []
  values = list(np.linspace(-5, 5, 11)) * 10
  for idx, value in enumerate(values):
    # Generates trials with large marginal parameter entropies by looping
    # through the feasible values of each parameter.
    max_ent_trials.append(
        vz.Trial(
            id=idx + 1,
            parameters={
                'continuous': vz.ParameterValue(value),
                'integer': vz.ParameterValue(int(value)),
                'categorical': vz.ParameterValue(str(value)),
                'discrete': vz.ParameterValue(value),
            },
        )
    )
    # Generates trials with zero marginal parameter entropies by setting every
    # parameter to the same value.
    min_ent_trials.append(
        vz.Trial(
            id=idx + 1,
            parameters={
                'continuous': vz.ParameterValue(values[55]),
                'integer': vz.ParameterValue(int(values[17])),
                'categorical': vz.ParameterValue(str(values[96])),
                'discrete': vz.ParameterValue(values[34]),
            },
        )
    )
  return vz.ProblemAndTrials(
      problem=problem, trials=min_ent_trials
  ), vz.ProblemAndTrials(problem=problem, trials=max_ent_trials)


class ExplorationScoreUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('Continuous', vz.ParameterType.DOUBLE),
      ('Discrete', vz.ParameterType.DISCRETE),
      ('Integer', vz.ParameterType.INTEGER),
      ('Categorical', vz.ParameterType.CATEGORICAL),
  )
  def test_compute_parameter_entropy(self, parameter_type):
    def _cast(v):
      if parameter_type == vz.ParameterType.INTEGER:
        return int(v)
      elif parameter_type == vz.ParameterType.CATEGORICAL:
        return str(v)
      else:
        return v

    feasible_values = [_cast(v) for v in np.linspace(-5, 5, 11)]
    if parameter_type in [vz.ParameterType.DOUBLE, vz.ParameterType.INTEGER]:
      parameter_config = vz.ParameterConfig.factory(
          name='param',
          bounds=(-5, 5)
          if parameter_type == vz.ParameterType.INTEGER
          else (-5.0, 5.0),
      )
    else:
      parameter_config = vz.ParameterConfig.factory(
          name='param', feasible_values=feasible_values
      )
    parameter_values = [vz.ParameterValue(value=v) for v in feasible_values]
    max_ent_parameter_values = parameter_values * 10
    min_ent_parameter_values = [parameter_values[5]] * len(
        max_ent_parameter_values
    )

    max_ent = exploration_score_utils.compute_parameter_entropy(
        parameter_config=parameter_config,
        parameter_values=max_ent_parameter_values,
    )
    min_ent = exploration_score_utils.compute_parameter_entropy(
        parameter_config=parameter_config,
        parameter_values=min_ent_parameter_values,
    )
    self.assertAlmostEqual(min_ent, 0)
    if parameter_type != vz.ParameterType.DOUBLE:
      self.assertAlmostEqual(max_ent, -np.log(1 / 11))
    else:
      # For continuous parameters the entropy depends on the histogram, whose
      # number of bins is chosen based on data. There are 11 unique values with
      # equal counts, and we simply assert that the entropy is greater than
      # the entropy of two bins with equal counts.
      self.assertGreater(max_ent, -np.log(0.5))

  @parameterized.parameters(3, 5, 11, 101, 1001, 10001)
  def test_compute_parameter_entropy_uniform_full_range_gt_half_range(
      self, sample_size
  ):
    parameter_config = vz.ParameterConfig.factory(
        name='param',
        bounds=(-5.0, 5.0),
    )
    full_range_uniform_ent = exploration_score_utils.compute_parameter_entropy(
        parameter_config=parameter_config,
        parameter_values=[
            vz.ParameterValue(value=v)
            for v in np.linspace(-5.0, 5.0, sample_size)
        ],
    )
    half_range_uniform_ent = exploration_score_utils.compute_parameter_entropy(
        parameter_config=parameter_config,
        parameter_values=[
            vz.ParameterValue(value=v)
            for v in np.linspace(-5.0, 0.0, sample_size)
        ],
    )
    # The entropy of a uniform distribution is ln(range), so when the range
    # shrinks by half, the entropy is expected to decrease additively by
    # ln(2.0). Relaxes the threshold to account for estimation errors.
    self.assertGreater(
        full_range_uniform_ent - half_range_uniform_ent, 0.6 * np.log(2.0)
    )

  @parameterized.parameters(11, 101, 1001, 10001)
  def test_compute_parameter_entropy_continuous_uniform_gt_standard_normal(
      self, sample_size
  ):
    parameter_config = vz.ParameterConfig.factory(
        name='param',
        bounds=(-5.0, 5.0),
    )
    uniform_ent_est = exploration_score_utils.compute_parameter_entropy(
        parameter_config=parameter_config,
        parameter_values=[
            vz.ParameterValue(value=v)
            for v in np.linspace(-5.0, 5.0, sample_size)
        ],
    )
    standard_normal_ent_est = exploration_score_utils.compute_parameter_entropy(
        parameter_config=parameter_config,
        parameter_values=[
            vz.ParameterValue(value=v)
            for v in np.random.default_rng(seed=sample_size).normal(
                size=sample_size
            )
        ],
    )
    # The entropy is ln(range) for a uniform distribution, and
    # ln(sqrt(2 * PI * e)) for a standard Normal distribution. Relaxes the
    # expected gap to account for estimation errors.
    expected_ent_gap = 0.8 * (
        np.log(10.0) - np.log(np.sqrt(2.0 * np.pi * np.exp(1.0)))
    )
    self.assertGreater(
        uniform_ent_est - standard_normal_ent_est, expected_ent_gap
    )

  def test_compute_average_marginal_parameter_entropy(self):
    min_ent_study, max_ent_study = _generate_min_and_max_ent_studies()
    max_ent = (
        exploration_score_utils.compute_average_marginal_parameter_entropy(
            {'spec_gen': {'hash_': {0: max_ent_study}}}
        )
    )
    min_ent = (
        exploration_score_utils.compute_average_marginal_parameter_entropy(
            {'spec_gen': {'hash_': {0: min_ent_study}}}
        )
    )
    self.assertAlmostEqual(min_ent, 0)
    # The study has four parameters: continuous, integer, categorical and
    # discrete. The entropy for all parameters except the continuous are
    # exactly -np.log(1/11) since every parameter has 11 unique values with
    # equal counts. The entropy of the continuous parameter depends on the
    # histogram, whose number of bins is chosen based on data, and we simply
    # lower bound it by the entropy of two bins with equal counts.
    self.assertGreater(
        max_ent, 0.75 * (-np.log(1 / 11)) + 0.25 * (-np.log(0.5))
    )


if __name__ == '__main__':
  absltest.main()
