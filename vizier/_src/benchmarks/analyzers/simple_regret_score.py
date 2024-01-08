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

"""Simple regret score to compare algorithms."""

from typing import Union

import numpy as np
from scipy import stats
from vizier import pyvizier as vz


def t_test_mean_score(baseline_mean_values: Union[list[float], np.ndarray],
                      candidate_mean_values: Union[list[float], np.ndarray],
                      objective_goal: vz.ObjectiveMetricGoal) -> float:
  """Computes the one-sided T-test score.

  In case of a maximization (minimizatoin) problem, it scores the confidence
  that the mean of 'baseline_mean_values' is less (greater) than the mean of
  'candidate_mean_values'.

  The lower the score the higher the confidence that it's the case.

  One-sample
  ----------
  The test assumes t-distribution with mean given by 'candidate' and compute
  the probability of observing sample mean of 'baseline' or less.

  Two-sample
  ----------
  The test assumes 'baseline' and 'candidate' have the same mean and computes
  the probability that the 'baseline' sample mean is less than the 'candidate'
  sample mean.

  The lower the T-test p-value score the more confidence we have that
  'candidate' is indeed "better" than 'baseline'.

  Arguments:
    baseline_mean_values: List of baseline simple regret values.
    candidate_mean_values: List of candidate simple regret values.
    objective_goal: The optimization problem type (MAXIMIZE or MINIMIZE).

  Returns:
    The p-value score of the one-sided T test.
  """
  if objective_goal == vz.ObjectiveMetricGoal.MAXIMIZE:
    alternative = 'less'
  else:
    alternative = 'greater'

  if len(candidate_mean_values) == 1:
    return stats.ttest_1samp(
        a=baseline_mean_values,
        popmean=candidate_mean_values[0],
        alternative=alternative).pvalue
  else:
    # use Welch’s t-test
    return stats.ttest_ind(
        baseline_mean_values,
        candidate_mean_values,
        equal_var=False,
        alternative=alternative).pvalue
