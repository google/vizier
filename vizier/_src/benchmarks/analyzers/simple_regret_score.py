# Copyright 2022 Google LLC.
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

"""Simple regret score to compare algorithms."""

from typing import Union

import numpy as np
from scipy import stats


def t_test_less_mean_score(
    baseline_simple_regrets: Union[list[float], np.ndarray],
    candidate_simple_regrets: Union[list[float], np.ndarray]) -> float:
  """Computes the one-sided T-test score.

  It scores the confidence that the mean of 'baseline_simple_regrets' is less
  than the mean of 'candidate_simple_regrets'. The lower the score the higher
  the confidence that it's the case.

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
    baseline_simple_regrets: list of baseline simple regret values.
    candidate_simple_regrets: list of candidate simple regret values.

  Returns:
    The p-value score of the one-sided T test.
  """
  if len(candidate_simple_regrets) == 1:
    return stats.ttest_1samp(
        a=baseline_simple_regrets,
        popmean=candidate_simple_regrets[0],
        alternative='less').pvalue
  else:
    # use Welch’s t-test
    return stats.ttest_ind(
        baseline_simple_regrets,
        candidate_simple_regrets,
        equal_var=False,
        alternative='less').pvalue
