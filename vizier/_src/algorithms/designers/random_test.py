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

"""Tests for random."""

import collections
import math

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.testing import test_runners
from vizier.testing import test_studies

from absl.testing import absltest


class RandomTest(absltest.TestCase):

  def test_on_flat_space(self):
    config = vz.ProblemStatement(test_studies.flat_space_with_all_types())
    designer = random.RandomDesigner(config.search_space, seed=None)
    self.assertLen(
        test_runners.run_with_random_metrics(
            designer, config, iters=50, batch_size=1), 50)

  def test_reproducible_random(self):
    config = vz.ProblemStatement(test_studies.flat_space_with_all_types())
    designer = random.RandomDesigner(config.search_space, seed=5)
    t1 = designer.suggest(10)

    designer = random.RandomDesigner(config.search_space, seed=5)
    t2 = designer.suggest(10)
    self.assertEqual(t1, t2)

  def test_log_scaling(self):
    """Confirm that LINEAR and LOG scaling give the right distribution."""
    config = vz.ProblemStatement(test_studies.flat_space_with_all_types())
    designer = random.RandomDesigner(config.search_space, seed=None)
    trials = test_runners.run_with_random_metrics(
        designer, config, iters=1000, batch_size=1
    )
    # LINEAR scaling.
    sum_lineardouble = 0.0
    for t in trials:
      sum_lineardouble += t.parameters['lineardouble'].value
    avg_lineardouble = sum_lineardouble / len(trials)
    # Delta=0.16 corresponds to ~6 standard deviations, so the probability of
    # accidental test failure is ~1e-6.
    self.assertAlmostEqual(avg_lineardouble, 0.5 * (-1 + 2), delta=0.16)
    # LOG scaling
    sum_log_logdouble = 0.0
    sum_logdouble = 0.0
    for t in trials:
      sum_logdouble += t.parameters['logdouble'].value
      sum_log_logdouble += math.log(t.parameters['logdouble'].value)
    avg_logdouble = sum_logdouble / len(trials)
    avg_log_logdouble = sum_log_logdouble / len(trials)
    # If the distribution were LINEAR, we'd expect avg_logdouble = 50;
    # if the distribution were LOG, we'd expect avg_logdouble = 7.2, so
    # we check that avg_logdouble is closer to the LOG expected value.
    self.assertLess(abs(avg_logdouble - 7.2), abs(avg_logdouble - 50))
    # The average of log(value) should be close to the average of
    # the logs of the endpoints.  $delta is set at 6 standard deviations to
    # yield a false failure rate of about 1e-6.
    self.assertAlmostEqual(
        avg_log_logdouble, 0.5 * (math.log(1e-4) + math.log(1e2)), delta=0.76
    )

  def test_discrete_parameter(self):
    """Confirm that DISCRETE parameters are sampled uniformally."""
    n_samples = 10_000
    search_space = vz.SearchSpace()
    search_space.root.add_discrete_param(
        'discrete1', feasible_values=[0, 70, 71, 72, 100]
    )
    search_space.root.add_discrete_param('discrete2', feasible_values=[0, 8, 9])
    random_designer = random.RandomDesigner(search_space, seed=1)
    values = [
        (sg.parameters['discrete1'].value, sg.parameters['discrete2'].value)
        for sg in random_designer.suggest(n_samples)
    ]
    counts = collections.defaultdict(int)
    for value in values:
      counts[str(value[0]) + str(value[1])] += 1
    for count in counts.values():
      # Check that there's no more than 1% deviation.
      self.assertLess(abs(count / n_samples - 1 / 15), 0.01)


if __name__ == '__main__':
  absltest.main()
