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

"""Tests for pythia."""

import functools

import mock
import pyglove as pg
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.algorithms.testing import test_runners
from vizier._src.pyglove import converters
from vizier._src.pyglove import pythia as pg_pythia
from vizier._src.pyglove import vizier_test_lib

from absl.testing import absltest

_RandomAlgorithm = vizier_test_lib.RandomAlgorithm


class TunerPolicyTest(absltest.TestCase):

  def test_stopping_policy(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('x', 0.0, 1.0)
    problem.metric_information.append(
        vz.MetricInformation(name='r', goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    )

    supporter = pythia.InRamPolicySupporter(problem)

    m = mock.create_autospec(pg.tuning.EarlyStoppingPolicy, instance=True)
    policy = pg_pythia.create_policy(
        supporter,
        problem,
        algorithm=_RandomAlgorithm(),
        early_stopping_policy=m,
    )

    supporter.AddTrials([
        vz.Trial(
            parameters={'x': 0.0},
            final_measurement=vz.Measurement({'r': 1.0}),
        ),
        vz.Trial(
            parameters={'x': 0.0},
            final_measurement=vz.Measurement({'r': 1.0}),
        ),
        vz.Trial(
            parameters={'x': 0.0},
            final_measurement=vz.Measurement({'r': 1.0}),
        ),
        vz.Trial(
            parameters={'x': 0.0},
            measurements=[vz.Measurement({'r': 1.0})],
        ),
    ])

    descriptor = supporter.study_descriptor()

    m.should_stop_early.return_value = False
    res = policy.early_stop(
        pythia.EarlyStopRequest(study_descriptor=descriptor, trial_ids=[4])
    )
    self.assertFalse(res.decisions[0].should_stop)

    m.should_stop_early.return_value = True
    res = policy.early_stop(
        pythia.EarlyStopRequest(study_descriptor=descriptor, trial_ids=[4])
    )
    self.assertTrue(res.decisions[0].should_stop)
    self.assertEqual(m.should_stop_early.call_count, 5)  # 3 + 1 + 1

  def test_random_algorithm_on_simple_search_space(self):
    """geno.Random wrapped into TunerPolicy should generate valid trials."""
    # Get a DNA spec.
    rewards = []

    def foo():
      r = pg.oneof([1, 2]) + pg.floatv(3., 4.)
      branch = pg.oneof(['negate', 'no-op'])
      if branch == 'negate':
        r = -r
      rewards.append(r)
      return r

    search_space = pg.hyper.DynamicEvaluationContext()
    with search_space.collect():
      foo()

    # Create a vizier converter.
    converter = converters.VizierConverter.from_dna_spec(
        search_space.dna_spec, ('',)
    )

    # Create pyglove algorithm.
    algorithm = _RandomAlgorithm()
    algorithm.setup(search_space.dna_spec)
    policy_factory = functools.partial(
        pg_pythia.TunerPolicy,
        converter=converter,
        algorithm=algorithm,
    )
    # Test the policy.
    tester = test_runners.RandomMetricsRunner(
        converter.problem,
        batch_size=5,
        validate_parameters=True,
    )
    tester.run_policy(policy_factory)


if __name__ == '__main__':
  absltest.main()
