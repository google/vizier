# Copyright 2023 Google LLC.
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
import pyglove as pg
from vizier._src.algorithms.testing import test_runners
from vizier._src.pyglove import converters
from vizier._src.pyglove import pythia as pg_pythia
from absl.testing import absltest


class DummyAlgorithm(pg.DNAGenerator):
  """Dummy algorithm for testing."""

  def __init__(self):
    super().__init__()
    self._random = pg.generators.Random(seed=1)

  def setup(self, dna_spec: pg.DNASpec):
    super().setup(dna_spec)
    self._random.setup(dna_spec)

  def _propose(self) -> pg.DNA:
    return self._random.propose()


class PythiaTest(absltest.TestCase):

  def test_simple_search_space(self):
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
    algorithm = DummyAlgorithm()
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
