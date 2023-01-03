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

import random

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.testing import test_runners
from vizier.testing import test_studies

from absl.testing import absltest


class HaltonTest(absltest.TestCase):

  def test_generate_primes(self):
    primes = quasi_random._generate_primes(100)
    # For each prime, make sure it is not evenly divisible by any number less
    # than itself.
    for p in primes:
      for n in range(2, int(p**0.5) + 1):
        self.assertNotEqual(0, p % n)

  def test_get_scrambled_halton_element(self):
    generator = quasi_random._HaltonSequence(
        num_dimensions=4, skip_points=100, scramble=True)
    sequence = [generator.get_next_list() for i in range(100)]

    self.assertLess(max(max(sequence)), 1)
    self.assertGreater(min(min(sequence)), 0)
    self.assertLen(sequence, 100)
    self.assertLen(sequence[0], 4)

  def test_deterministic(self):
    generator_1 = quasi_random._HaltonSequence(
        num_dimensions=4, skip_points=100, primes_override=[3, 5, 7, 11])
    generator_2 = quasi_random._HaltonSequence(
        num_dimensions=4, skip_points=100, primes_override=[3, 5, 7, 11])
    self.assertEqual([generator_1.get_next_list() for i in range(100)],
                     [generator_2.get_next_list() for i in range(100)])

  def test_unscrambled_sequence(self):
    generator = quasi_random._HaltonSequence(
        num_dimensions=1, skip_points=0, primes_override=[3], scramble=False)
    sequence = [generator.get_next_list()[0] for _ in range(7)]

    expected_sequence = [
        1 / 3,
        2 / 3,
        1 / 9,
        4 / 9,
        7 / 9,
        2 / 9,
        5 / 9,
    ]

    self.assertSequenceAlmostEqual(sequence, expected_sequence)


class QuasiRandomTest(absltest.TestCase):

  def test_on_flat_space(self):
    problem = vz.ProblemStatement(test_studies.flat_space_with_all_types())
    designer = quasi_random.QuasiRandomDesigner(problem.search_space)
    self.assertLen(
        test_runners.run_with_random_metrics(
            designer, problem, iters=50, batch_size=5), 250)

  def test_dump_and_load(self):
    # Check metadata checkpointing.
    problem = vz.ProblemStatement(test_studies.flat_space_with_all_types())

    designer = quasi_random.QuasiRandomDesigner(problem.search_space)
    designer.suggest(random.randint(1, 50))
    metadata = designer.dump()

    designer2 = quasi_random.QuasiRandomDesigner(problem.search_space)
    designer2.load(metadata)

    num_suggestions = 10
    self.assertEqual(
        designer.suggest(num_suggestions), designer2.suggest(num_suggestions))


if __name__ == '__main__':
  absltest.main()
