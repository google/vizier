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

"""Tests for bocs."""
from vizier._src.algorithms.designers import bocs
from vizier._src.algorithms.testing import test_runners
from vizier._src.benchmarks.experimenters import combo_experimenter

from absl.testing import absltest
from absl.testing import parameterized


class BocsTest(parameterized.TestCase):

  @parameterized.parameters((bocs.SemiDefiniteProgramming,),
                            (bocs.SimulatedAnnealing,))
  def test_make_suggestions(self, acquisition_optimizer_factory):
    experimenter = combo_experimenter.IsingExperimenter(lamda=0.01)
    designer = bocs.BOCSDesigner(
        experimenter.problem_statement(),
        acquisition_optimizer_factory=acquisition_optimizer_factory,
        num_initial_randoms=1)

    trials = test_runners.run_with_random_metrics(
        designer,
        experimenter.problem_statement(),
        iters=5,
        batch_size=1,
        verbose=1,
        validate_parameters=True)
    self.assertLen(trials, 5)


if __name__ == '__main__':
  absltest.main()
