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

from vizier._src.algorithms.designers import pycmaes
from vizier._src.algorithms.testing import test_runners
from vizier.benchmarks import experimenters

from absl.testing import absltest
from absl.testing import parameterized


class PycmaesTest(parameterized.TestCase):

  def setUp(self):
    self.experimenter = experimenters.BBOBExperimenterFactory("Sphere", 2)()
    super().setUp()

  @parameterized.parameters(
      dict(batch_size=1),
      dict(batch_size=1, popsize=7),
      dict(batch_size=3),
      dict(batch_size=3, popsize=2),
      dict(batch_size=3, popsize=5),
  )
  def test_e2e(self, batch_size: int, popsize: int | None = None):
    designer = pycmaes.PyCMAESDesigner(
        self.experimenter.problem_statement(), popsize=popsize
    )

    trials = test_runners.run_with_random_metrics(
        designer,
        self.experimenter.problem_statement(),
        iters=10,
        batch_size=batch_size,
        verbose=1,
        validate_parameters=True,
    )
    self.assertLen(trials, batch_size * 10)

  def test_invalid_popsize(self):
    with self.assertRaisesRegex(ValueError, "Popsize must be at least 2"):
      pycmaes.PyCMAESDesigner(self.experimenter.problem_statement(), popsize=1)


if __name__ == "__main__":
  absltest.main()
