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

from vizier._src.benchmarks.experimenters.synthetic import multiobjective_optproblems as mopt
from vizier._src.benchmarks.testing import experimenter_testing
from absl.testing import absltest
from absl.testing import parameterized


class WFGxperimenterTest(parameterized.TestCase):

  @parameterized.parameters(
      ("WFG1",),
      ("WFG2",),
      ("WFG3",),
      ("WFG4",),
      ("WFG5",),
      ("WFG6",),
      ("WFG7",),
      ("WFG8",),
      ("WFG9",),
  )
  def test_e2e_smoke(self, name):
    exptr_factory = mopt.WFGExperimenterFactory(name, dim=6, num_objectives=3)
    experimenter_testing.assert_evaluates_random_suggestions(
        self, exptr_factory()
    )


class DTLZExperimenterTest(parameterized.TestCase):

  @parameterized.parameters(
      ("DTLZ1",),
      ("DTLZ2",),
      ("DTLZ3",),
      ("DTLZ4",),
      ("DTLZ5",),
      ("DTLZ6",),
      ("DTLZ7",),
  )
  def test_e2e_smoke(self, name):
    exptr_factory = mopt.DTLZExperimenterFactory(name, dim=4, num_objectives=3)
    experimenter_testing.assert_evaluates_random_suggestions(
        self, exptr_factory()
    )


class ZDTExperimenterTest(parameterized.TestCase):

  @parameterized.parameters(
      ("ZDT1",),
      ("ZDT2",),
      ("ZDT3",),
      ("ZDT4",),
      ("ZDT6",),
  )
  def test_e2e_smoke(self, name):
    exptr_factory = mopt.ZDTExperimenterFactory(name, dim=5)
    experimenter_testing.assert_evaluates_random_suggestions(
        self, exptr_factory()
    )


if __name__ == "__main__":
  absltest.main()
