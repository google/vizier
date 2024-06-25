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

from vizier._src.benchmarks.experimenters.synthetic import deb
from vizier._src.benchmarks.testing import experimenter_testing
from absl.testing import absltest
from absl.testing import parameterized


class DHExperimenterTest(parameterized.TestCase):

  @parameterized.parameters(
      (deb.DHExperimenter.DH1,),
      (deb.DHExperimenter.DH2,),
      (deb.DHExperimenter.DH3,),
      (deb.DHExperimenter.DH4,),
  )
  def test_e2e_smoke(self, experimenter_cls):
    experimenter_testing.assert_evaluates_random_suggestions(
        self, experimenter_cls(num_dimensions=4)
    )


if __name__ == "__main__":
  absltest.main()
