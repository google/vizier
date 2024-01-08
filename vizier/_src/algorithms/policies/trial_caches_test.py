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

from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.algorithms.policies import trial_caches
from absl.testing import absltest


def new_completed() -> vz.CompletedTrial:
  return vz.Trial(final_measurement=vz.Measurement())


def new_active() -> vz.Trial:
  return vz.Trial()


class CompletedTrialIdsCacheTest(absltest.TestCase):

  def test_all(self):
    problem = vz.ProblemStatement()
    supporter = pythia.InRamPolicySupporter(problem)
    cache = trial_caches.IdDeduplicatingTrialLoader(supporter)

    ################## PHASE 1 ################
    active_trial = new_active()
    supporter.AddTrials([new_completed(), active_trial, new_completed()])

    self.assertLen(cache.get_newly_completed_trials(3), 2)
    self.assertEqual(cache.num_incorporated_trials(), 2)

    # dump and load should restore the state.
    dump = cache.dump()
    cache = trial_caches.IdDeduplicatingTrialLoader(supporter)
    cache.load(dump)
    self.assertEmpty(cache.get_newly_completed_trials(3))
    self.assertEqual(cache.num_incorporated_trials(), 2)

    ################## PHASE 2 ################

    # Trial 2 completed
    active_trial.complete(vz.Measurement())
    # Trial 4 added as pending. 5 added as completed
    supporter.AddTrials([new_active(), new_completed()])
    self.assertLen(cache.get_newly_completed_trials(5), 2)
    self.assertLen(cache.get_active_trials(), 1)
    self.assertEqual(cache.num_incorporated_trials(), 4)

    # dump and load should restore the state.
    dump = cache.dump()
    cache = trial_caches.IdDeduplicatingTrialLoader(supporter)
    cache.load(dump)
    self.assertEmpty(cache.get_newly_completed_trials(5))
    self.assertEqual(cache.num_incorporated_trials(), 4)

    # clear should reset.
    cache.clear()
    self.assertEmpty(cache._incorporated_completed_trial_ids)
    self.assertLen(cache.get_newly_completed_trials(5), 4)


if __name__ == "__main__":
  absltest.main()
