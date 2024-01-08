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

"""Cache the list of trials so that we can avoid reloading them."""

import json
from typing import Sequence

from absl import logging
import attrs
from vizier import pythia
from vizier import pyvizier as vz
from vizier.interfaces import serializable


_INCOPORATED_COMPLETED_TRIALS_IDS = 'incorporated_completed_trials_ids'


@attrs.define
class IdDeduplicatingTrialLoader(serializable.PartiallySerializable):
  """Wrapper around PolicySupporter to avoid reloading completed trials.

  get_newly_completed_trials() only return the trials completed since the
  last call.

  get_active_trials() is just a convienence wrapper.
  """

  # TODO: Storing every id is inefficient and max_trial id
  # can be out of sync. Optimize this class.

  _supporter: pythia.PolicySupporter = attrs.field(
      repr=str,  # Use the concise representation.
  )
  _incorporated_completed_trial_ids: set[int] = attrs.field(
      kw_only=True,
      factory=set,
      repr=lambda x: repr(x) if len(x) < 100 else f'{len(x)} elements.',
  )
  _include_intermediate_measurements: bool = attrs.field(
      kw_only=True, default=False
  )

  def __attrs_post_init__(self):
    logging.info('Initialized %s', self)

  def num_incorporated_trials(self) -> int:
    return len(self._incorporated_completed_trial_ids)

  def clear(self) -> None:
    """Make the next call to get_newly_completed_trials() return all trials."""
    self._incorporated_completed_trial_ids = set()

  def get_active_trials(self) -> Sequence[vz.Trial]:
    """Returns all active trials."""
    trials = self._supporter.GetTrials(
        status_matches=vz.TrialStatus.ACTIVE,
        include_intermediate_measurements=self._include_intermediate_measurements,
    )

    logging.info(
        'Loaded %s active trials.',
        len(trials),
    )
    return trials

  def get_newly_completed_trials(self, max_trial_id: int) -> Sequence[vz.Trial]:
    """Returns trials completed between the last call and max_trial_id."""
    if len(self._incorporated_completed_trial_ids) == max_trial_id:
      # no trials need to be loaded.
      return []
    all_trial_ids = set(range(1, max_trial_id + 1))
    # Exclude completed trials that were already passed to the designer.
    trial_ids_to_load = all_trial_ids - self._incorporated_completed_trial_ids
    new_trials = self._supporter.GetTrials(
        trial_ids=trial_ids_to_load,
        status_matches=vz.TrialStatus.COMPLETED,
        include_intermediate_measurements=self._include_intermediate_measurements,
    )
    # Add new completed trials to not report on them again in the future.
    self._incorporated_completed_trial_ids |= set(t.id for t in new_trials)

    logging.info(
        'Loaded %s completed trials. Max trial id is %s.',
        len(new_trials),
        max_trial_id,
    )
    return new_trials

  def dump(self) -> vz.Metadata:
    """Dump state.

    Returns:
      Metadata has the following namespace hierarchy:
        Namespace([self._ns_root]): contains the policy's state.
        Namespace([self._ns_root, self._ns_designer]: contains the designer's
          state.
    """
    md = vz.Metadata()
    # TODO: Storing every id is inefficient. Optimize this.
    # We don't store/load ACTIVE trials as we always pass all of them.
    md[_INCOPORATED_COMPLETED_TRIALS_IDS] = json.dumps(
        list(self._incorporated_completed_trial_ids)
    )
    return md

  def load(self, md: vz.Metadata) -> None:
    if _INCOPORATED_COMPLETED_TRIALS_IDS in md:
      try:
        # We don't store/load ACTIVE trials as we always pass all of them.
        self._incorporated_completed_trial_ids = set(
            json.loads(md[_INCOPORATED_COMPLETED_TRIALS_IDS])
        )
      except json.JSONDecodeError as e:
        raise serializable.HarmlessDecodeError from e
    else:
      raise serializable.HarmlessDecodeError('Missing expected metadata keys.')

    logging.info(
        'Successfully recovered the cache, which has %s completed trials.',
        len(self._incorporated_completed_trial_ids),
    )
