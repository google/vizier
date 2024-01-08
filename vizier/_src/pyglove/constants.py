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

"""Constants."""

PARAMETER_NAME_ROOT = '$'
STUDY_METADATA_KEY_TUNER_ID = 'PRIMARY_TUNER_ID'
STUDY_METADATA_KEY_DNA_SPEC = 'TUNER_DNA_SPEC'
STUDY_METADATA_KEY_USES_EXTERNAL_DNA_SPEC = 'USES_EXTERNAL_DNA_SPEC'
TRIAL_METADATA_KEY_DNA_METADATA = 'DNA_METADATA'
TRIAL_METADATA_KEY_CUSTOM_TYPE_DECISIONS = 'CUSTOM_TYPE_DECISIONS'
TRIAL_METADATA_KEYS = frozenset(
    [TRIAL_METADATA_KEY_DNA_METADATA, TRIAL_METADATA_KEY_CUSTOM_TYPE_DECISIONS])
FROM_VIZIER_STUDY_HINT = 'from_vizier_study'
METADATA_NAMESPACE = 'pyglove'
RELATED_LINKS_SUBNAMESPACE = 'related_links'
DUMMY_PARAMETER_NAME = 'dummy_parameter'
DUMMY_PARAMETER_VALUE = 'UNUSED'

REWARD_METRIC_NAME = 'reward'

STUDY_METADATA_KEYS = frozenset([
    STUDY_METADATA_KEY_TUNER_ID,
    STUDY_METADATA_KEY_DNA_SPEC,
    STUDY_METADATA_KEY_USES_EXTERNAL_DNA_SPEC,
])
