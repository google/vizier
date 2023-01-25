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

"""Hard coded constants used in /service/ folder."""

# The metadata namespace under which the Pythia endpoint is stored.
PYTHIA_ENDPOINT_NAMESPACE = 'service'

# The Study.metadata key where a Pythia endpoint can be stored.
PYTHIA_ENDPOINT_KEY = 'PYTHIA_ENDPOINT'

# A hard coded value which indicates that the Pythia endpoint is not set.
NO_ENDPOINT = 'NO_ENDPOINT'

# Default client ID.
UNUSED_CLIENT_ID = 'Unused client id.'
