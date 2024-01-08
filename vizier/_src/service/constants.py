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

"""Hard coded constants used in /service/ folder."""
import os

# The metadata namespace under which the Pythia endpoint is stored.
PYTHIA_ENDPOINT_NAMESPACE = 'service'

# The Study.metadata key where a Pythia endpoint can be stored.
PYTHIA_ENDPOINT_KEY = 'PYTHIA_ENDPOINT'

# A hard coded value which indicates that the Pythia endpoint is not set.
NO_ENDPOINT = 'NO_ENDPOINT'

# Default client ID.
UNUSED_CLIENT_ID = 'unused_client_id'

# Max int32 value.
MAX_STUDY_ID = 2147483647

# Will use RAM for SQL memory.
SQL_MEMORY_URL = 'sqlite:///:memory:'

# Will use local file path for HDD storage.
SERVICE_DIR = os.path.dirname(os.path.realpath(__file__))
VIZIER_DB_PATH = os.path.join(SERVICE_DIR, 'vizier.db')
SQL_LOCAL_URL = f'sqlite:///{VIZIER_DB_PATH}'
