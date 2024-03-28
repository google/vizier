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

"""These are the externally-useful symbols for OSS Pythia."""
# pylint: disable=unused-import

# The API for a Pythia Policy -- i.e. the algorithm that Pythia serves.
from vizier._src.pythia.local_policy_supporters import InRamPolicySupporter
from vizier._src.pythia.policy import EarlyStopDecision
from vizier._src.pythia.policy import EarlyStopDecisions
from vizier._src.pythia.policy import EarlyStopRequest
from vizier._src.pythia.policy import Policy
from vizier._src.pythia.policy import SuggestDecision
from vizier._src.pythia.policy import SuggestRequest
from vizier._src.pythia.policy_factory import PolicyFactory
from vizier._src.pythia.policy_supporter import PolicySupporter
from vizier._src.pythia.pythia_errors import CancelComputeError
from vizier._src.pythia.pythia_errors import InactivateStudyError
from vizier._src.pythia.pythia_errors import LoadTooLargeError
from vizier._src.pythia.pythia_errors import PythiaError
from vizier._src.pythia.pythia_errors import PythiaFallbackError
from vizier._src.pythia.pythia_errors import PythiaProtocolError
from vizier._src.pythia.pythia_errors import TemporaryPythiaError
from vizier._src.pythia.pythia_errors import VizierDatabaseError
from vizier._src.pythia.suggest_default import seed_with_default

# Documentation
assert issubclass(CancelComputeError, Exception)
assert issubclass(PythiaProtocolError, Exception)
assert issubclass(VizierDatabaseError, Exception)
assert issubclass(InactivateStudyError, PythiaError)
assert issubclass(LoadTooLargeError, PythiaError)
assert issubclass(TemporaryPythiaError, PythiaError)
