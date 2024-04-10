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

"""PyVizier classes for general use.

Contains API shared across all platforms (Internal, Cloud, OSS).
"""

from vizier._src.pyvizier.oss.study_config import ParameterValueSequence
from vizier._src.pyvizier.pythia.study import StudyDescriptor
from vizier._src.pyvizier.pythia.study import StudyState
from vizier._src.pyvizier.pythia.study import StudyStateInfo
from vizier._src.pyvizier.shared.base_study_config import MetricInformation
from vizier._src.pyvizier.shared.base_study_config import MetricsConfig
from vizier._src.pyvizier.shared.base_study_config import MetricType
from vizier._src.pyvizier.shared.base_study_config import ObjectiveMetricGoal
from vizier._src.pyvizier.shared.base_study_config import ProblemStatement
from vizier._src.pyvizier.shared.common import Metadata
from vizier._src.pyvizier.shared.common import MetadataValue
from vizier._src.pyvizier.shared.common import Namespace
from vizier._src.pyvizier.shared.parameter_config import ExternalType
from vizier._src.pyvizier.shared.parameter_config import FidelityConfig
from vizier._src.pyvizier.shared.parameter_config import MonotypeParameterSequence
from vizier._src.pyvizier.shared.parameter_config import ParameterConfig
from vizier._src.pyvizier.shared.parameter_config import ParameterType
from vizier._src.pyvizier.shared.parameter_config import ScaleType
from vizier._src.pyvizier.shared.parameter_config import SearchSpace
from vizier._src.pyvizier.shared.parameter_config import SearchSpaceSelector
from vizier._src.pyvizier.shared.parameter_iterators import SequentialParameterBuilder
from vizier._src.pyvizier.shared.study import ProblemAndTrials
from vizier._src.pyvizier.shared.trial import CompletedTrial
from vizier._src.pyvizier.shared.trial import CompletedTrialWithMeasurements
from vizier._src.pyvizier.shared.trial import Measurement
from vizier._src.pyvizier.shared.trial import MetadataDelta
from vizier._src.pyvizier.shared.trial import Metric
from vizier._src.pyvizier.shared.trial import NaNMetric
from vizier._src.pyvizier.shared.trial import ParameterDict
from vizier._src.pyvizier.shared.trial import ParameterValue
from vizier._src.pyvizier.shared.trial import ParameterValueTypes
from vizier._src.pyvizier.shared.trial import PendingTrial
from vizier._src.pyvizier.shared.trial import PendingTrialWithMeasurements
from vizier._src.pyvizier.shared.trial import Trial
from vizier._src.pyvizier.shared.trial import TrialFilter
from vizier._src.pyvizier.shared.trial import TrialStatus
from vizier._src.pyvizier.shared.trial import TrialSuggestion

FlatSpace = SearchSpace

# Documentation
assert isinstance(NaNMetric, Metric)
assert issubclass(CompletedTrial, Trial)
assert issubclass(CompletedTrialWithMeasurements, Trial)
assert issubclass(PendingTrial, Trial)
assert issubclass(PendingTrialWithMeasurements, Trial)
assert issubclass(Trial, TrialSuggestion)
