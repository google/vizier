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

"""PyVizier classes relevant for the service, including OSS-specific objects."""

# TODO: Re-evalaute what to expose to users, once closed.
# TODO: Alternatively, remove this import.
from vizier._src.pyvizier.oss import metadata_util
from vizier._src.pyvizier.oss.automated_stopping import AutomatedStoppingConfig
from vizier._src.pyvizier.oss.automated_stopping import AutomatedStoppingConfigProto
from vizier._src.pyvizier.oss.proto_converters import EarlyStopConverter
from vizier._src.pyvizier.oss.proto_converters import MeasurementConverter
from vizier._src.pyvizier.oss.proto_converters import MetadataDeltaConverter
from vizier._src.pyvizier.oss.proto_converters import MonotypeParameterSequence
from vizier._src.pyvizier.oss.proto_converters import ParameterConfigConverter
from vizier._src.pyvizier.oss.proto_converters import ParameterType
from vizier._src.pyvizier.oss.proto_converters import ParameterValueConverter
from vizier._src.pyvizier.oss.proto_converters import ProblemStatementConverter
from vizier._src.pyvizier.oss.proto_converters import ScaleType
from vizier._src.pyvizier.oss.proto_converters import StudyStateConverter
from vizier._src.pyvizier.oss.proto_converters import SuggestConverter
from vizier._src.pyvizier.oss.proto_converters import TrialConverter
from vizier._src.pyvizier.oss.proto_converters import TrialSuggestionConverter
from vizier._src.pyvizier.oss.study_config import Algorithm
from vizier._src.pyvizier.oss.study_config import ObservationNoise
from vizier._src.pyvizier.oss.study_config import ParameterValueSequence
from vizier._src.pyvizier.oss.study_config import StudyConfig
from vizier._src.pyvizier.pythia.study import StudyDescriptor
from vizier._src.pyvizier.pythia.study import StudyState
from vizier._src.pyvizier.pythia.study import StudyStateInfo
from vizier._src.pyvizier.shared.base_study_config import MetricInformation
from vizier._src.pyvizier.shared.base_study_config import MetricsConfig
from vizier._src.pyvizier.shared.base_study_config import ObjectiveMetricGoal
from vizier._src.pyvizier.shared.base_study_config import ProblemStatement
from vizier._src.pyvizier.shared.common import Metadata
from vizier._src.pyvizier.shared.common import MetadataValue
from vizier._src.pyvizier.shared.common import Namespace
from vizier._src.pyvizier.shared.parameter_config import ExternalType
from vizier._src.pyvizier.shared.parameter_config import ParameterConfig
from vizier._src.pyvizier.shared.parameter_config import SearchSpace
from vizier._src.pyvizier.shared.parameter_config import SearchSpaceSelector
from vizier._src.pyvizier.shared.study import ProblemAndTrials
from vizier._src.pyvizier.shared.trial import CompletedTrial
from vizier._src.pyvizier.shared.trial import CompletedTrialWithMeasurements
from vizier._src.pyvizier.shared.trial import Measurement
from vizier._src.pyvizier.shared.trial import MetadataDelta
from vizier._src.pyvizier.shared.trial import Metric
from vizier._src.pyvizier.shared.trial import NaNMetric
from vizier._src.pyvizier.shared.trial import ParameterDict
from vizier._src.pyvizier.shared.trial import ParameterValue
from vizier._src.pyvizier.shared.trial import PendingTrial
from vizier._src.pyvizier.shared.trial import PendingTrialWithMeasurements
from vizier._src.pyvizier.shared.trial import Trial
from vizier._src.pyvizier.shared.trial import TrialFilter
from vizier._src.pyvizier.shared.trial import TrialStatus
from vizier._src.pyvizier.shared.trial import TrialSuggestion
