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

"""Lightweight experimenters which do not require installing `requirements-benchmarks.txt`."""

from vizier._src.benchmarks.experimenters import experimenter_factory
from vizier._src.benchmarks.experimenters.combo_experimenter import CentroidExperimenter
from vizier._src.benchmarks.experimenters.combo_experimenter import ContaminationExperimenter
from vizier._src.benchmarks.experimenters.combo_experimenter import IsingExperimenter
from vizier._src.benchmarks.experimenters.combo_experimenter import MAXSATExperimenter
from vizier._src.benchmarks.experimenters.combo_experimenter import PestControlExperimenter
from vizier._src.benchmarks.experimenters.discretizing_experimenter import DiscretizingExperimenter
from vizier._src.benchmarks.experimenters.experimenter import Experimenter
from vizier._src.benchmarks.experimenters.experimenter_factory import BBOBExperimenterFactory
from vizier._src.benchmarks.experimenters.experimenter_factory import CombinedExperimenterFactory
from vizier._src.benchmarks.experimenters.experimenter_factory import ExperimenterFactory
from vizier._src.benchmarks.experimenters.experimenter_factory import SerializableExperimenterFactory
from vizier._src.benchmarks.experimenters.experimenter_factory import SingleObjectiveExperimenterFactory
from vizier._src.benchmarks.experimenters.infeasible_experimenter import HashingInfeasibleExperimenter
from vizier._src.benchmarks.experimenters.infeasible_experimenter import ParamRegionInfeasibleExperimenter
from vizier._src.benchmarks.experimenters.l1_categorical_experimenter import L1CategorialExperimenter
from vizier._src.benchmarks.experimenters.multiobjective_experimenter import MultiObjectiveExperimenter
from vizier._src.benchmarks.experimenters.noisy_experimenter import NoisyExperimenter
from vizier._src.benchmarks.experimenters.normalizing_experimenter import NormalizingExperimenter
from vizier._src.benchmarks.experimenters.numpy_experimenter import MultiObjectiveNumpyExperimenter
from vizier._src.benchmarks.experimenters.numpy_experimenter import NumpyExperimenter
from vizier._src.benchmarks.experimenters.permuting_experimenter import PermutingExperimenter
from vizier._src.benchmarks.experimenters.shifting_experimenter import ShiftingExperimenter
from vizier._src.benchmarks.experimenters.sign_flip_experimenter import SignFlipExperimenter
from vizier._src.benchmarks.experimenters.sparse_experimenter import SparseExperimenter
from vizier._src.benchmarks.experimenters.surrogate_experimenter import PredictorExperimenter
from vizier._src.benchmarks.experimenters.switch_experimenter import SwitchExperimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob
from vizier._src.benchmarks.experimenters.synthetic.branin import Branin2DExperimenter
from vizier._src.benchmarks.experimenters.synthetic.deb import DHExperimenter
from vizier._src.benchmarks.experimenters.synthetic.hartmann import HartmannExperimenter
from vizier._src.benchmarks.experimenters.synthetic.simplekd import SimpleKDExperimenter
