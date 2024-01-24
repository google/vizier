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

"""List of Designers and related algorithms."""

from vizier._src.algorithms.designers.bocs import BOCSDesigner
from vizier._src.algorithms.designers.cmaes import CMAESDesigner
from vizier._src.algorithms.designers.eagle_strategy.eagle_strategy import EagleStrategyDesigner
from vizier._src.algorithms.designers.emukit import EmukitDesigner
from vizier._src.algorithms.designers.gp_bandit import VizierGPBandit
from vizier._src.algorithms.designers.gp_ucb_pe import VizierGPUCBPEBandit
from vizier._src.algorithms.designers.grid import GridSearchDesigner
from vizier._src.algorithms.designers.harmonica import HarmonicaDesigner
from vizier._src.algorithms.designers.quasi_random import QuasiRandomDesigner
from vizier._src.algorithms.designers.random import RandomDesigner
from vizier._src.algorithms.designers.unsafe_as_infeasible_designer import UnsafeAsInfeasibleDesigner
from vizier._src.algorithms.ensemble.ensemble_design import AdaptiveEnsembleDesign
from vizier._src.algorithms.ensemble.ensemble_design import EnsembleDesign
from vizier._src.algorithms.ensemble.ensemble_design import EXP3IXEnsembleDesign
from vizier._src.algorithms.ensemble.ensemble_design import EXP3UniformEnsembleDesign
from vizier._src.algorithms.ensemble.ensemble_design import RandomEnsembleDesign
from vizier._src.algorithms.evolution.nsga2 import NSGA2Designer
