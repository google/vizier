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

"""Curve Analysis utilities for benchmarking and metalearning."""
# pylint: disable=unused-import

from vizier._src.benchmarks.analyzers.convergence_curve import ConvergenceComparator
from vizier._src.benchmarks.analyzers.convergence_curve import ConvergenceComparatorFactory
from vizier._src.benchmarks.analyzers.convergence_curve import ConvergenceCurve
from vizier._src.benchmarks.analyzers.convergence_curve import ConvergenceCurveConverter
from vizier._src.benchmarks.analyzers.convergence_curve import HypervolumeCurveConverter
from vizier._src.benchmarks.analyzers.convergence_curve import LogEfficiencyConvergenceCurveComparator
from vizier._src.benchmarks.analyzers.convergence_curve import LogEfficiencyConvergenceCurveComparatorFactory
from vizier._src.benchmarks.analyzers.convergence_curve import MultiMetricCurveConverter
from vizier._src.benchmarks.analyzers.convergence_curve import PercentageBetterConvergenceCurveComparator
from vizier._src.benchmarks.analyzers.convergence_curve import PercentageBetterConvergenceCurveComparatorFactory
from vizier._src.benchmarks.analyzers.convergence_curve import RestartingCurveConverter
from vizier._src.benchmarks.analyzers.convergence_curve import StatefulCurveConverter
from vizier._src.benchmarks.analyzers.convergence_curve import WinRateConvergenceCurveComparator
from vizier._src.benchmarks.analyzers.convergence_curve import WinRateConvergenceCurveComparatorFactory
from vizier._src.benchmarks.analyzers.plot_utils import plot_from_records
from vizier._src.benchmarks.analyzers.plot_utils import plot_mean_convergence
from vizier._src.benchmarks.analyzers.plot_utils import plot_median_convergence
from vizier._src.benchmarks.analyzers.state_analyzer import BenchmarkRecord
from vizier._src.benchmarks.analyzers.state_analyzer import BenchmarkRecordAnalyzer
from vizier._src.benchmarks.analyzers.state_analyzer import BenchmarkStateAnalyzer
from vizier._src.benchmarks.analyzers.state_analyzer import PlotElement
