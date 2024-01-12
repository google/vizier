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

import itertools
from matplotlib.pylab import plt
import numpy as np
from vizier import benchmarks as vzb
from vizier._src.algorithms.designers import grid
from vizier._src.algorithms.designers import random
from vizier._src.benchmarks.analyzers import plot_utils
from vizier._src.benchmarks.analyzers import state_analyzer
from vizier.benchmarks import experimenters
from absl.testing import absltest


class PlotUtilsTest(absltest.TestCase):

  def test_plot_median_convergence(self):
    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    plot_utils.plot_median_convergence(
        ax,
        curves=np.asarray([[1, 1, 2, 3, 4], [1, 1, 1, 2, float('nan')]]),
        percentiles=((40, 60), (30, 70)),
        alphas=(0.2, 0.4),
        xs=np.arange(1, 6),
        color='r',
    )

  def test_plot_mean_convergence(self):
    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    plot_utils.plot_mean_convergence(
        ax,
        curves=np.asarray([[1, 1, 2, 3, 4], [1, 1, 1, 2, float('nan')]]),
        xs=np.arange(1, 6),
        color='r',
    )

  def test_plot_median_convergence_omit_optional_args(self):
    _, ax = plt.subplots(1, 1, figsize=(12, 8))
    plot_utils.plot_median_convergence(
        ax, curves=np.asarray([[1, 1, 2, 3, 4], [1, 1, 1, 2, float('nan')]])
    )

  def test_plot_records(self):
    function_names = ['Sphere', 'Discus']
    dimensions = [4, 8]
    product_list = list(itertools.product(function_names, dimensions))

    experimenter_factories = []
    for product in product_list:
      experimenter_factory = experimenters.BBOBExperimenterFactory(
          name=product[0], dim=product[1]
      )
      experimenter_factories.append(experimenter_factory)

    num_repeats = 5
    num_iterations = 20
    runner = vzb.BenchmarkRunner(
        benchmark_subroutines=[
            vzb.GenerateSuggestions(),
            vzb.EvaluateActiveTrials(),
        ],
        num_repeats=num_iterations,
    )
    algorithms = {
        'grid': grid.GridSearchDesigner.from_problem,
        'random': random.RandomDesigner.from_problem,
    }

    records = []
    for experimenter_factory in experimenter_factories:
      for algo_name, algo_factory in algorithms.items():
        benchmark_state_factory = vzb.ExperimenterDesignerBenchmarkStateFactory(
            experimenter_factory=experimenter_factory,
            designer_factory=algo_factory,
        )
        states = []
        for _ in range(num_repeats):
          benchmark_state = benchmark_state_factory()
          runner.run(benchmark_state)
          states.append(benchmark_state)
        record = state_analyzer.BenchmarkStateAnalyzer.to_record(
            algorithm=algo_name,
            experimenter_factory=experimenter_factory,
            states=states,
        )
        records.append(record)
    plot_utils.plot_from_records(records)


if __name__ == '__main__':
  absltest.main()
