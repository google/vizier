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

from absl.testing import parameterized
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import grid
from vizier._src.benchmarks.experimenters.synthetic import simplekd
from vizier._src.benchmarks.runners import benchmark_runner
from vizier._src.benchmarks.runners import benchmark_state
from absl.testing import absltest


class SimpleKDExperimenterTest(parameterized.TestCase):

  @parameterized.product(
      best_category=['corner', 'center', 'mixed'],
      output_relative_error=[True, False],
  )
  def test_sweep(
      self,
      best_category: simplekd.SimpleKDCategory,
      output_relative_error: bool,
  ) -> None:
    experimenter = simplekd.SimpleKDExperimenter(
        best_category, output_relative_error=output_relative_error
    )
    runner = benchmark_runner.BenchmarkRunner(
        benchmark_subroutines=[
            benchmark_runner.GenerateSuggestions(),
            benchmark_runner.EvaluateActiveTrials(),
        ],
        num_repeats=300,
    )
    state = benchmark_state.BenchmarkState(
        experimenter,
        benchmark_state.PolicySuggester.from_designer_factory(
            experimenter.problem_statement(),
            grid.GridSearchDesigner.from_problem,
        ),
    )
    runner.run(state)

  @parameterized.parameters(
      dict(best_category='corner'),
      dict(best_category='center'),
      dict(best_category='mixed'),
  )
  def test_compute_optimal_objective(
      self, best_category: simplekd.SimpleKDCategory
  ) -> None:
    exptr_simple4d = simplekd.SimpleKDExperimenter(best_category)
    categorical_values = ['corner', 'center', 'mixed']
    discrete_values = (1, 2, 5, 6, 8)
    integer_values = [1, 2, 3]
    continuous_values = np.linspace(-1, 1, 1001)

    opt_value = np.nan
    for cat_value in categorical_values:
      for disc_value in discrete_values:
        for cont_value in continuous_values:
          for int_value in integer_values:
            values = {
                'categorical': [cat_value],
                'discrete': [disc_value],
                'int': [int_value],
                'float': [cont_value],
            }
            opt_value = np.nanmax([exptr_simple4d._compute(values), opt_value])

    self.assertAlmostEqual(opt_value, exptr_simple4d.optimal_objective)

  @parameterized.parameters(
      dict(best_category='corner'),
      dict(best_category='center'),
      dict(best_category='mixed'),
  )
  def test_optimal_relative_error(
      self, best_category: simplekd.SimpleKDCategory
  ) -> None:
    exptr_simple4d = simplekd.SimpleKDExperimenter(
        best_category, output_relative_error=True
    )
    categorical_values = ['corner', 'center', 'mixed']
    discrete_values = (1, 2, 5, 6, 8)
    integer_values = [1, 2, 3]
    continuous_values = np.linspace(-1, 1, 1001)

    opt_rel_error = np.nan
    for cat_value in categorical_values:
      for disc_value in discrete_values:
        for cont_value in continuous_values:
          for int_value in integer_values:
            trial = vz.Trial({
                'categorical': cat_value,
                'discrete_0': disc_value,
                'int_0': int_value,
                'float_0': cont_value,
            })
            exptr_simple4d.evaluate([trial])
            if trial.final_measurement is None:
              raise ValueError('Final measurement is None.')
            rel_err = trial.final_measurement.metrics['value'].value
            opt_rel_error = np.nanmin([rel_err, opt_rel_error])

    self.assertAlmostEqual(opt_rel_error, 0.0)


if __name__ == '__main__':
  absltest.main()
