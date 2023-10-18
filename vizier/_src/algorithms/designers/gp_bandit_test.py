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

"""Tests for gp_bandit."""

from typing import Callable, Union
from unittest import mock

import jax
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import gp_bandit
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.optimizers import eagle_strategy as es
from vizier._src.algorithms.optimizers import lbfgsb_optimizer as lo
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.algorithms.testing import test_runners
from vizier.jax import optimizers
from vizier.pyvizier import converters
from vizier.pyvizier.converters import padding
from vizier.testing import test_studies
from vizier.utils import profiler

from absl.testing import absltest
from absl.testing import parameterized


ard_optimizer = optimizers.default_optimizer()
vectorized_optimizer_factory = vb.VectorizedOptimizerFactory(
    strategy_factory=es.VectorizedEagleStrategyFactory(),
    max_evaluations=10,
)
lbfgsb_optimizer_factory = lo.LBFGSBOptimizerFactory()


def _build_mock_continuous_array_specs(n):
  continuous_spec = mock.create_autospec(converters.NumpyArraySpec)
  continuous_spec.type = converters.NumpyArraySpecType.CONTINUOUS
  continuous_spec.num_dimensions = 1
  return [continuous_spec] * n


def _setup_lambda_search(
    f: Callable[[float], float], num_trials: int = 100
) -> tuple[gp_bandit.VizierGPBandit, list[vz.Trial], vz.ProblemStatement]:
  """Sets up a GP designer and outputs completed studies for `f`.

  Args:
    f: 1D objective to be optimized, i.e. f(x), where x is a scalar in [-5., 5.)
    num_trials: Number of mock "evaluated" trials to return.

  Returns:
  A GP designer set up for the problem of optimizing the objective, without any
  data updated.
  Evaluated trials against `f`.
  """
  assert (
      num_trials > 0
  ), f'Must provide a positive number of trials. Got {num_trials}.'

  search_space = vz.SearchSpace()
  search_space.root.add_float_param('x0', -5.0, 5.0)
  problem = vz.ProblemStatement(
      search_space=search_space,
      metric_information=vz.MetricsConfig(
          metrics=[
              vz.MetricInformation('obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE),
          ]
      ),
  )

  suggestions = quasi_random.QuasiRandomDesigner(
      problem.search_space, seed=1
  ).suggest(num_trials)

  obs_trials = []
  for idx, suggestion in enumerate(suggestions):
    trial = suggestion.to_trial(idx)
    x = suggestions[idx].parameters['x0'].value
    trial.complete(vz.Measurement(metrics={'obj': f(x)}))
    obs_trials.append(trial)

  gp_designer = gp_bandit.VizierGPBandit(problem, ard_optimizer=ard_optimizer)
  return gp_designer, obs_trials, problem


def _compute_mse(
    designer: gp_bandit.VizierGPBandit,
    test_trials: list[vz.Trial],
    y_test: list[float],
) -> float:
  """Evaluate the designer's accuracy on the test set.

  Args:
    designer: The GP bandit designer to predict from.
    test_trials: The trials of the test set
    y_test: The results of the test set

  Returns:
    The MSE of `designer` on `test_trials` and `y_test`
  """
  preds = designer.predict(test_trials)
  return np.sum(np.square(preds.mean - y_test))


class GoogleGpBanditTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          iters=3,
          batch_size=5,
          num_seed_trials=5,
          padding_schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.MULTIPLES_OF_10,
          ),
      ),
      dict(
          iters=3,
          batch_size=5,
          num_seed_trials=5,
      ),
      dict(
          iters=3,
          batch_size=5,
          num_seed_trials=5,
      ),
  )
  def test_on_flat_continuous_space(
      self,
      *,
      iters: int = 5,
      batch_size: int = 1,
      num_seed_trials: int = 1,
      ensemble_size: int = 1,
      padding_schedule: padding.PaddingSchedule = padding.PaddingSchedule(),
      use_trust_region: bool = False,
      acquisition_optimizer_factory: Union[
          vb.VectorizedOptimizerFactory, lo.LBFGSBOptimizerFactory
      ] = vectorized_optimizer_factory,
  ):
    # We use string names so that test case names are readable. Convert them
    # to objects.
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.append(
        vz.MetricInformation(
            name='metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )

    designer = gp_bandit.VizierGPBandit(
        problem=problem,
        acquisition_optimizer_factory=acquisition_optimizer_factory,
        ard_optimizer=optimizers.JaxoptLbfgsB(
            optimizers.LbfgsBOptions(maxiter=5, num_line_search_steps=5)
        ),
        num_seed_trials=num_seed_trials,
        ensemble_size=ensemble_size,
        padding_schedule=padding_schedule,
        use_trust_region=use_trust_region,
        rng=jax.random.PRNGKey(0),
        linear_coef=0.1,
    )
    with profiler.collect_events() as events:
      self.assertLen(
          test_runners.RandomMetricsRunner(
              problem,
              iters=iters,
              batch_size=batch_size,
              verbose=1,
              validate_parameters=True,
              seed=1,
          ).run_designer(designer),
          iters * batch_size,
      )

    self.assertIn('VizierGPBandit.suggest', profiler.get_latencies_dict(events))

    quasi_random_sampler = quasi_random.QuasiRandomDesigner(
        problem.search_space,
    )
    predict_trials = quasi_random_sampler.suggest(count=3)
    # test the sample method.
    samples = designer.sample(predict_trials, num_samples=5)
    self.assertSequenceEqual(samples.shape, (5, 3))
    self.assertFalse(np.isnan(samples).any())
    empty_samples = designer.sample([], num_samples=5)
    self.assertSequenceEqual(empty_samples.shape, (5, 0))
    # test the predict method.
    prediction = designer.predict(predict_trials)
    self.assertLen(prediction.mean, 3)
    self.assertLen(prediction.stddev, 3)
    self.assertFalse(np.isnan(prediction.mean).any())
    self.assertFalse(np.isnan(prediction.stddev).any())


if __name__ == '__main__':
  # Jax disables float64 computations by default and will silently convert
  # float64s to float32s. We must explicitly enable float64.
  jax.config.update('jax_enable_x64', True)
  absltest.main()
