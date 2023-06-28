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

from typing import Optional
from unittest import mock

import jax
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import gp_bandit
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.optimizers import eagle_strategy as es
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


def _build_mock_continuous_array_specs(n):
  continuous_spec = mock.create_autospec(converters.NumpyArraySpec)
  continuous_spec.type = converters.NumpyArraySpecType.CONTINUOUS
  continuous_spec.num_dimensions = 1
  return [continuous_spec] * n


class GoogleGpBanditTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(iters=3, batch_size=2, num_seed_trials=1, ensemble_size=2),
      dict(iters=3, batch_size=1, num_seed_trials=1, ensemble_size=2),
      dict(
          iters=3,
          batch_size=5,
          num_seed_trials=5,
          use_categorical_kernel=True,
      ),
      dict(
          iters=3,
          batch_size=5,
          num_seed_trials=5,
          padding_schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.MULTIPLES_OF_10,
              num_features=padding.PaddingType.POWERS_OF_2,
          ),
      ),
      dict(
          iters=5,
          batch_size=1,
          num_seed_trials=3,
          padding_schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.POWERS_OF_2,
              num_features=padding.PaddingType.POWERS_OF_2,
          ),
      ),
      dict(
          padding_schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.POWERS_OF_2,
              num_features=padding.PaddingType.POWERS_OF_2,
          ),
          ensemble_size=3,
      ),
  )
  def test_on_flat_continuous_space(
      self,
      *,
      iters: int = 5,
      batch_size: int = 1,
      num_seed_trials: int = 1,
      ensemble_size: int = 1,
      padding_schedule: Optional[padding.PaddingSchedule] = None,
      use_categorical_kernel: bool = False,
      use_trust_region: bool = False,
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
    vectorized_optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=es.VectorizedEagleStrategyFactory(),
        max_evaluations=10,
    )

    designer = gp_bandit.VizierGPBandit(
        problem=problem,
        acquisition_optimizer_factory=vectorized_optimizer_factory,
        ard_optimizer=optimizers.JaxoptLbfgsB(
            optimizers.LbfgsBOptions(maxiter=5, num_line_search_steps=5)
        ),
        num_seed_trials=num_seed_trials,
        ensemble_size=ensemble_size,
        padding_schedule=padding_schedule,
        use_categorical_kernel=use_categorical_kernel,
        use_trust_region=use_trust_region,
        rng=jax.random.PRNGKey(0),
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
    predict_trials = quasi_random_sampler.suggest(count=7)
    prediction = designer.predict(predict_trials)
    self.assertLen(prediction.mean, 7)
    self.assertLen(prediction.stddev, 7)
    self.assertFalse(np.isnan(prediction.mean).any())
    self.assertFalse(np.isnan(prediction.stddev).any())

  @parameterized.parameters(
      dict(iters=3, batch_size=5, num_seed_trials=5),
      dict(iters=5, batch_size=1, num_seed_trials=2),
      dict(iters=5, batch_size=1, num_seed_trials=1),
      dict(
          iters=3,
          batch_size=5,
          num_seed_trials=5,
          padding_schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.MULTIPLES_OF_10,
              num_features=padding.PaddingType.POWERS_OF_2,
          ),
      ),
      dict(
          iters=3,
          batch_size=5,
          num_seed_trials=5,
          use_categorical_kernel=True,
      ),
  )
  def test_on_flat_mixed_space(
      self,
      iters: int,
      batch_size: int,
      num_seed_trials: int,
      padding_schedule: Optional[padding.PaddingSchedule] = None,
      use_categorical_kernel: bool = False,
      use_trust_region: bool = True,
  ):
    problem = vz.ProblemStatement(test_studies.flat_space_with_all_types())
    problem.metric_information.append(
        vz.MetricInformation(
            name='metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    vectorized_optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=es.VectorizedEagleStrategyFactory(), max_evaluations=10
    )
    designer = gp_bandit.VizierGPBandit(
        problem=problem,
        acquisition_optimizer_factory=vectorized_optimizer_factory,
        num_seed_trials=num_seed_trials,
        padding_schedule=padding_schedule,
        use_categorical_kernel=use_categorical_kernel,
        use_trust_region=use_trust_region,
    )
    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=iters,
            batch_size=batch_size,
            verbose=1,
            validate_parameters=True,
        ).run_designer(designer),
        iters * batch_size,
    )
    quasi_random_sampler = quasi_random.QuasiRandomDesigner(
        problem.search_space
    )
    predict_trials = quasi_random_sampler.suggest(count=7)
    prediction = designer.predict(predict_trials)
    self.assertLen(prediction.mean, 7)
    self.assertLen(prediction.stddev, 7)
    self.assertFalse(np.isnan(prediction.mean).any())
    self.assertFalse(np.isnan(prediction.stddev).any())

  def test_prediction_accuracy(self):
    search_space = vz.SearchSpace()
    search_space.root.add_float_param('x0', -5.0, 5.0)
    problem = vz.ProblemStatement(
        search_space=search_space,
        metric_information=vz.MetricsConfig(
            metrics=[
                vz.MetricInformation(
                    'obj', goal=vz.ObjectiveMetricGoal.MAXIMIZE
                ),
            ]
        ),
    )
    f = lambda x: -((x - 0.5) ** 2)

    suggestions = quasi_random.QuasiRandomDesigner(
        problem.search_space, seed=1
    ).suggest(100)

    obs_trials = []
    for idx, suggestion in enumerate(suggestions):
      trial = suggestion.to_trial(idx)
      x = suggestions[idx].parameters['x0'].value
      trial.complete(vz.Measurement(metrics={'obj': f(x)}))
      obs_trials.append(trial)

    gp_designer = gp_bandit.VizierGPBandit(problem, ard_optimizer=ard_optimizer)
    gp_designer.update(vza.CompletedTrials(obs_trials), vza.ActiveTrials())
    pred_trial = vz.Trial({'x0': 0.0})
    pred = gp_designer.predict([pred_trial])
    self.assertLess(np.abs(pred.mean[0] - f(0.0)), 2e-2)

  # TODO: Add assertions to this test. Ideally
  # create two designers with the same trial count (without padding)
  # padding is just an internal detail that should be tested separately.
  def test_jit_once(self, *args):
    del args
    jax.clear_caches()

    space = test_studies.flat_continuous_space_with_scaling()
    problem = vz.ProblemStatement(space)
    problem.metric_information.append(
        vz.MetricInformation(
            name='metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    def create_designer(problem):
      return gp_bandit.VizierGPBandit(
          problem=problem,
          acquisition_optimizer_factory=vb.VectorizedOptimizerFactory(
              strategy_factory=es.VectorizedEagleStrategyFactory(),
              max_evaluations=10,
          ),
          num_seed_trials=3,
          ensemble_size=2,
          padding_schedule=padding.PaddingSchedule(
              num_trials=padding.PaddingType.MULTIPLES_OF_10,
              num_features=padding.PaddingType.MULTIPLES_OF_10,
          ),
      )

    def create_runner(problem):
      return test_runners.RandomMetricsRunner(
          problem,
          iters=5,
          batch_size=1,
          verbose=1,
          validate_parameters=True,
      )

    designer = create_designer(problem)
    # Padding schedule should avoid retracing every iteration.
    create_runner(problem).run_designer(designer)

    # Padding schedule should avoid retracing with one more feature.
    space.root.add_float_param('x0', -5.0, 5.0)
    designer1 = create_designer(problem)
    create_runner(problem).run_designer(designer1)

    # Retracing should not occur when a new VizierGPBandit instance is created.
    designer2 = create_designer(problem)
    create_runner(problem).run_designer(designer2)


if __name__ == '__main__':
  # Jax disables float64 computations by default and will silently convert
  # float64s to float32s. We must explicitly enable float64.
  jax.config.update('jax_enable_x64', True)
  jax.config.update('jax_log_compiles', True)
  absltest.main()
