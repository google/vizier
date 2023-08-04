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

from typing import Callable
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


def _setup_lambda_search(
    f: Callable[[float], float], num_trials: int = 100
) -> tuple[gp_bandit.VizierGPBandit, list[vz.Trial]]:
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
  return gp_designer, obs_trials


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
      dict(iters=3, batch_size=2, num_seed_trials=1, ensemble_size=2),
      dict(iters=3, batch_size=1, num_seed_trials=1, ensemble_size=2),
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
      padding_schedule: padding.PaddingSchedule = padding.PaddingSchedule(),
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
  )
  def test_on_flat_mixed_space(
      self,
      iters: int,
      batch_size: int,
      num_seed_trials: int,
      padding_schedule: padding.PaddingSchedule = padding.PaddingSchedule(),
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
    predict_trials = quasi_random_sampler.suggest(count=3)
    # Test the sample method.
    samples = designer.sample(predict_trials, num_samples=5)
    self.assertSequenceEqual(samples.shape, (5, 3))
    samples = designer.sample(predict_trials, num_samples=5)
    self.assertSequenceEqual(samples.shape, (5, 3))
    self.assertFalse(np.isnan(samples).any())
    empty_samples = designer.sample([], num_samples=5)
    self.assertSequenceEqual(empty_samples.shape, (5, 0))
    # Test the predict method.
    prediction = designer.predict(predict_trials)
    self.assertLen(prediction.mean, 3)
    self.assertLen(prediction.stddev, 3)
    self.assertFalse(np.isnan(prediction.mean).any())
    self.assertFalse(np.isnan(prediction.stddev).any())

  def test_prediction_accuracy(self):
    f = lambda x: -((x - 0.5) ** 2)
    gp_designer, obs_trials = _setup_lambda_search(f)
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

  def test_priors_work(self):
    f = lambda x: -((x - 0.5) ** 2)

    # X is in range of what is defined in `_setup_lambda_search`, [-5.0, 5.0)
    x_test = np.random.default_rng(1).uniform(-5.0, 5.0, 100)
    y_test = [f(x) for x in x_test]
    test_trials = [vz.Trial({'x0': x}) for x in x_test]

    # Create the designer with a prior and the trials to train the prior.
    gp_designer_with_prior, obs_trials_for_prior = _setup_lambda_search(
        f=f, num_trials=100
    )

    # Update prior with above trials.
    gp_designer_with_prior.update_priors(
        [vza.CompletedTrials(obs_trials_for_prior)]
    )

    # Purposefully set a low number of trials for the actual study, so that
    # the designer without the prior will predict with poor accuracy.
    gp_designer_no_prior, obs_trials = _setup_lambda_search(f=f, num_trials=20)

    # Update both priors with the actual study.
    gp_designer_no_prior.update(
        vza.CompletedTrials(obs_trials), vza.ActiveTrials()
    )
    gp_designer_with_prior.update(
        vza.CompletedTrials(obs_trials), vza.ActiveTrials()
    )

    # Evaluate the no prior designer's accuracy on the test set.
    mse_no_prior = _compute_mse(gp_designer_no_prior, test_trials, y_test)

    # Evaluate the designer with prior's accuracy on the test set.
    mse_with_prior = _compute_mse(gp_designer_with_prior, test_trials, y_test)

    # The designer with a prior should predict better.
    self.assertLess(mse_with_prior, mse_no_prior)

  def test_multiple_priors(self):
    """Tests that a multi-prior GP predicts better than a GP with one prior."""
    f = lambda x: -((x - 0.5) ** 2)
    multi_prior_gp_designer, multi_prior_trials = _setup_lambda_search(
        f, num_trials=300
    )
    prior_0, prior_1, top = np.array_split(multi_prior_trials, 3)
    multi_prior_gp_designer.update_priors([vza.CompletedTrials(prior_0)])
    multi_prior_gp_designer.update_priors([vza.CompletedTrials(prior_1)])
    multi_prior_gp_designer.update(vza.CompletedTrials(top), vza.ActiveTrials())
    self.assertLen(multi_prior_gp_designer._prior_studies, 2)
    self.assertLen(multi_prior_gp_designer._trials, len(top))

    single_prior_gp_designer, single_prior_trials = _setup_lambda_search(
        f, num_trials=200
    )
    prior, top = np.array_split(single_prior_trials, 2)
    single_prior_gp_designer.update_priors([vza.CompletedTrials(prior)])
    single_prior_gp_designer.update(
        vza.CompletedTrials(top), vza.ActiveTrials()
    )
    self.assertLen(single_prior_gp_designer._prior_studies, 1)
    self.assertLen(single_prior_gp_designer._trials, len(top))

    x_test = np.random.default_rng(1).uniform(-5.0, 5.0, 100)
    y_test = [f(x) for x in x_test]
    test_trials = [vz.Trial({'x0': x}) for x in x_test]
    multi_prior_mse = _compute_mse(multi_prior_gp_designer, test_trials, y_test)
    single_prior_mse = _compute_mse(
        single_prior_gp_designer, test_trials, y_test
    )
    self.assertLess(multi_prior_mse, single_prior_mse)


if __name__ == '__main__':
  # Jax disables float64 computations by default and will silently convert
  # float64s to float32s. We must explicitly enable float64.
  jax.config.update('jax_enable_x64', True)
  jax.config.update('jax_log_compiles', True)
  absltest.main()
