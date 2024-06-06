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

"""Tests for gp_bandit."""

from typing import Callable, Union
import unittest
from unittest import mock

import jax
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import gp_bandit
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.algorithms.optimizers import eagle_strategy as es
from vizier._src.algorithms.optimizers import lbfgsb_optimizer as lo
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier._src.algorithms.testing import simplekd_runner
from vizier._src.algorithms.testing import test_runners
from vizier._src.benchmarks.experimenters.synthetic import simplekd
from vizier._src.jax import types
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
      dict(iters=3, batch_size=2, num_seed_trials=1, ensemble_size=2),
      dict(
          iters=3,
          batch_size=1,
          num_seed_trials=1,
          ensemble_size=2,
          acquisition_optimizer_factory=lbfgsb_optimizer_factory,
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
          acquisition_optimizer_factory=lbfgsb_optimizer_factory,
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

  def test_invariance_to_trials_padding_on_flat_mixed_space(
      self,
  ):
    problem = vz.ProblemStatement(test_studies.flat_space_with_all_types())
    problem.metric_information.append(
        vz.MetricInformation(
            name='metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    iters = 5
    num_seed_trials = 1
    acquisition_optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=es.VectorizedEagleStrategyFactory(),
        max_evaluations=100,
    )
    # TODO: The test fails with proper ARD. Fix that and then turn
    # on ARD again.
    noop_ard_optimizer = optimizers.default_optimizer(maxiter=0)
    desinger_rng = jax.random.PRNGKey(0)
    designer = gp_bandit.VizierGPBandit(
        problem=problem,
        acquisition_optimizer_factory=acquisition_optimizer_factory,
        ard_optimizer=noop_ard_optimizer,
        num_seed_trials=num_seed_trials,
        rng=desinger_rng,
    )
    padding_designer = gp_bandit.VizierGPBandit(
        problem=problem,
        acquisition_optimizer_factory=acquisition_optimizer_factory,
        ard_optimizer=noop_ard_optimizer,
        num_seed_trials=num_seed_trials,
        padding_schedule=padding.PaddingSchedule(
            num_trials=padding.PaddingType.MULTIPLES_OF_10,
        ),
        rng=desinger_rng,
    )
    metrics_runner_seed = 1
    designer_suggestions = test_runners.RandomMetricsRunner(
        problem,
        iters=iters,
        verbose=1,
        seed=metrics_runner_seed,
        validate_parameters=True,
    ).run_designer(designer)

    padding_designer_suggestions = test_runners.RandomMetricsRunner(
        problem,
        iters=iters,
        verbose=1,
        seed=metrics_runner_seed,
        validate_parameters=True,
    ).run_designer(padding_designer)

    self.assertLen(designer_suggestions, iters)
    self.assertLen(padding_designer_suggestions, iters)
    for idx, (suggestion, padding_suggestion) in enumerate(
        zip(designer_suggestions, padding_designer_suggestions)
    ):
      params1 = suggestion.parameters.as_dict()
      params2 = padding_suggestion.parameters.as_dict()
      self.assertSameElements(params1.keys(), params2.keys())
      for key in params1.keys():
        self.assertAlmostEqual(
            params1[key],
            params2[key],
            places=5,
            msg=f'Mismatch in parameter: {key}, suggestions {idx}',
        )

  def test_prediction_accuracy(self):
    f = lambda x: -((x - 0.5) ** 2)
    gp_designer, obs_trials, _ = _setup_lambda_search(f)
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
          acquisition_optimizer_factory=vectorized_optimizer_factory,
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

  def test_parallel_acquisition(self):
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.append(
        vz.MetricInformation(
            name='metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )

    def _qei_factory(data: types.ModelData) -> acquisitions.AcquisitionFunction:
      best_labels = acquisitions.get_best_labels(data.labels)
      return acquisitions.QEI(best_labels=best_labels, num_samples=100)

    scoring_fn_factory = acquisitions.bayesian_scoring_function_factory(
        _qei_factory
    )

    n_parallel = 4
    iters = 3
    designer = gp_bandit.VizierGPBandit(
        problem=problem,
        acquisition_optimizer_factory=vectorized_optimizer_factory,
        ard_optimizer=optimizers.JaxoptLbfgsB(
            optimizers.LbfgsBOptions(maxiter=5, num_line_search_steps=5)
        ),
        scoring_function_factory=scoring_fn_factory,
        scoring_function_is_parallel=True,
        use_trust_region=False,
        num_seed_trials=n_parallel,
        ensemble_size=3,
        rng=jax.random.PRNGKey(0),
        linear_coef=0.1,
    )
    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=iters,
            batch_size=n_parallel,
            verbose=1,
            validate_parameters=True,
            seed=1,
        ).run_designer(designer),
        iters * n_parallel,
    )

  def test_multi_metrics(self):
    search_space = vz.SearchSpace()
    search_space.root.add_float_param('x0', -5.0, 5.0)
    problem = vz.ProblemStatement(
        search_space=search_space,
        metric_information=vz.MetricsConfig(
            metrics=[
                vz.MetricInformation(
                    'obj1', goal=vz.ObjectiveMetricGoal.MAXIMIZE
                ),
                vz.MetricInformation(
                    'obj2', goal=vz.ObjectiveMetricGoal.MAXIMIZE
                ),
            ]
        ),
    )

    iters = 2
    designer = gp_bandit.VizierGPBandit.from_problem(problem)
    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=iters,
            verbose=1,
            validate_parameters=True,
            seed=1,
        ).run_designer(designer),
        iters,
    )


class GPBanditSimplekDTest(parameterized.TestCase):
  """Simplekd convergence tests for gp bandit designer."""

  @parameterized.parameters(
      dict(best_category='corner', max_relative_error=0.5),
      dict(best_category='center', max_relative_error=0.1),
      dict(best_category='mixed', max_relative_error=0.1),
  )
  def test_convergence(
      self,
      best_category: simplekd.SimpleKDCategory,
      *,
      max_relative_error: float,
  ) -> None:
    simplekd_runner.SimpleKDConvergenceTester(
        best_category=best_category,
        designer_factory=(
            # pylint: disable=g-long-lambda
            lambda problem, seed: gp_bandit.VizierGPBandit(
                problem,
                rng=jax.random.PRNGKey(seed),
                padding_schedule=padding.PaddingSchedule(
                    num_trials=padding.PaddingType.MULTIPLES_OF_10
                ),
            )
        ),
        num_trials=20,
        max_relative_error=max_relative_error,
        num_repeats=1,
        target_num_convergence=1,
    ).assert_convergence()


# TODO: Fix transfer learning and enable tests.
@unittest.skip('The current transfer learning seems broken and test failing.')
class GPBanditPriorsTest(parameterized.TestCase):

  def test_prior_warping(self):
    f = lambda x: -((x - 0.5) ** 2)
    transform_f = lambda x: -3 * ((x - 0.5) ** 2) + 10

    # X is in range of what is defined in `_setup_lambda_search`, [-5.0, 5.0)
    x_test = np.random.default_rng(1).uniform(-5.0, 5.0, 100)
    y_test = [transform_f(x) for x in x_test]
    test_trials = [vz.Trial({'x0': x}) for x in x_test]

    # Create the designer with a prior and the trials to train the prior.
    gp_designer_with_prior, obs_trials_for_prior, _ = _setup_lambda_search(
        f=f, num_trials=100
    )

    # Set priors to above trials.
    gp_designer_with_prior.set_priors(
        [vza.CompletedTrials(obs_trials_for_prior)]
    )

    # Create a no prior designer on the transformed function `transform_f`.
    # Also use the generated trials to update both the designer with prior and
    # the designer without. This tests that the prior designer is resilient
    # to linear transforms between the prior and the top level study.
    gp_designer_no_prior, obs_trials, _ = _setup_lambda_search(
        f=transform_f, num_trials=20
    )

    # Update both designers with the actual study.
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

  @parameterized.parameters(
      dict(iters=3, batch_size=5),
      dict(iters=5, batch_size=1),
  )
  def test_run_with_priors(self, *, iters, batch_size):
    f = lambda x: -((x - 0.5) ** 2)

    # Create the designer with a prior and the trials to train the prior.
    gp_designer_with_prior, obs_trials_for_prior, problem = (
        _setup_lambda_search(f=f, num_trials=100)
    )

    # Set priors to the above trials.
    gp_designer_with_prior.set_priors(
        [vza.CompletedTrials(obs_trials_for_prior)]
    )

    self.assertLen(
        test_runners.RandomMetricsRunner(
            problem,
            iters=iters,
            batch_size=batch_size,
            verbose=1,
            validate_parameters=True,
            seed=1,
        ).run_designer(gp_designer_with_prior),
        iters * batch_size,
    )


if __name__ == '__main__':
  # Jax disables float64 computations by default and will silently convert
  # float64s to float32s. We must explicitly enable float64.
  jax.config.update('jax_enable_x64', True)
  absltest.main()
