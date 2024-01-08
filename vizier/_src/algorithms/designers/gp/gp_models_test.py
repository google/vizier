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

"""Tests for `train_gp.py`."""

from typing import Callable

import jax
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.algorithms.designers.gp import gp_models
from vizier._src.jax import types
from vizier.jax import optimizers
from vizier.pyvizier import converters

from absl.testing import absltest
from absl.testing import parameterized


def _setup_lambda_search(
    f: Callable[[float], float],
    num_train: int = 100,
    num_test: int = 100,
    linear_coef: float = 0.0,
    ensemble_size: int = 1,
) -> tuple[gp_models.GPTrainingSpec, types.ModelData, types.ModelData]:
  """Sets up training state for a GP and outputs an test set for `f`.

  Args:
    f: 1D objective to be optimized, i.e. f(x), where x is a scalar in [-5., 5.)
    num_train: Number of training samples to generate.
    num_test: Number of testing samples to generate.
    linear_coef: If set, uses a linear kernel with coef `linear_coef` for the GP
    ensemble_size: Ensembles together `ensemble_size` GPs.

  Returns:
  A GP training spec.
  A generated train set.
  A generated test set.
  """
  assert num_train > 0 and num_test > 0, (
      f'Must provide a positive number of trials. Got {num_train} training and'
      f' {num_test} testing.'
  )

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

  converter = converters.TrialToModelInputConverter.from_problem(problem)
  quasi_random_designer = quasi_random.QuasiRandomDesigner(
      problem.search_space, seed=1
  )

  def create_model_data(
      num_entries: int,
  ) -> tuple[types.ModelData, list[vz.Trial]]:
    suggestions = quasi_random_designer.suggest(num_entries)
    obs_trials: list[vz.Trial] = []
    for idx, suggestion in enumerate(suggestions):
      trial = suggestion.to_trial(idx)
      x = suggestions[idx].parameters['x0'].value
      trial.complete(vz.Measurement(metrics={'obj': f(x)}))
      obs_trials.append(trial)

    model_data = converter.to_xy(obs_trials)
    return model_data, obs_trials

  train_data, _ = create_model_data(num_entries=num_train)
  train_spec = gp_models.GPTrainingSpec(
      ard_optimizer=optimizers.default_optimizer(),
      ard_rng=jax.random.PRNGKey(0),
      coroutine=gp_models.get_vizier_gp_coroutine(
          data=train_data, linear_coef=linear_coef
      ),
      ensemble_size=ensemble_size,
      ard_random_restarts=optimizers.DEFAULT_RANDOM_RESTARTS,
  )
  test_data, _ = create_model_data(num_entries=num_test)
  return train_spec, train_data, test_data


def _compute_mse(
    predictive: acquisitions.Predictive, test_data: types.ModelData
) -> float:
  """Computes the mean-squared error of `predictive` on `test_data."""

  pred_dist, _ = predictive.predict_with_aux(test_data.features)

  # We need this reshape to prevent a broadcast from (num_samples, ) -
  # (num_samples, 1) yielding (num_samples, num_samples) and breaking this
  # calculation.
  test_labels_reshaped = np.asarray(test_data.labels.unpad()).reshape(-1)

  mse = np.sum(np.square(pred_dist.mean() - test_labels_reshaped))
  return mse


class TrainedGPTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(linear_coef=0.0, ensemble_size=1),
      dict(linear_coef=0.4, ensemble_size=1),
      dict(linear_coef=0.0, ensemble_size=5),
      dict(linear_coef=0.4, ensemble_size=5),
  )
  def test_mse_no_base(
      self, *, linear_coef: float = 0.0, ensemble_size: int = 1
  ):
    f = lambda x: -((x - 0.5) ** 2)
    spec, train_data, test_data = _setup_lambda_search(
        f,
        num_train=100,
        num_test=100,
        linear_coef=linear_coef,
        ensemble_size=ensemble_size,
    )
    gp = gp_models.train_gp(spec, train_data)
    mse = _compute_mse(gp, test_data)
    self.assertLess(mse, 2e-2)


class StackedResidualGPTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(linear_coef=0.0, ensemble_size=1),
      dict(linear_coef=0.4, ensemble_size=1),
      dict(linear_coef=0.0, ensemble_size=5),
      dict(linear_coef=0.4, ensemble_size=5),
  )
  def test_sequential_base_accuracy(
      self, *, linear_coef: float = 0.0, ensemble_size: int = 1
  ):
    """Tests that a good base with a bad top beats a bad independent model.

    Train a base on n = 100 samples, and a top GP with n = 5 samples.
    Combine these together into one predictor.

    Compare the MSE of this predictor with a test predictor trained on n = 5
    samples. The transfer learning enabled predictor should beat the test
    predictor.

    Args:
      linear_coef: The linear coefficient for the GP. Used for all trained GPs.
      ensemble_size: The number of GPs to ensemble together. Used for all
        trained GPs.
    """
    base_samples = 100
    bad_num_samples = 5
    num_test = 100
    f = lambda x: -((x - 0.5) ** 2)

    base_spec, base_train_data, _ = _setup_lambda_search(
        f,
        num_train=base_samples,
        num_test=num_test,
        linear_coef=linear_coef,
        ensemble_size=ensemble_size,
    )

    # This is purposefully bad with `bad_num_samples`
    top_spec, top_train_data, test_data = _setup_lambda_search(
        f,
        num_train=bad_num_samples,
        num_test=num_test,
        linear_coef=linear_coef,
        ensemble_size=ensemble_size,
    )

    # Combine the good base and the bad top into transfer learning GP.
    seq_base_gp = gp_models.train_gp(
        [base_spec, top_spec], [base_train_data, top_train_data]
    )

    # Create a purposefully-bad GP with `bad_num_samples` for comparison.
    test_gp_spec, test_gp_train_data, _ = _setup_lambda_search(
        f,
        num_train=bad_num_samples,
        num_test=num_test,
        linear_coef=linear_coef,
        ensemble_size=ensemble_size,
    )
    test_gp = gp_models.train_gp(test_gp_spec, test_gp_train_data)

    seq_base_mse = _compute_mse(seq_base_gp, test_data)
    test_mse = _compute_mse(test_gp, test_data)

    self.assertLess(seq_base_mse, test_mse)

  @parameterized.parameters(
      dict(linear_coef=0.0, ensemble_size=1),
      dict(linear_coef=0.4, ensemble_size=1),
      dict(linear_coef=0.0, ensemble_size=5),
      dict(linear_coef=0.4, ensemble_size=5),
  )
  def test_multi_base(
      self, *, linear_coef: float = 0.0, ensemble_size: int = 1
  ):
    """Tests that multiple bases predict well.

    Train two good bases on n = 100 samples, and a top on n = 5 samples

    The MSE of this predictor should be similar to a GP trained on n = 100
    samples.

    Args:
      linear_coef: The linear coefficient for the GP. Used for all trained GPs.
      ensemble_size: The number of GPs to ensemble together. Used for all
        trained GPs.
    """
    large_n = 100
    small_n = 5
    num_test = 100
    f = lambda x: -((x - 0.5) ** 2)

    # Create a top GP with low training data
    top_spec, top_train_data, _ = _setup_lambda_search(
        f,
        num_train=small_n,
        num_test=num_test,
        linear_coef=linear_coef,
        ensemble_size=ensemble_size,
    )

    train_specs = []
    train_data = []

    for _ in range(2):
      base_spec, base_train_data, _ = _setup_lambda_search(
          f,
          num_train=large_n,
          num_test=num_test,
          linear_coef=linear_coef,
          ensemble_size=ensemble_size,
      )
      train_specs.append(base_spec)
      train_data.append(base_train_data)
    train_specs.append(top_spec)
    train_data.append(top_train_data)

    seq_base_gp = gp_models.train_gp(train_specs, train_data)

    # Create a good GP with sufficient training data
    test_gp_spec, test_gp_train_data, test_data = _setup_lambda_search(
        f,
        num_train=large_n,
        num_test=num_test,
        linear_coef=linear_coef,
        ensemble_size=ensemble_size,
    )
    test_gp = gp_models.train_gp(test_gp_spec, test_gp_train_data)

    seq_base_mse = _compute_mse(seq_base_gp, test_data)
    test_mse = _compute_mse(test_gp, test_data)

    self.assertAlmostEqual(seq_base_mse, test_mse, places=4)
    self.assertLess(test_mse, 2e-2)

  @parameterized.parameters(
      dict(linear_coef=0.0, ensemble_size=1),
      dict(linear_coef=0.4, ensemble_size=1),
      dict(linear_coef=0.0, ensemble_size=5),
      dict(linear_coef=0.4, ensemble_size=5),
  )
  def test_bad_base_resilience(
      self, *, linear_coef: float = 0.0, ensemble_size: int = 1
  ):
    """Tests that predictions are resilient to a bad base.

    Train a bad base on a fake objective with n = 100 samples.
    Trains a top predictor on the actual objective with n = 100 samples.
    Combines them into one predictor.

    The MSE of this predictor better than a GP trained purely on the bad
    objective.

    Args:
      linear_coef: The linear coefficient for the GP. Used for all trained GPs.
      ensemble_size: The number of GPs to ensemble together. Used for all
        trained GPs.
    """
    large_n = 100
    num_test = 100
    f = lambda x: -((x - 0.5) ** 2)
    fake_f = lambda x: x**5

    bad_base_spec, bad_base_train_data, _ = _setup_lambda_search(
        fake_f,
        num_train=large_n,
        num_test=num_test,
        linear_coef=linear_coef,
        ensemble_size=ensemble_size,
    )

    top_spec, top_train_data, test_data = _setup_lambda_search(
        f,
        num_train=large_n,
        num_test=num_test,
        linear_coef=linear_coef,
        ensemble_size=ensemble_size,
    )

    # Combine the good base and the bad top into transfer learning GP.
    seq_base_gp = gp_models.train_gp(
        [
            bad_base_spec,
            top_spec,
        ],
        [bad_base_train_data, top_train_data],
    )

    # Create a GP on the fake objective with sufficient training data
    test_gp_spec, test_gp_train_data, _ = _setup_lambda_search(
        fake_f,
        num_train=large_n,
        num_test=num_test,
        linear_coef=linear_coef,
        ensemble_size=ensemble_size,
    )
    test_gp = gp_models.train_gp(test_gp_spec, test_gp_train_data)

    seq_base_mse = _compute_mse(seq_base_gp, test_data)
    test_mse = _compute_mse(test_gp, test_data)

    self.assertLess(seq_base_mse, test_mse)

  def test_single_list_same_as_singleton(self):
    """Tests that `[state]` and `state` are treated the same."""
    large_n = 100
    num_test = 100
    f = lambda x: -((x - 0.5) ** 2)
    spec, train_data, test_data = _setup_lambda_search(
        f, num_train=large_n, num_test=num_test
    )

    list_gp = gp_models.train_gp(spec, train_data)
    singleton_gp = gp_models.train_gp([spec], [train_data])

    list_gp_mse = _compute_mse(list_gp, test_data)
    singleton_gp_mse = _compute_mse(singleton_gp, test_data)

    self.assertAlmostEqual(list_gp_mse, singleton_gp_mse)

  def test_multi_task(self):
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

    converter = converters.TrialToModelInputConverter.from_problem(problem)
    quasi_random_designer = quasi_random.QuasiRandomDesigner(
        problem.search_space, seed=1
    )
    num_entries = 100
    suggestions = quasi_random_designer.suggest(num_entries)
    obs_trials: list[vz.Trial] = []
    for idx, suggestion in enumerate(suggestions):
      trial = suggestion.to_trial(idx)
      x = suggestions[idx].parameters['x0'].value
      trial.complete(vz.Measurement(metrics={'obj1': x + 1, 'obj2': 2 * x - 1}))
      obs_trials.append(trial)

    train_entries = 60
    train_trials = obs_trials[:train_entries]
    test_trials = obs_trials[train_entries:]
    model_data = converter.to_xy(train_trials)
    train_spec = gp_models.GPTrainingSpec(
        ard_optimizer=optimizers.default_optimizer(),
        ard_rng=jax.random.PRNGKey(0),
        coroutine=gp_models.get_vizier_gp_coroutine(data=model_data),
    )
    gp = gp_models.train_gp(train_spec, model_data)

    test_data = converter.to_xy(test_trials)
    pred_dist, _ = gp.predict_with_aux(test_data.features)
    mse = np.mean(np.square(pred_dist.mean() - test_data.labels.unpad()))
    self.assertLess(mse, 1e-2)


if __name__ == '__main__':
  # Jax disables float64 computations by default and will silently convert
  # float64s to float32s. We must explicitly enable float64.
  jax.config.update('jax_enable_x64', True)
  absltest.main()
