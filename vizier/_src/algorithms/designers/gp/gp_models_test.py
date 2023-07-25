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

"""Tests for `train_gp.py`."""

from typing import Callable

import jax
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
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
) -> tuple[gp_models.GPState, types.ModelData]:
  """Sets up a trained GP and outputs completed studies for `f`.

  Args:
    f: 1D objective to be optimized, i.e. f(x), where x is a scalar in [-5., 5.)
    num_train: Number of training samples to generate.
    num_test: Number of testing samples to generate.
    linear_coef: If set, uses a linear kernel with coef `linear_coef` for the GP
    ensemble_size: Ensembles together `ensemble_size` GPs.

  Returns:
  A trained GP on the training set.
  A test set.
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
  gp = gp_models.train_gp(
      data=train_data,
      ard_optimizer=optimizers.default_optimizer(),
      ard_rng=jax.random.PRNGKey(0),
      coroutine=gp_models.get_vizier_gp_coroutine(
          features=train_data.features, linear_coef=linear_coef
      ),
      ensemble_size=ensemble_size,
  )
  test_data, _ = create_model_data(num_entries=num_test)
  return gp, test_data


class TrainedGPTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(linear_coef=0.0, ensemble_size=1),
      dict(linear_coef=0.4, ensemble_size=1),
      dict(linear_coef=0.0, ensemble_size=5),
      dict(linear_coef=0.4, ensemble_size=5),
  )
  def test_mse(self, *, linear_coef: float = 0.0, ensemble_size: int = 1):
    f = lambda x: -((x - 0.5) ** 2)
    gp, test_data = _setup_lambda_search(
        f,
        num_train=100,
        num_test=100,
        linear_coef=linear_coef,
        ensemble_size=ensemble_size,
    )
    test_pred_dist, _ = gp.predictive.predict_with_aux(test_data.features)

    # We need this reshape to prevent a broadcast from (num_samples, ) -
    # (num_samples, 1) yielding (num_samples, num_samples) and breaking this
    # calculation.
    test_labels_reshaped = np.asarray(test_data.labels.unpad()).reshape(-1)
    mse = np.sum(np.square(test_pred_dist.mean() - test_labels_reshaped))
    self.assertLess(mse, 2e-2)


if __name__ == '__main__':
  # Jax disables float64 computations by default and will silently convert
  # float64s to float32s. We must explicitly enable float64.
  jax.config.update('jax_enable_x64', True)
  absltest.main()
