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

"""Tests for acquisitions."""

import functools

import jax
from jax import config
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.jax import gp_bandit_utils
from vizier._src.jax import predictive_fns
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier._src.jax.models import tuned_gp_models
from vizier.jax import optimizers

from absl.testing import absltest


tfd = tfp.distributions


class GPBanditAcquisitionBuilderTest(absltest.TestCase):

  def test_categorical_kernel(self, best_n=2):
    # Random key
    key = jax.random.PRNGKey(0)
    # Simulate data
    n_samples = 10
    n_continuous = 3
    n_categorical = 5
    features = types.ModelInput(
        continuous=types.PaddedArray.as_padded(
            jax.random.normal(key, shape=(n_samples, n_continuous)),
        ),
        categorical=types.PaddedArray.as_padded(
            jax.random.normal(key, shape=(n_samples, n_categorical)),
        ),
    )
    labels = types.PaddedArray.as_padded(
        jax.random.normal(key, shape=(n_samples, 1)),
    )
    xs = features
    # Model
    coroutine = tuned_gp_models.VizierGaussianProcess(
        _dim=types.ContinuousAndCategorical[int](n_continuous, n_categorical)
    )
    model = sp.StochasticProcessModel(coroutine)
    loss_fn = functools.partial(
        jax.jit(
            gp_bandit_utils.stochastic_process_model_loss_fn,
            static_argnames=('model', 'normalize'),
        ),
        model=model,
        data=types.ModelData(features=features, labels=labels),
    )
    setup = lambda rng: model.init(rng, features)['params']
    constraints = sp.get_constraints(model)

    # ARD
    ard_optimizer = optimizers.JaxoptScipyLbfgsB()
    key, init_key = jax.random.split(key, 2)
    best_model_params, _ = ard_optimizer(
        init_params=jax.vmap(setup)(jax.random.split(init_key, 4)),
        loss_fn=loss_fn,
        rng=key,
        constraints=constraints,
        best_n=best_n,
    )

    def precompute_cholesky(params):
      _, pp_state = model.apply(
          {'params': params},
          features,
          labels,
          method=model.precompute_predictive,
          mutable='predictive',
      )
      return pp_state

    pp_state = jax.vmap(precompute_cholesky)(best_model_params)

    # Create the state.
    model_state = {'params': best_model_params, **pp_state}
    # Define the problem.
    space = vz.SearchSpace()
    root = space.root
    for j in range(n_continuous):
      root.add_float_param(f'f{j}', -1.0, 2.0)
    for j in range(n_categorical):
      root.add_categorical_param(f'c{j}', ['a', 'b', 'c'])
    problem = vz.ProblemStatement(space)
    problem.metric_information.append(
        vz.MetricInformation(
            name='metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    data = types.ModelData(features=features, labels=labels)
    state = types.GPState(model_state=model_state, data=data)
    pred_dict = predictive_fns.predict_on_array(xs, model=model, state=state)

    acquisition_val = predictive_fns.acquisition_on_array(
        xs, model, acquisition_fn=acquisitions.UCB(), state=state
    )
    self.assertFalse(np.any(np.isnan(pred_dict['mean'])))
    self.assertFalse(np.any(np.isnan(pred_dict['stddev'])))
    self.assertFalse(np.any(np.isnan(acquisition_val)))


if __name__ == '__main__':
  config.update('jax_enable_x64', True)
  absltest.main()
