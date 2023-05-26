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

"""Tests for acquisitions."""

import functools

import jax
from jax.config import config
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import gp_bandit
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.jax import gp_bandit_utils
from vizier._src.jax import predictive_fns
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier._src.jax.models import tuned_gp_models
from vizier._src.jax.optimizers import optimizers

from absl.testing import absltest


tfd = tfp.distributions


class GPBanditAcquisitionBuilderTest(absltest.TestCase):

  def test_sample_on_array(self):
    ard_optimizer = optimizers.JaxoptLbfgsB(random_restarts=8, best_n=5)
    search_space = vz.SearchSpace()
    for i in range(16):
      search_space.root.add_float_param(f'x{i}', 0.0, 1.0)

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
    gp_designer = gp_bandit.VizierGPBandit(problem, ard_optimizer=ard_optimizer)
    suggestions = quasi_random.QuasiRandomDesigner(
        problem.search_space
    ).suggest(11)

    trials = []
    for idx, suggestion in enumerate(suggestions):
      trial = suggestion.to_trial(idx)
      trial.complete(vz.Measurement(metrics={'obj': np.random.randn()}))
      trials.append(trial)

    gp_designer.update(vza.CompletedTrials(trials), vza.ActiveTrials())
    state, _ = gp_designer._compute_state()
    xs = np.random.randn(10, 16)
    samples = predictive_fns.sample_on_array(
        xs,
        15,
        jax.random.PRNGKey(0),
        model=tuned_gp_models.VizierGaussianProcess.build_model(
            state.data.features
        ),
        state=state,
    )
    self.assertEqual(samples.shape, (15, 10))
    self.assertEqual(np.sum(np.isnan(samples)), 0)

  def test_categorical_kernel(self, best_n=2):
    # Random key
    key = jax.random.PRNGKey(0)
    # Simulate data
    n_samples = 10
    n_continuous = 3
    n_categorical = 5
    features = types.ContinuousAndCategoricalArray(
        jax.random.normal(key, shape=(n_samples, n_continuous)),
        jax.random.normal(key, shape=(n_samples, n_categorical)),
    )
    labels = jax.random.normal(key, shape=(n_samples,))
    xs = features
    # Model
    model = tuned_gp_models.VizierGaussianProcessWithCategorical.build_model(
        features
    )
    loss_fn = functools.partial(
        jax.jit(
            gp_bandit_utils.stochastic_process_model_loss_fn,
            static_argnames=('model', 'normalize'),
        ),
        model=model,
        data=types.StochasticProcessModelData(features=features, labels=labels),
    )
    setup = lambda rng: model.init(rng, features)['params']
    constraints = sp.get_constraints(model)

    # ARD
    ard_optimizer = optimizers.JaxoptLbfgsB(random_restarts=4, best_n=best_n)
    use_vmap = ard_optimizer.best_n != 1
    best_model_params, _ = ard_optimizer(
        setup, loss_fn, key, constraints=constraints
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

    if not use_vmap:
      pp_state = precompute_cholesky(best_model_params)
    else:
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
    data = types.StochasticProcessModelData(features=features, labels=labels)
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
