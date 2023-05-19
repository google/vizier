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

import jax
from jax import numpy as jnp
import mock
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import gp_bandit
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.jax import stochastic_process_model as sp
from vizier._src.jax import types
from vizier._src.jax.models import tuned_gp_models
from vizier._src.jax.optimizers import optimizers
from vizier.pyvizier import converters

from absl.testing import absltest


tfd = tfp.distributions


def _build_mock_continuous_array_specs(n):
  continuous_spec = mock.create_autospec(converters.NumpyArraySpec)
  continuous_spec.type = converters.NumpyArraySpecType.CONTINUOUS
  continuous_spec.num_dimensions = 1
  return [continuous_spec] * n


class AcquisitionsTest(absltest.TestCase):

  def test_ucb(self):
    acq = acquisitions.UCB(coefficient=2.0)
    self.assertAlmostEqual(acq(tfd.Normal(0.1, 1)), 2.1)

  def test_hvs(self):
    acq = acquisitions.HyperVolumeScalarization(coefficient=2.0)
    self.assertAlmostEqual(acq(tfd.Normal(0.1, 1)), 0.1)

  def test_ei(self):
    acq = acquisitions.EI()
    self.assertAlmostEqual(
        acq(tfd.Normal(jnp.float64(0.1), 1), labels=jnp.array([0.2])),
        0.34635347,
    )

  def test_pi(self):
    acq = acquisitions.PI()
    self.assertAlmostEqual(
        acq(tfd.Normal(jnp.float64(0.1), 1), labels=jnp.array([0.2])),
        0.46017216,
    )

  def test_qei(self):
    acq = acquisitions.QEI(num_samples=2000)
    batch_shape = [6]
    dist = tfd.Normal(loc=0.1 * jnp.ones(batch_shape), scale=1.0)
    qei = acq(dist, labels=jnp.array([0.2]))
    # QEI reduces over the batch shape.
    self.assertEmpty(qei.shape)

    dist_single_point = tfd.Normal(jnp.array([0.1], dtype=jnp.float64), 1)
    qei_single_point = acq(dist_single_point, labels=jnp.array([0.2]))
    # Parallel matches non-parallel for a single point.
    np.testing.assert_allclose(qei_single_point, 0.346, atol=1e-2)
    self.assertEmpty(qei_single_point.shape)

  def test_qucb_shape(self):
    acq = acquisitions.QUCB()
    batch_shape = [6]
    dist = tfd.Normal(loc=0.1 * jnp.ones(batch_shape), scale=1.0)
    qucb = acq(dist, labels=jnp.array([0.2]))
    # QUCB reduces over the batch shape.
    self.assertEmpty(qucb.shape)

  def test_qucb_equals_ucb(self):
    # The QUCB coefficient should be multiplied by sqrt(pi/2) for equivalency
    # with the UCB coefficient (assuming a Gaussian distribution).
    acq_ucb = acquisitions.UCB(coefficient=2.0)
    acq_qucb = acquisitions.QUCB(
        num_samples=2000, coefficient=2.0 * np.sqrt(np.pi / 2.0)
    )
    dist_single_point = tfd.Normal(jnp.array([0.1], dtype=jnp.float64), 1)
    qucb_single_point = acq_qucb(dist_single_point, labels=jnp.array([0.2]))
    ucb_single_point = acq_ucb(dist_single_point, labels=jnp.array([0.2]))
    # Parallel matches non-parallel for a single point.
    np.testing.assert_allclose(
        qucb_single_point, ucb_single_point[0], atol=1e-2
    )
    self.assertEmpty(qucb_single_point.shape)


class TrustRegionTest(absltest.TestCase):

  def test_trust_region_small(self):
    tr = acquisitions.TrustRegion(
        np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]),
        _build_mock_continuous_array_specs(4),
    )

    np.testing.assert_allclose(
        tr.min_linf_distance(
            np.array([
                [0.0, 0.2, 0.3, 0.0],
                [0.9, 0.8, 0.9, 0.9],
                [1.0, 1.0, 1.0, 1.0],
            ]),
        ),
        np.array([0.3, 0.2, 0.0]),
    )
    self.assertAlmostEqual(tr.trust_radius, 0.224, places=3)

  def test_trust_region_bigger(self):
    tr = acquisitions.TrustRegion(
        np.vstack(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
            * 10
        ),
        _build_mock_continuous_array_specs(4),
    )
    np.testing.assert_allclose(
        tr.min_linf_distance(
            np.array([
                [0.0, 0.2, 0.3, 0.0],
                [0.9, 0.8, 0.9, 0.9],
                [1.0, 1.0, 1.0, 1.0],
            ]),
        ),
        np.array([0.3, 0.2, 0.0]),
    )
    self.assertAlmostEqual(tr.trust_radius, 0.44, places=3)

  def test_trust_region_padded_small(self):
    # Test that padding still retrieves the same distance computations as
    # `test_trust_region_small`.
    tr = acquisitions.TrustRegion(
        np.array([
            [0.0, 0.0, 0.0, 0.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
        ]),
        _build_mock_continuous_array_specs(4),
        feature_is_missing=np.array([False, False, False, False, True, True]),
    )

    np.testing.assert_allclose(
        tr.min_linf_distance(
            np.array([
                [0.0, 0.2, 0.3, 0.0, -100.0, 20.0],
                [0.9, 0.8, 0.9, 0.9, 23.0, 27.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, -3.0],
            ]),
        ),
        np.array([0.3, 0.2, 0.0]),
    )
    self.assertAlmostEqual(tr.trust_radius, 0.224, places=3)


class TrustRegionWithCategoricalTest(absltest.TestCase):

  def test_trust_region_with_categorical(self):
    n_trusted = 20
    n_samples = 5
    d_cont = 6
    d_cat = 3

    trusted = types.ContinuousAndCategoricalArray(
        np.random.randn(n_trusted, d_cont), np.random.randn(n_trusted, d_cat)
    )
    xs = types.ContinuousAndCategoricalArray(
        continuous=np.random.randn(n_samples, d_cont),
        categorical=np.random.randn(n_samples, d_cat),
    )
    tr = acquisitions.TrustRegionWithCategorical(trusted)
    self.assertEqual(tr.min_linf_distance(xs).shape, (n_samples,))

  def test_trust_region_small(self):
    trusted = types.ContinuousAndCategoricalArray(
        continuous=np.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ]),
        categorical=np.random.randint(0, 10, size=(2, 4)),
    )
    tr = acquisitions.TrustRegionWithCategorical(trusted)
    np.testing.assert_allclose(
        tr.min_linf_distance(
            types.ContinuousAndCategoricalArray(
                continuous=np.array([
                    [0.0, 0.3],
                    [0.9, 0.8],
                    [1.0, 1.0],
                ]),
                categorical=np.random.randint(0, 10, size=(3, 4)),
            )
        ),
        np.array([0.3, 0.2, 0.0]),
    )


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
    gp_designer._compute_state()
    xs = np.random.randn(10, 16)
    samples = gp_designer._acquisition_builder.sample_on_array(
        xs, 15, jax.random.PRNGKey(0)
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
    model, loss_fn = (
        tuned_gp_models.VizierGaussianProcessWithCategorical.model_and_loss_fn(
            features, labels
        )
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
    state = {'params': best_model_params, **pp_state}
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
    converter = converters.TrialToArrayConverter.from_study_config(
        problem,
        scale=True,
        pad_oovs=True,
        max_discrete_indices=0,
        flip_sign_for_minimization_metrics=True,
    )
    acq_builder = acquisitions.GPBanditAcquisitionBuilder(
        use_trust_region=False
    )
    acq_builder.build(
        problem, model, state, features, labels, converter, use_vmap=use_vmap
    )
    pred_dict = jax.tree_map(
        np.sum,
        jax.tree_map(np.isnan, acq_builder.predict_on_array(xs)),
    )
    self.assertEqual(pred_dict['mean'], 0.0)
    self.assertEqual(pred_dict['stddev'], 0.0)
    self.assertEqual(np.sum(np.isnan(acq_builder.acquisition_on_array(xs))), 0)


if __name__ == '__main__':
  absltest.main()
