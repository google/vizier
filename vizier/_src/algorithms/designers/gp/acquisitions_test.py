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

import jax
from jax import config
from jax import numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier._src.algorithms.designers import scalarization
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.jax import types
from absl.testing import absltest

tfd = tfp.distributions
tfpk = tfp.math.psd_kernels
tfpke = tfp.experimental.psd_kernels


def _make_test_model_data(labels_array, num_categorical=1):
  labels = types.PaddedArray.as_padded(labels_array)
  features = types.ModelInput(
      continuous=types.PaddedArray.as_padded(
          jnp.zeros((labels_array.shape[0], 3), dtype=jnp.float64)
      ),
      categorical=types.PaddedArray.as_padded(
          jnp.zeros(
              (labels_array.shape[0], num_categorical), dtype=types.INT_DTYPE
          ),
      ),
  )
  return types.ModelData(features=features, labels=labels)


class AcquisitionsTest(absltest.TestCase):

  def test_ucb(self):
    acq = acquisitions.UCB(coefficient=2.0)
    self.assertAlmostEqual(acq(tfd.Normal(0.1, 1)), 2.1)

  def test_lcb(self):
    acq = acquisitions.LCB(coefficient=2.0)
    self.assertAlmostEqual(acq(tfd.Normal(0.1, 1)), -1.9)

  def test_hvs(self):
    acq = acquisitions.HyperVolumeScalarization(coefficient=2.0)
    self.assertAlmostEqual(acq(tfd.Normal(0.1, 1)), 0.1)

  def test_ei(self):
    labels = types.PaddedArray.as_padded(jnp.array([[0.2]]))
    best_labels = acquisitions.get_best_labels(labels)
    acq = acquisitions.EI(best_labels)
    self.assertAlmostEqual(
        acq(
            tfd.Normal(jnp.float64(0.1), 1),
        ),
        0.34635347,
    )

  def test_scalarized_ucb(self):
    labels = types.PaddedArray.as_padded(
        jnp.array([[0.2, 0.3], [0.01, 0.5], [0.5, 0.01]])
    )
    reference_point = acquisitions.get_worst_labels(labels)
    ucb = acquisitions.UCB(coefficient=2.0)
    scalarizer = scalarization.HyperVolumeScalarization(
        weights=jnp.array([0.1, 0.2]), reference_point=reference_point
    )
    acq = acquisitions.ScalarizedAcquisition(ucb, scalarizer)
    self.assertAlmostEqual(
        acq(tfd.Normal([0.1, 0.2], [1, 2])), jnp.array(20.9), delta=1e-2
    )

  def test_ehvi_approximation(self):
    labels = types.PaddedArray.as_padded(
        jnp.array([[0.2, 0.3], [0.01, 0.5], [0.5, 0.01]])
    )
    reference_point = acquisitions.get_reference_point(labels)
    ucb = acquisitions.UCB(coefficient=2.0)
    scalarizer = scalarization.HyperVolumeScalarization(
        weights=jnp.array([[0.1, 0.2], [0.2, 0.1], [1, 1]]),
        reference_point=reference_point,
    )
    acq = acquisitions.ScalarizedAcquisition(
        ucb, scalarizer, reduction_fn=lambda x: jnp.mean(x, axis=0)
    )
    self.assertAlmostEqual(
        acq(tfd.Normal([0.1, 0.2], [1, 2])), jnp.array(11.343), delta=1e-2
    )

  def test_pi(self):
    labels = types.PaddedArray.as_padded(jnp.array([[0.2]]))
    best_labels = acquisitions.get_best_labels(labels)
    acq = acquisitions.PI(best_labels)
    self.assertAlmostEqual(
        acq(
            tfd.Normal(jnp.float64(0.1), 1),
        ),
        0.46017216,
    )

  def test_mes(self):
    num_obs = 10
    num_pred = 6
    labels = np.random.normal(size=([num_obs, 1]))
    data = _make_test_model_data(labels, num_categorical=0)
    init_features = types.ModelInput(
        continuous=types.PaddedArray.as_padded(
            jnp.ones((num_pred, 3), dtype=jnp.float64)
        ),
        categorical=types.PaddedArray.as_padded(
            jnp.zeros((num_pred, 0), dtype=types.INT_DTYPE)
        ),
    )

    class _TestPredictive:

      def predict_with_aux(self, x):
        gp = tfd.GaussianProcess(
            kernel=tfpke.FeatureScaledWithCategorical(
                kernel=tfpk.ExponentiatedQuadratic(),
                scale_diag=tfpke.ContinuousAndCategoricalValues(
                    continuous=jnp.ones([3]), categorical=jnp.ones([0])
                ),
            ),
            index_points=tfpke.ContinuousAndCategoricalValues(
                continuous=data.features.continuous.padded_array,
                categorical=data.features.categorical.padded_array,
            ),
            observation_noise_variance=np.float64(1.0),
        )
        return (
            gp.posterior_predictive(
                observations=data.labels.padded_array[:, 0],
                predictive_index_points=tfpke.ContinuousAndCategoricalValues(
                    x.continuous.padded_array, x.categorical.padded_array
                ),
            ),
            {},
        )

    score_fn = acquisitions.MaxValueEntropySearch.scoring_fn_factory(
        data=data, predictive=_TestPredictive(), use_trust_region=True
    )
    score = score_fn.score(init_features, seed=jax.random.PRNGKey(0))
    self.assertEqual(score.shape, (num_pred,))

  def test_acq_pi_tr_good_point(self):
    data = _make_test_model_data((jnp.array([[100.0]])))
    acq = acquisitions.AcquisitionTrustRegion.default_ucb_pi(data=data)
    self.assertAlmostEqual(
        acq(tfd.Normal(jnp.float64(0.1), 1)),
        -1.0e12,
    )

  def test_acq_pi_tr_bad_point(self):
    data = _make_test_model_data((jnp.array([[-100.0]])))
    acq = acquisitions.AcquisitionTrustRegion.default_ucb_pi(data=data)
    self.assertAlmostEqual(
        acq(tfd.Normal(jnp.float64(0.1), 1)),
        1.9,
    )

  def test_acq_lcb_tr_bad_point(self):
    data = _make_test_model_data((jnp.array([[100.0]])))
    acq = acquisitions.AcquisitionTrustRegion.default_ucb_lcb(data=data)
    self.assertAlmostEqual(
        acq(tfd.Normal(jnp.float64(0.1), 1)),
        jnp.array([-1e12]),
        delta=2.0,
    )

  def test_acq_lcb_tr_good_point(self):
    data = _make_test_model_data((jnp.array([[-100.0]])))
    acq = acquisitions.AcquisitionTrustRegion.default_ucb_lcb(data=data)
    self.assertAlmostEqual(
        acq(tfd.Normal(jnp.float64(0.1), 1)),
        1.9,
    )

  def test_acq_lcb_delay_tr(self):
    data = _make_test_model_data((jnp.array([[100.0], [-100.0]])))
    acq = acquisitions.AcquisitionTrustRegion.default_ucb_lcb_delay_tr(
        data=data
    )
    self.assertAlmostEqual(
        acq(tfd.Normal(jnp.float64(0.1), 1)),
        jnp.array([1.9]),
    )

  def test_qei(self):
    best_labels = jnp.array([0.2])
    acq = acquisitions.QEI(num_samples=2000, best_labels=best_labels)
    batch_shape = [6]
    dist = tfd.Normal(loc=0.1 * jnp.ones(batch_shape), scale=1.0)
    qei = acq(dist, seed=jax.random.PRNGKey(0))
    # QEI reduces over the batch shape.
    self.assertEmpty(qei.shape)

    dist_single_point = tfd.Normal(jnp.array([0.1], dtype=jnp.float64), 1)
    qei_single_point = acq(dist_single_point, seed=jax.random.PRNGKey(0))
    # Parallel matches non-parallel for a single point.
    np.testing.assert_allclose(qei_single_point, 0.346, atol=1e-2)
    self.assertEmpty(qei_single_point.shape)

  def test_qpi(self):
    best_labels = jnp.array([0.2])
    acq = acquisitions.QPI(num_samples=5000, best_labels=best_labels)
    batch_shape = [6]
    dist = tfd.Normal(loc=0.1 * jnp.ones(batch_shape), scale=1.0)
    qpi = acq(dist, seed=jax.random.PRNGKey(0))
    # QPI reduces over the batch shape.
    self.assertEmpty(qpi.shape)

    dist_single_point = tfd.Normal(jnp.array([0.1], dtype=jnp.float64), 1)
    qpi_single_point = acq(dist_single_point, seed=jax.random.PRNGKey(0))
    # Parallel matches non-parallel for a single point.
    pi_single_point = acquisitions.PI(best_labels=best_labels)(
        dist_single_point
    )
    np.testing.assert_allclose(qpi_single_point, pi_single_point[0], atol=1e-2)
    self.assertEmpty(qpi_single_point.shape)

  def test_qucb_shape(self):
    acq = acquisitions.QUCB()
    batch_shape = [6]
    dist = tfd.Normal(loc=0.1 * jnp.ones(batch_shape), scale=1.0)
    qucb = acq(dist, seed=jax.random.PRNGKey(0))
    # QUCB reduces over the batch shape.
    self.assertEmpty(qucb.shape)

  def test_qucb_equals_ucb(self):
    # The QUCB coefficient should be multiplied by sqrt(pi/2) for equivalency
    # with the UCB coefficient (assuming a Gaussian distribution).
    acq_ucb = acquisitions.UCB(coefficient=0.5)
    acq_qucb = acquisitions.QUCB(
        num_samples=5000,
        coefficient=0.5 * np.sqrt(np.pi / 2.0),
    )
    dist_single_point = tfd.Normal(jnp.array([0.1], dtype=jnp.float64), 1)
    qucb_single_point = acq_qucb(dist_single_point, seed=jax.random.PRNGKey(1))
    ucb_single_point = acq_ucb(dist_single_point)
    # Parallel matches non-parallel for a single point.
    np.testing.assert_allclose(
        qucb_single_point, ucb_single_point[0], atol=2e-2
    )
    self.assertEmpty(qucb_single_point.shape)

  def test_multi_acquisition(self):
    best_labels = jnp.array([0.2])
    ucb = acquisitions.UCB()
    ei = acquisitions.EI(best_labels=best_labels)
    acq = acquisitions.MultiAcquisitionFunction({'ucb': ucb, 'ei': ei})
    dist = tfd.Normal(jnp.float64(0.1), 1)
    acq_val = acq(dist)
    ucb_val = ucb(dist)
    ei_val = ei(dist)
    np.testing.assert_allclose(acq_val, jnp.stack([ucb_val, ei_val]))


class TrustRegionTest(absltest.TestCase):

  def test_trust_region_small(self):
    trusted = types.ModelInput(
        continuous=types.PaddedArray.as_padded(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
        ),
        categorical=types.PaddedArray.as_padded(np.array([[0, 0], [1, 1]])),
    )
    tr = acquisitions.TrustRegion(trusted=trusted)

    xs = types.ModelInput(
        continuous=types.PaddedArray.as_padded(
            np.array([
                [0.0, 0.3],
                [0.9, 0.8],
                [1.0, 1.0],
            ]),
        ),
        categorical=types.PaddedArray.as_padded(
            np.array([[0, 0], [1, 1], [0, 1]]),
        ),
    )
    np.testing.assert_allclose(
        tr.min_linf_distance(xs),
        np.array([0.3, 0.2, 0.0]),
    )
    self.assertAlmostEqual(tr.trust_radius, 0.224, places=3)

  def test_trust_region_bigger(self):
    xs_cont = np.vstack(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
        * 10
    )
    xs = types.ModelInput(
        continuous=types.PaddedArray.as_padded(xs_cont),
        categorical=types.PaddedArray.as_padded(
            np.ones(xs_cont.shape, dtype=types.INT_DTYPE),
        ),
    )
    tr = acquisitions.TrustRegion(trusted=xs)

    xs_cont_test = np.array([[0.0, 0.3], [0.9, 0.8], [1.0, 1.0]])
    xs_test = types.ModelInput(
        continuous=types.PaddedArray.as_padded(xs_cont_test),
        categorical=types.PaddedArray.as_padded(
            np.ones(xs_cont_test.shape, dtype=types.INT_DTYPE),
        ),
    )
    np.testing.assert_allclose(
        tr.min_linf_distance(xs_test),
        np.array([0.3, 0.2, 0.0]),
    )
    self.assertAlmostEqual(tr.trust_radius, 0.44, places=3)

  def test_trust_region_padded_small(self):
    # Test that padding still retrieves the same distance computations as
    # `test_trust_region_small`.
    xs_cont = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
    ])
    xs = types.ModelInput(
        continuous=types.PaddedArray.from_array(
            xs_cont, target_shape=(4, 6), fill_value=0.0
        ),
        categorical=types.PaddedArray.from_array(
            np.ones(xs_cont.shape, dtype=types.INT_DTYPE),
            target_shape=(4, 5),
            fill_value=0,
        ),
    )
    tr = acquisitions.TrustRegion(
        trusted=xs,
    )

    xs_cont_test = np.array([
        [[0.0, 0.3], [0.9, 0.8], [1.0, 1.0]],
        [[1.0, 1.0], [0.0, 0.3], [0.9, 0.8]],
    ])
    xs_test = types.ModelInput(
        continuous=types.PaddedArray.from_array(
            xs_cont_test, target_shape=(3, 3, 6), fill_value=-100.0
        ),
        categorical=types.PaddedArray.from_array(
            np.ones(xs_cont_test.shape, dtype=types.INT_DTYPE),
            target_shape=(3, 3, 5),
            fill_value=-100,
        ),
    )
    np.testing.assert_allclose(
        tr.min_linf_distance(xs_test),
        np.array([[0.3, 0.2, 0.0], [0.0, 0.3, 0.2], [0.0, 0.0, 0.0]]),
    )
    self.assertAlmostEqual(tr.trust_radius, 0.224, places=3)

  def test_trust_region_padded_all_categorical_multi_batch_dims(self):
    xs = types.ModelInput(
        continuous=types.PaddedArray.as_padded(jnp.array([])),
        categorical=types.PaddedArray.from_array(
            np.ones((2, 2), dtype=types.INT_DTYPE),
            target_shape=(4, 5),
            fill_value=0,
        ),
    )
    tr = acquisitions.TrustRegion(
        trusted=xs,
    )
    xs_test = types.ModelInput(
        continuous=types.PaddedArray.as_padded(jnp.array([])),
        categorical=types.PaddedArray.from_array(
            np.ones((2, 3, 2), dtype=types.INT_DTYPE),
            target_shape=(3, 3, 5),
            fill_value=-100,
        ),
    )
    np.testing.assert_allclose(
        tr.min_linf_distance(xs_test),
        np.ones((3, 3)) * -np.inf,
    )


if __name__ == '__main__':
  config.update('jax_enable_x64', True)
  absltest.main()
