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

from unittest import mock

import jax
from jax import numpy as jnp
from jax.config import config
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.jax import types
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

  def test_acq_tr_good_point(self):
    acq = acquisitions.AcquisitionTrustRegion.default_ucb_pi()
    self.assertAlmostEqual(
        acq(tfd.Normal(jnp.float64(0.1), 1), labels=jnp.array([100.0])),
        -1.0e12,
    )

  def test_acq_tr_bad_point(self):
    acq = acquisitions.AcquisitionTrustRegion.default_ucb_pi()
    self.assertAlmostEqual(
        acq(tfd.Normal(jnp.float64(0.1), 1), labels=jnp.array([-100.0])),
        1.9,
    )

  def test_qei(self):
    acq = acquisitions.QEI(num_samples=2000)
    batch_shape = [6]
    dist = tfd.Normal(loc=0.1 * jnp.ones(batch_shape), scale=1.0)
    qei = acq(dist, labels=jnp.array([0.2]), seed=jax.random.PRNGKey(0))
    # QEI reduces over the batch shape.
    self.assertEmpty(qei.shape)

    dist_single_point = tfd.Normal(jnp.array([0.1], dtype=jnp.float64), 1)
    qei_single_point = acq(
        dist_single_point, labels=jnp.array([0.2]), seed=jax.random.PRNGKey(0)
    )
    # Parallel matches non-parallel for a single point.
    np.testing.assert_allclose(qei_single_point, 0.346, atol=1e-2)
    self.assertEmpty(qei_single_point.shape)

  def test_qucb_shape(self):
    acq = acquisitions.QUCB()
    batch_shape = [6]
    dist = tfd.Normal(loc=0.1 * jnp.ones(batch_shape), scale=1.0)
    qucb = acq(dist, labels=jnp.array([0.2]), seed=jax.random.PRNGKey(0))
    # QUCB reduces over the batch shape.
    self.assertEmpty(qucb.shape)

  def test_qucb_equals_ucb(self):
    # The QUCB coefficient should be multiplied by sqrt(pi/2) for equivalency
    # with the UCB coefficient (assuming a Gaussian distribution).
    acq_ucb = acquisitions.UCB(coefficient=0.5)
    acq_qucb = acquisitions.QUCB(
        num_samples=5000, coefficient=0.5 * np.sqrt(np.pi / 2.0)
    )
    dist_single_point = tfd.Normal(jnp.array([0.1], dtype=jnp.float64), 1)
    qucb_single_point = acq_qucb(
        dist_single_point, labels=jnp.array([0.2]), seed=jax.random.PRNGKey(1)
    )
    ucb_single_point = acq_ucb(dist_single_point, labels=jnp.array([0.2]))
    # Parallel matches non-parallel for a single point.
    np.testing.assert_allclose(
        qucb_single_point, ucb_single_point[0], atol=2e-2
    )
    self.assertEmpty(qucb_single_point.shape)

  def test_multi_acquisition(self):
    ucb = acquisitions.UCB()
    ei = acquisitions.EI()
    acq = acquisitions.MultiAcquisitionFunction({'ucb': ucb, 'ei': ei})
    dist = tfd.Normal(jnp.float64(0.1), 1)
    labels = jnp.array([0.2])
    acq_val = acq(dist, labels=labels)
    ucb_val = ucb(dist, labels=labels)
    ei_val = ei(dist, labels=labels)
    np.testing.assert_allclose(acq_val, jnp.stack([ucb_val, ei_val]))


class TrustRegionTest(absltest.TestCase):

  def test_trust_region_small(self):
    data = types.StochasticProcessModelData(
        features=np.array([
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]),
        labels=np.array([0.0, 0.0]),
    )
    tr = acquisitions.TrustRegion.build(
        _build_mock_continuous_array_specs(4), data=data
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
    features = np.vstack(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
        * 10
    )
    labels = np.sum(features, axis=-1)
    data = types.StochasticProcessModelData(
        features=features,
        labels=labels,
    )
    tr = acquisitions.TrustRegion.build(
        _build_mock_continuous_array_specs(4),
        data=data,
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
    data = types.StochasticProcessModelData(
        features=np.array([
            [0.0, 0.0, 0.0, 0.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
        ]),
        labels=np.array([0.0, 0.0]),
    )
    tr = acquisitions.TrustRegion.build(
        _build_mock_continuous_array_specs(4),
        data=data,
    )
    data_masked = types.StochasticProcessModelData(
        features=np.array([
            [0.0, 0.0, 0.0, 0.0, np.nan, np.nan],
            [1.0, 1.0, 1.0, 1.0, np.nan, np.nan],
        ]),
        labels=np.array([0.0, 0.0]),
        dimension_is_missing=np.array([False, False, False, False, True, True]),
    )
    tr = acquisitions.TrustRegion.build(
        _build_mock_continuous_array_specs(4),
        data=data_masked,
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
    data = types.StochasticProcessModelData(
        features=trusted, labels=np.random.randn(n_trusted)
    )
    xs = types.ContinuousAndCategoricalArray(
        continuous=np.random.randn(n_samples, d_cont),
        categorical=np.random.randn(n_samples, d_cat),
    )
    tr = acquisitions.TrustRegion.build(specs=None, data=data)
    self.assertEqual(tr.min_linf_distance(xs).shape, (n_samples,))

  def test_trust_region_small(self):
    trusted = types.ContinuousAndCategoricalArray(
        continuous=np.array([
            [0.0, 0.0],
            [1.0, 1.0],
        ]),
        categorical=np.random.randint(0, 10, size=(2, 4)),
    )
    data = types.StochasticProcessModelData(
        features=trusted, labels=np.random.randn(2)
    )
    tr = acquisitions.TrustRegion.build(specs=None, data=data)
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


if __name__ == '__main__':
  config.update('jax_enable_x64', True)
  absltest.main()
