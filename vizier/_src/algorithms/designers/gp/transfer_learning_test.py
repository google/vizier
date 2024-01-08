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

"""Tests for `transfer_learning.py`."""

import chex
import jax
from tensorflow_probability.substrates import jax as tfp
from vizier._src.algorithms.designers.gp import transfer_learning as vtl

from absl.testing import absltest


tfd = tfp.distributions


class GoogleGpBanditTest(absltest.TestCase):

  def test_sequential_combine_predictions(self):
    prior_pred = vtl.TransferPredictionState(
        pred=tfd.Normal(loc=[0.1, 0.2], scale=[10.0, 5.0]),
        aux={},
        training_data_count=10,
        num_hyperparameters=5,
    )  # Try one case with `training_data_count` > `num_hyperparameters`.

    top_pred = vtl.TransferPredictionState(
        pred=tfd.Normal(loc=[0.1, 0.2], scale=[10.0, 5.0]),
        aux={},
        training_data_count=10,
        num_hyperparameters=15,
    )  # Try one case with `training_data_count` < `num_hyperparameters`.

    comb_pred, comb_aux = vtl.combine_predictions_with_aux(
        top_pred=top_pred, base_pred=prior_pred
    )

    # The sum of means should be precisely `0.1 + 0.1` and `0.2 + 0.2`.
    self.assertEqual(list(comb_pred.mean()), [0.2, 0.4])

    # The combination of stddevs should be approximately the same as each
    # individual standard deviation, because we are combining the same
    # standard deviations together.
    self.assertSequenceAlmostEqual(comb_pred.stddev(), [10.0, 5.0], places=4)

    # Batch shapes in aux must be the same as the batch shape of predictions
    chex.assert_tree_shape_prefix(comb_aux, [2])


if __name__ == '__main__':
  # Jax disables float64 computations by default and will silently convert
  # float64s to float32s. We must explicitly enable float64.
  jax.config.update('jax_enable_x64', True)
  jax.config.update('jax_log_compiles', True)
  absltest.main()
