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

"""Tests on core.py."""


import chex
from jax import numpy as jnp
from vizier._src.jax.optimizers import core
from absl.testing import absltest


class GetBestParamsTest(absltest.TestCase):

  def test_best_n_none(self):
    losses = jnp.array([
        3.0,
        1.0,
        4.0,
        2.0,
        5.0,
    ])
    all_params = {
        'a': losses + 0.1,
        'b': jnp.stack([losses + 0.2, losses + 0.3], axis=1),
    }
    actual = core.get_best_params(losses, all_params, best_n=None)
    expected = {'a': jnp.array(1.1), 'b': jnp.array([1.2, 1.3])}

    chex.assert_trees_all_close(actual, expected)

  def test_best_n_2(self):
    losses = jnp.array([
        3.0,
        1.0,
        4.0,
        2.0,
        5.0,
    ])
    all_params = {
        'a': losses + 0.1,
        'b': jnp.stack([losses + 0.2, losses + 0.3], axis=1),
    }
    actual = core.get_best_params(losses, all_params, best_n=2)
    expected = {
        'a': jnp.array([1.1, 2.1]),
        'b': jnp.array([[1.2, 1.3], [2.2, 2.3]]),
    }

    chex.assert_trees_all_close(actual, expected)


if __name__ == '__main__':
  absltest.main()
