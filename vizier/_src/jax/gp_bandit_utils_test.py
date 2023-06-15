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

"""Tests for gp_bandit_utils."""

import functools

from absl.testing import parameterized
import jax
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.gp import acquisitions
from vizier._src.algorithms.optimizers import random_vectorized_optimizer as rvo
from vizier._src.jax import gp_bandit_utils
from vizier._src.jax import types
from vizier._src.jax.models import tuned_gp_models
from vizier.pyvizier import converters

from absl.testing import absltest


class GpBanditUtilsTest(parameterized.TestCase):

  @parameterized.parameters((3, acquisitions.QEI), (None, acquisitions.EI))
  def test_optimize_acquisition(
      self, num_parallel_candidates, acquisition_fn_class
  ):
    num_features = 2
    num_obs = 12
    features = np.random.normal(size=[num_obs, num_features]).astype(np.float32)
    labels = np.random.normal(size=[num_obs]).astype(np.float32)

    model = tuned_gp_models.VizierGaussianProcess.build_model(features)

    data = types.StochasticProcessModelData(
        features=features,
        labels=labels,
    )
    params = model.init(jax.random.PRNGKey(0), features)
    _, pp_state = gp_bandit_utils.precompute_cholesky(
        model=model, data=data, params=params['params']
    )
    model_state = dict(**params, **pp_state)
    state = types.GPState(data=data, model_state=model_state)

    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('f1', 0.0, 10.0)
    problem.search_space.root.add_float_param('f2', 0.0, 10.0)

    if num_parallel_candidates:
      prior_features = np.stack([features] * num_parallel_candidates, axis=-1)
      for i in range(1, num_parallel_candidates):
        problem.search_space.root.add_float_param(f'f1_{i}', 0.0, 10.0)
        problem.search_space.root.add_float_param(f'f2_{i}', 0.0, 10.0)
      expected_shape = [1, num_features * num_parallel_candidates]
    else:
      prior_features = features
      expected_shape = [1, num_features]

    converter = converters.TrialToArrayConverter.from_study_config(problem)

    random_optimizer = rvo.create_random_optimizer(
        converter=converter, max_evaluations=100, suggestion_batch_size=10
    )

    acquisition_fn = functools.partial(
        acquisition_fn_class(), seed=jax.random.PRNGKey(1)
    )
    suggestions = gp_bandit_utils.optimize_acquisition(
        count=1,
        model=model,
        acquisition_fn=acquisition_fn,
        optimizer=random_optimizer,
        prior_features=prior_features,
        state=state,
        seed=jax.random.PRNGKey(0),
        use_vmap=False,
        num_parallel_candidates=num_parallel_candidates,
    )
    self.assertSequenceEqual(suggestions.features.shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
