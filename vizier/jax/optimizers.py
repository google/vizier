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

"""Thin wrappers around Jax optimizers."""
# pylint: disable=unused-import

import functools

from vizier._src.jax.optimizers.core import LossFunction
from vizier._src.jax.optimizers.core import Optimizer
from vizier._src.jax.optimizers.core import Params
from vizier._src.jax.optimizers.core import Setup
from vizier._src.jax.optimizers.jaxopt_wrappers import JaxoptLbfgsB
from vizier._src.jax.optimizers.jaxopt_wrappers import JaxoptScipyLbfgsB
from vizier._src.jax.optimizers.jaxopt_wrappers import LbfgsBOptions
from vizier._src.jax.optimizers.optax_wrappers import OptaxTrainWithRandomRestarts


@functools.lru_cache
def default_optimizer(random_restarts: int = 4, maxiter: int = 50) -> Optimizer:
  """Default optimizer that works okay for most cases."""
  # NOTE: Production algorithms are recommended to stay away from using this.
  return JaxoptScipyLbfgsB(
      LbfgsBOptions(
          random_restarts=random_restarts, maxiter=maxiter, best_n=None
      )
  )
