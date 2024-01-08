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

"""Thin wrappers around Jax optimizers."""
# pylint: disable=unused-import

import functools

from vizier._src.jax.optimizers.core import LossFunction
from vizier._src.jax.optimizers.core import Optimizer
from vizier._src.jax.optimizers.core import Params
from vizier._src.jax.optimizers.jaxopt_wrappers import JaxoptLbfgsB
from vizier._src.jax.optimizers.jaxopt_wrappers import JaxoptScipyLbfgsB
from vizier._src.jax.optimizers.jaxopt_wrappers import LbfgsBOptions
from vizier._src.jax.optimizers.optax_wrappers import OptaxTrain

DEFAULT_RANDOM_RESTARTS = 4


@functools.lru_cache
def default_optimizer(maxiter: int = 50) -> Optimizer:
  """Default optimizer and random restarts that work okay for most cases."""
  # NOTE: Production algorithms are recommended to stay away from using this.
  return JaxoptScipyLbfgsB(LbfgsBOptions(maxiter=maxiter, best_n=None))
