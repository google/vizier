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

"""Scalarizations that weights multiple metrics into a scalar."""

import abc
from typing import Callable, Optional

import attr
import jax
from jax import numpy as jnp
from jax import typing
import jaxtyping as jt


@attr.define
class Scalarization(abc.ABC):
  """Reduces an array of objectives to a single float.

  Assumes all objectives are for MAXIMIZATION.
  """

  # Weights shape should be broadcastable with objectives when called.
  weights: typing.ArrayLike = attr.ib()

  def __attrs_post_init__(self):
    if any(self.weights <= 0):
      raise ValueError(f'Non-positive weights {self.weights}')

  def __call__(
      self, objectives: jt.Float[jax.Array, '*BATCH OBJ']
  ) -> jt.Float[jax.Array, '*BATCH']:
    pass


# Scalarization factory from weights.
ScalarizationFromWeights = Callable[[typing.ArrayLike], Scalarization]


@attr.define
class LinearScalarization(Scalarization):
  """Linear Scalarization."""

  def __call__(
      self, objectives: jt.Float[jax.Array, '*BATCH OBJ']
  ) -> jt.Float[jax.Array, '*BATCH']:
    return jnp.sum(self.weights * objectives, axis=-1)


@attr.define
class ChebyshevScalarization(Scalarization):
  """Chebyshev Scalarization."""

  def __call__(
      self, objectives: jt.Float[jax.Array, '*BATCH OBJ']
  ) -> jt.Float[jax.Array, '*BATCH']:
    return jnp.min(objectives * self.weights, axis=-1)


@attr.define
class HyperVolumeScalarization(Scalarization):
  """HyperVolume Scalarization."""

  reference_point: Optional[jt.Float[jax.Array, '* #OBJ']] = attr.ib(
      default=None, kw_only=True
  )

  def __call__(
      self, objectives: jt.Float[jax.Array, '*BATCH OBJ']
  ) -> jt.Float[jax.Array, '*BATCH']:
    # Uses scalarizations in https://arxiv.org/abs/2006.04655 for
    # non-convex multiobjective optimization. Removes the exponentiation
    # factor in number of objectives as it is a monotone transformation and
    # removes the non-negativity for easier gradients.
    if self.reference_point is not None:
      return jnp.min(
          (objectives - self.reference_point) / self.weights, axis=-1
      )
    else:
      return jnp.min(objectives / self.weights, axis=-1)


@attr.define
class LinearAugmentedScalarization(Scalarization):
  """Scalarization augmented with a linear sum.

  See https://arxiv.org/pdf/1904.05760.pdf.
  """

  scalarization_factory: ScalarizationFromWeights = attr.ib(
      kw_only=True, default=HyperVolumeScalarization
  )
  augment_weight: float = attr.ib(
      default=1.0,
      validator=[attr.validators.instance_of(float), attr.validators.ge(0.0)],
      kw_only=True,
  )

  def __call__(
      self, objectives: jt.Float[jax.Array, '*BATCH OBJ']
  ) -> jt.Float[jax.Array, '*BATCH']:
    return self.scalarization_factory(self.weights)(
        objectives
    ) + self.augment_weight * LinearScalarization(weights=self.weights)(
        objectives
    )
