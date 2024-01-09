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

"""Scalarizations that weights multiple metrics into a scalar."""

import abc
from typing import Callable, Optional

import equinox as eqx
import jax
from jax import numpy as jnp
import jaxtyping as jt
import typeguard


class Scalarization(abc.ABC, eqx.Module):
  """Reduces an array of objectives to a single float.

  Assumes all objectives are for MAXIMIZATION.
  """

  # Weights shape should be broadcastable with objectives when called.
  weights: jt.Float[jax.Array, '#Obj'] = eqx.field(converter=jnp.asarray)

  @jt.jaxtyped(typechecker=typeguard.typechecked)
  def __call__(
      self, objectives: jt.Float[jax.Array, '*Batch Obj']
  ) -> jt.Float[jax.Array, '*Batch']:
    pass


# Scalarization factory from weights.
ScalarizationFromWeights = Callable[
    [jt.Float[jax.Array, '#Obj']], Scalarization
]


class LinearScalarization(Scalarization):
  """Linear Scalarization."""

  @jt.jaxtyped(typechecker=typeguard.typechecked)
  def __call__(
      self, objectives: jt.Float[jax.Array, '*Batch Obj']
  ) -> jt.Float[jax.Array, '*Batch']:
    return jnp.sum(self.weights * objectives, axis=-1)


class ChebyshevScalarization(Scalarization):
  """Chebyshev Scalarization."""

  @jt.jaxtyped(typechecker=typeguard.typechecked)
  def __call__(
      self, objectives: jt.Float[jax.Array, '*Batch Obj']
  ) -> jt.Float[jax.Array, '*Batch']:
    return jnp.min(objectives * self.weights, axis=-1)


class HyperVolumeScalarization(Scalarization):
  """HyperVolume Scalarization."""

  reference_point: Optional[jt.Float[jax.Array, '* #Obj']] = eqx.field(
      default=None
  )

  @jt.jaxtyped(typechecker=typeguard.typechecked)
  def __call__(
      self, objectives: jt.Float[jax.Array, '*Batch Obj']
  ) -> jt.Float[jax.Array, '*Batch']:
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


class LinearAugmentedScalarization(Scalarization):
  """Scalarization augmented with a linear sum.

  See https://arxiv.org/pdf/1904.05760.pdf.
  """

  scalarization_factory: ScalarizationFromWeights = eqx.field(static=True)
  augment_weight: jt.Float[jax.Array, ''] = eqx.field(
      default=1.0, converter=jnp.asarray
  )

  @jt.jaxtyped(typechecker=typeguard.typechecked)
  def __call__(
      self, objectives: jt.Float[jax.Array, '*Batch Obj']
  ) -> jt.Float[jax.Array, '*Batch']:
    return self.scalarization_factory(self.weights)(
        objectives
    ) + self.augment_weight * LinearScalarization(weights=self.weights)(
        objectives
    )
