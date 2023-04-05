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

"""Experimenter factories."""

from typing import Optional, Protocol

import attr
import numpy as np
from vizier._src.benchmarks.experimenters import discretizing_experimenter
from vizier._src.benchmarks.experimenters import experimenter
from vizier._src.benchmarks.experimenters import noisy_experimenter
from vizier._src.benchmarks.experimenters import normalizing_experimenter
from vizier._src.benchmarks.experimenters import numpy_experimenter
from vizier._src.benchmarks.experimenters import shifting_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob


class ExperimenterFactory(Protocol):
  """Abstraction for creating Experimenters."""

  def __call__(
      self, *, seed: Optional[int] = None
  ) -> experimenter.Experimenter:
    """Creates the Experimenter."""


@attr.define
class BBOBExperimenterFactory(ExperimenterFactory):
  """Factory for a BBOB function."""

  # Should be a BBOB function name in bbob.py (name should match exactly).
  name: str = attr.field(validator=attr.validators.instance_of(str))
  dim: int = attr.field(
      validator=[attr.validators.instance_of(int),
                 attr.validators.gt(0)])

  def __call__(
      self, seed: Optional[int] = None
  ) -> numpy_experimenter.NumpyExperimenter:
    del seed
    bbob_function = getattr(bbob, self.name, None)
    if bbob_function is None:
      raise ValueError(f'{self.name} is not a valid BBOB function in bbob.py')
    return numpy_experimenter.NumpyExperimenter(
        bbob_function, bbob.DefaultBBOBProblemStatement(self.dim))


@attr.define
class SingleObjectiveExperimenterFactory(ExperimenterFactory):
  """Factory for a single objective Experimenter."""

  base_factory: ExperimenterFactory = attr.field()
  # An array of doubles that is broadcastable to dim of search space.
  shift: Optional[np.ndarray] = attr.field(default=None)
  # Should be one of the noise types in noisy_experimenter.py
  noise_type: Optional[str] = attr.field(default=None)
  # Number of normalization samples. If zero, no normalization is done.
  num_normalization_samples: int = attr.field(default=0)
  # Dictionary of parameter indices to discretize in a grid.
  # Key = index of parameter to be discretize and Value = Number of feasible
  # points to discretize to. For example, {0: 3, 2 : 2} discretizes the first
  # parameter to 3 feasible points and the third to 2 feasible points.
  # Note: Generally, this should be used only when base_factory generates
  # only continuous parameters.
  discrete_dict: dict[int, int] = attr.field(default=attr.Factory(dict))
  # Dictionary of parameter indices to categorize in a grid.
  # Key = index of parameter to be categorize and Value = Number of feasible
  # points to categorize to. See discrete_dict.
  categorical_dict: dict[int, int] = attr.field(default=attr.Factory(dict))

  def __call__(self, seed: Optional[int] = None) -> experimenter.Experimenter:
    """Creates the SingleObjective Experimenter."""
    exptr = self.base_factory()
    if self.shift is not None:
      exptr = shifting_experimenter.ShiftingExperimenter(
          exptr, shift=self.shift)
    if self.num_normalization_samples:
      exptr = normalizing_experimenter.NormalizingExperimenter(
          exptr, num_normalization_samples=self.num_normalization_samples
      )

    # Discretization and categorization.
    if self.discrete_dict.keys() & self.categorical_dict.keys():
      raise ValueError(
          f'{self.discrete_dict} discretizing indicies overlap with '
          f'{self.categorical_dict} categorical indicies'
      )

    pcs = list(exptr.problem_statement().search_space.parameters)
    if self.discrete_dict:
      discretization = {
          pcs[idx].name: points for idx, points in self.discrete_dict.items()
      }
      exptr = (
          discretizing_experimenter.DiscretizingExperimenter.create_with_grid(
              exptr, discretization, convert_to_str=False
          )
      )

    if self.categorical_dict:
      categorization = {
          pcs[idx].name: points for idx, points in self.categorical_dict.items()
      }
      exptr = (
          discretizing_experimenter.DiscretizingExperimenter.create_with_grid(
              exptr, categorization, convert_to_str=True
          )
      )
    if self.noise_type is not None:
      exptr = noisy_experimenter.NoisyExperimenter.from_type(
          exptr, noise_type=self.noise_type.upper(), seed=seed
      )

    return exptr
