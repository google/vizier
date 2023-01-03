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
from vizier._src.benchmarks.experimenters import experimenter
from vizier._src.benchmarks.experimenters import noisy_experimenter
from vizier._src.benchmarks.experimenters import numpy_experimenter
from vizier._src.benchmarks.experimenters import shifting_experimenter
from vizier._src.benchmarks.experimenters.synthetic import bbob


class ExperimenterFactory(Protocol):
  """Abstraction for creating Experimenters."""

  def __call__(self) -> experimenter.Experimenter:
    """Creates the Experimenter."""


@attr.define
class BBOBExperimenterFactory(ExperimenterFactory):
  """Factory for a BBOB function."""

  # Should be a BBOB function name in bbob.py (name should match exactly).
  name: str = attr.field(validator=attr.validators.instance_of(str))
  dim: int = attr.field(
      validator=[attr.validators.instance_of(int),
                 attr.validators.gt(0)])

  def __call__(self) -> experimenter.Experimenter:
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

  def __call__(self) -> experimenter.Experimenter:
    """Creates the SingleObjective Experimenter."""
    exptr = self.base_factory()
    if self.shift is not None:
      exptr = shifting_experimenter.ShiftingExperimenter(
          exptr, shift=self.shift)
    if self.noise_type is not None:
      exptr = noisy_experimenter.NoisyExperimenter(
          exptr, noise_type=self.noise_type.upper())
    return exptr
