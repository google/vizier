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

"""Input warping.

Input warping has shown to help optimize non-stationary functions:
http://proceedings.mlr.press/v32/snoek14.pdf.

Note that the dth output dimension is a function of the dth input dimension.

Example
-------
search_space = ...
problem = vz.ProblemStatement(search_space)
converter = converters.TrialToArrayConverter.from_study_config(problem)
input_warper = input_warping.KumaraswamyInputWarpingConverter(
    converter, a=0.1, b=0.8
)
trials = ...
features = input_warper.to_features(trials)
"""

from typing import Sequence

import attr
import numpy as np
from vizier import pyvizier as vz
from vizier.pyvizier.converters import core


def kumaraswamy_cdf(x: np.ndarray, a: float, b: float) -> np.ndarray:
  """Compute the Kumaraswamy CDF.

  Arguments:
    x: values in [0,1]. shape: (num_samples, num_features)
    a: positive value.
    b: positive value.

  Returns:
    The CDF(x). shape: (num_samples, num_cdfs).
  """
  return 1 - (1 - x**a) ** b


def kumaraswamy_inv_cdf(f: np.ndarray, a: float, b: float) -> np.ndarray:
  """Compute the inverse of the Kumaraswamy CDF.

  Arguments:
    f: values in [0,1]. shape: (num_samples, num_cdfs)
    a: positive value.
    b: positive value.

  Returns:
    The Inv_CDF(x). shape: (num_samples, num_features).
  """
  return (1 - (1 - f) ** (1 / b)) ** (1 / a)


@attr.define
class KumaraswamyInputWarpingConverter:
  """Input Warping based on Kumaraswamy distribution.

  Reference: https://en.wikipedia.org/wiki/Kumaraswamy_distribution
  """

  converter: core.TrialToArrayConverter
  a: float = attr.field(validator=attr.validators.instance_of(float))
  b: float = attr.field(validator=attr.validators.instance_of(float))

  def __attrs_post_init__(self):
    if self.a <= 0:
      raise ValueError(f"Attribute 'a' has to be positive, received {self.a}")
    if self.b <= 0:
      raise ValueError(f"Attribute 'b' has to be positive, received {self.b}")

  def to_features(self, trials) -> np.ndarray:
    features = self.converter.to_features(trials)
    return kumaraswamy_cdf(features, self.a, self.b)

  def to_xy(self, trials) -> tuple[np.ndarray, np.ndarray]:
    return self.to_features(trials), self.converter.to_labels(trials)

  def to_parameters(self, features: np.ndarray) -> Sequence[vz.ParameterDict]:
    """Convert to warput features into parameters."""
    return self.converter.to_parameters(
        kumaraswamy_inv_cdf(features, self.a, self.b)
    )
