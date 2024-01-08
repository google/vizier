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

"""Applies the noise function to each metric in final measurement."""

import functools
from typing import Callable, Optional, Sequence

import attr
import numpy as np
from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter


@attr.define(auto_attribs=True)
class NoisyExperimenter(experimenter.Experimenter):
  """NoisyExperimenter applies noise to all metric(s) in final measurement.

  It stores the unnoised metric as metric_name + '_before_noise'. The noise
  function is a callable that is applied to metrics.
  """

  exptr: experimenter.Experimenter = attr.field()
  noise_fn: Callable[[float], float] = attr.field()

  @classmethod
  def from_type(
      cls,
      exptr: experimenter.Experimenter,
      noise_type: str,
      seed: Optional[int] = None,
  ) -> 'NoisyExperimenter':
    """Initializes noise_fn with noise given by noise_type."""
    dim = len(exptr.problem_statement().search_space.parameters)
    noise_fn = _create_noise_fn(
        noise_type,
        dimension=dim,
        seed=seed,
    )
    return cls(exptr, noise_fn)

  def problem_statement(self) -> pyvizier.ProblemStatement:
    return self.exptr.problem_statement()

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    self.exptr.evaluate(suggestions)
    for suggestion in suggestions:
      if suggestion.final_measurement is None:
        continue
      metric_dict_with_noise = {}
      for name, metric in suggestion.final_measurement.metrics.items():
        metric_dict_with_noise[name] = pyvizier.Metric(
            value=self.noise_fn(metric.value))
        metric_dict_with_noise[name + '_before_noise'] = metric
      suggestion.final_measurement.metrics = metric_dict_with_noise

  def __repr__(self):
    return f'NoisyExperimenter({self.noise_fn}) on {str(self.exptr)}'


def _create_noise_fn(
    noise: str,
    dimension: int,
    target_value: float = 1e-8,
    seed: Optional[int] = None,
) -> Callable[[float], float]:
  """Creates a noise function via NumPy.

  See https://bee22.com/resources/bbob%20noisy%20functions.pdf

  Args:
    noise: Noise specification
    dimension: Dimensionality of bbob function that the noise is applied to.
    target_value: The noise does not apply to values less than this.
    seed:

  Returns:
    Callable that returns the noisy version of the input.

  Raises:
    ValueError: if noise is not supported.
  """
  rng = np.random.default_rng(seed or 0)
  if noise == 'NO_NOISE':
    noise_fn = lambda v: v
  elif noise == 'MODERATE_GAUSSIAN':
    noise_fn = lambda v: v * rng.lognormal(0, 0.01)
  elif noise == 'SEVERE_GAUSSIAN':
    noise_fn = lambda v: v * rng.lognormal(0, 0.1)
  elif noise == 'MODERATE_UNIFORM':
    noise_fn = functools.partial(
        _uniform_noise,
        rng=rng,
        amplifying_exponent=0.01 * (0.49 + 1.0 / dimension),
        shrinking_exponent=0.01,
    )
  elif noise == 'SEVERE_UNIFORM':
    noise_fn = functools.partial(
        _uniform_noise,
        rng=rng,
        amplifying_exponent=0.1 * (0.49 + 1.0 / dimension),
        shrinking_exponent=0.1,
    )
  elif noise == 'MODERATE_SELDOM_CAUCHY':
    noise_fn = functools.partial(
        _cauchy_noise, rng=rng, noise_strength=0.01, noise_frequency=0.05
    )
  elif noise == 'SEVERE_SELDOM_CAUCHY':
    noise_fn = functools.partial(
        _cauchy_noise, rng=rng, noise_strength=0.1, noise_frequency=0.25
    )
  elif noise == 'LIGHT_ADDITIVE_GAUSSIAN':
    return functools.partial(_additive_normal_noise, rng=rng, stddev=0.01)
  elif noise == 'MODERATE_ADDITIVE_GAUSSIAN':
    return functools.partial(_additive_normal_noise, rng=rng, stddev=0.1)
  elif noise == 'SEVERE_ADDITIVE_GAUSSIAN':
    return functools.partial(_additive_normal_noise, rng=rng, stddev=1.0)
  else:
    raise ValueError('Noise was not supported: {}'.format(noise))
  return lambda v: _stabilized_noise(v, noise_fn, target_value)


def _uniform_noise(
    value: float,
    amplifying_exponent: float,
    shrinking_exponent: float,
    rng: np.random.Generator,
    epsilon: float = 1e-99,
) -> float:
  """Uniform noise model for bbob-noisy benchmark.

  The noise strength increases when value is small.

  Args:
    value: Function value to apply noise to.
    amplifying_exponent: "alpha" in the paper. The higher this number is, the
      more likely it is for the noisy value to be greater than the input value.
      0 or less means the noise never amplifies the function value.
    shrinking_exponent: "beta" in the paper. The higher this number is, the more
      likely it is for the noisy value to be less than the input value. 0 or
      less means the noise never shrinks the function value.
    rng: Rng.
    epsilon: "epsilon" in the paper. Prevents division by zero.

  Returns:
    Noisy version of value.
  """
  f1 = np.power(rng.uniform(), np.max([0.0, shrinking_exponent]))
  f2 = np.power(1e9 / (value + epsilon), amplifying_exponent * rng.uniform())
  return value * f1 * np.max([1.0, f2])


def _additive_normal_noise(
    value: float, stddev: float, rng: np.random.Generator
) -> float:
  """Additive normal noise."""
  return value + rng.normal(0.0, stddev)


def _cauchy_noise(
    value: float,
    noise_strength: float,
    noise_frequency: float,
    rng: np.random.Generator,
) -> float:
  """Cauchy noise model for bbob-noisy benchmark.

  The noise is infrequent and difficult to analyze due to large outliers.

  Args:
    value: Function value to apply noise to.
    noise_strength: "alpha" in the paper. Its absolute value determines the
      noise strength. The recommended setup as in the paper is to use a positive
      number.
    noise_frequency: "p" in the paper. Determines the probability of the noisy
      evaluation. Clipped (not explicitly but effectively) to [0, 1] range.
    rng:

  Returns:
    Noisy version of value.
  """
  noise = (rng.uniform() < noise_frequency) * rng.standard_cauchy()
  return value + noise_strength * np.max([0.0, 1000.0 + noise])


def _stabilized_noise(value: float,
                      noisy_fn: Callable[[float], float],
                      target_value: float = 1e-8) -> float:
  """Post processing of noise for bbob-noisy benchmark.

  We do not apply noise if the value is close to the global optima. This keeps
  the optimal value intact.


  Args:
    value: Function value to apply noise to.
    noisy_fn: "f_XX" in the paper. It applies noise to the input.
    target_value: If value is less than this number, then we do not apply the
      noise.

  Returns:
    value, if it is less than target_value. Otherwise, noisy version of
    value.
  """

  if value >= target_value:
    return noisy_fn(value) + 1.01 * target_value
  else:
    return value
