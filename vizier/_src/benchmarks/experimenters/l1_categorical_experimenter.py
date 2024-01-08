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

"""L1 categorical experimenter.

The experimenter's evaluation function counts the number of different parameters
values between the optimal point ('optimum') and the suggestion trial.
"""

import copy
import logging
from typing import Optional, Sequence

import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter


class L1CategorialExperimenter(experimenter.Experimenter):
  """Categorical experimenter associated minimum optimization problem."""

  def __init__(
      self,
      *,
      num_categories: Sequence[int],
      optimum: Optional[Sequence[int]] = None,
      seed: Optional[int] = None,
      verbose: bool = False,
  ):
    """Constructor.

    If optimum is not provided, a randomly generated optimum point will be
    created.

    Arguments:
      num_categories: The number of categories in each parameter.
      optimum: Optional list of indices indicating the optimum point. If not
        set, randomly created from seed.
      seed: Optional random generator seed.
      verbose: Whether to show logs.
    """
    rng = np.random.default_rng(seed=seed)
    self._problem = vz.ProblemStatement()
    self._optimum = {}
    for i, num_category in enumerate(num_categories):
      categories = [str(x) for x in range(num_category)]
      param_name = f'c{i}'
      self._problem.search_space.root.add_categorical_param(
          param_name, categories)
      if optimum is None:
        # Define the optimum point by randomally selecting categorical values.
        self._optimum[param_name] = str(rng.integers(low=0, high=num_category))
      elif optimum[i] >= num_category:
        raise ValueError("Optimum point doesn't match category dimensions!")
      else:
        self._optimum[param_name] = str(optimum[i])

    self._problem.metric_information.append(
        vz.MetricInformation(
            name='objective', goal=vz.ObjectiveMetricGoal.MINIMIZE))
    if verbose:
      logging.info('L1CategoricalExperimenter optimum point: %s', self._optimum)

  def evaluate(self, suggestions: Sequence[vz.Trial]):
    for suggestion in suggestions:
      loss = 0.0
      for param_config in self._problem.search_space.parameters:
        if suggestion.parameters[param_config.name].value != self._optimum[
            param_config.name]:
          loss += 1.0
      suggestion.complete(vz.Measurement(metrics={'objective': loss}))

  @property
  def optimal_trial(self) -> vz.Trial:
    """Evaluates and returns the optimal trial."""
    optimal_trial = vz.Trial(parameters=self._optimum)
    self.evaluate([optimal_trial])
    return optimal_trial

  def problem_statement(self):
    """See superclass."""
    return copy.deepcopy(self._problem)
