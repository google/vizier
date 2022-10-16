# Copyright 2022 Google LLC.
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

"""Simple categorical experimneter."""

from typing import Optional, Sequence, List

import numpy as np
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter


class CategoricalExperimenter(experimenter.Experimenter):
  """Categorical experimenter associated minimum optimization problem."""

  def __init__(
      self,
      categorical_dim: int,
      num_categories: int,
      optimum: Optional[List[str]] = None,
      seed: int = 0,
  ):
    """Constructor.

    Arguments:
      categorical_dim: The number of categorical parameters.
      num_categories: The number of categories in each categorical parameter.
      optimum: Optional list of indices indicating the optimum point.
      seed: Optional random generator seed.
    """
    rng = np.random.default_rng(seed=seed)
    self._problem = vz.ProblemStatement()
    if optimum is not None:
      if len(optimum) != categorical_dim:
        raise ValueError('optimum length is different than categorical_dim!')
    self._optimum = {}

    for i in range(categorical_dim):
      categories = [str(x) for x in range(num_categories)]
      self._problem.search_space.root.add_categorical_param(f'c{i}', categories)
      if optimum is None:
        self._optimum[f'c{i}'] = str(rng.integers(low=0, high=num_categories))
      else:
        self._optimum[f'c{i}'] = optimum[i]

    self._problem.metric_information.append(
        vz.MetricInformation(
            name='objective', goal=vz.ObjectiveMetricGoal.MINIMIZE))

  def evaluate(self, suggestions: Sequence[vz.Trial]):
    for suggestion in suggestions:
      loss = 0
      for param_config in self._problem.search_space.parameters:
        if suggestion.parameters[param_config.name].value != self._optimum[
            param_config.name]:
          loss += 1
      suggestion.complete(vz.Measurement(metrics={'objective': loss}))

  @property
  def optimum_trial(self) -> vz.Trial:
    print(self._optimum)
    return vz.Trial(parameters=self._optimum)

  def problem_statement(self):
    return self._problem
