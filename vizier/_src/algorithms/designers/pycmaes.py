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

"""CMA-ES designer using pycma https://github.com/CMA-ES/pycma."""

from typing import Optional, Sequence

import cma
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier.pyvizier import converters


class PyCMAESDesigner(vza.Designer):
  """CMA-ES designer wrapping pycma."""

  def __init__(
      self,
      problem_statement: vz.ProblemStatement,
      sigma0: float = 0.1,
      popsize: int | None = None,
  ):
    """Init.

    Args:
      problem_statement: Must use a flat DOUBLE-only search space.
      sigma0: The initial standard deviation of the CMA-ES algorithm.
      popsize: The size of the population to use in each CMA-ES update. If None,
        the default popsize is used.
    """
    self._problem_statement = problem_statement
    self._metric_name = self._problem_statement.metric_information.item().name

    if popsize is not None and popsize < 2:
      raise ValueError(f'Popsize must be at least 2. Got {popsize}.')

    self._search_space = self._problem_statement.search_space
    if self._search_space.is_conditional:
      raise ValueError(
          f'This designer {self} does not support conditional search.')
    elif len(self._problem_statement.metric_information) != 1:
      raise ValueError(f'{type(self)} works with exactly one metric.')

    init_params = vz.ParameterDict()
    for parameter_config in self._search_space.parameters:
      if not parameter_config.type.is_continuous():
        raise ValueError(
            f'This designer {self} only supports continuous parameters.')
      if parameter_config.default_value is not None:
        init_params[parameter_config.name] = parameter_config.default_value
      elif parameter_config.bounds is not None:
        init_params[parameter_config.name] = (
            parameter_config.bounds[0] + parameter_config.bounds[1]
        ) / 2.0
      else:
        raise ValueError(
            f'The continuous parameter: {parameter_config.name} is missing'
            ' bounds.'
        )

    self._converter = converters.TrialToArrayConverter.from_study_config(
        self._problem_statement,
        scale=True,
        flip_sign_for_minimization_metrics=True,
    )
    self._x0 = self._converter.to_features([vz.TrialSuggestion(init_params)])[0]
    self._sigma0 = sigma0
    self._popsize = popsize
    self._all_completed_trials: list[vz.Trial] = []

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    self._all_completed_trials.extend(completed.trials)

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    """Make new suggestions.

    Args:
      count: Makes best effort to generate this many suggestions. If None,
        suggests as many as the algorithm wants.

    Returns:
      New suggestions.
    """
    count = count or 1
    num_completed_trials = len(self._all_completed_trials)
    # The trial converter scales the parameters to [0, 1] range.
    if self._popsize is None:
      options = {'bounds': [0.0, 1.0]}
    else:
      options = {'bounds': [0.0, 1.0], 'popsize': self._popsize}
    cma_evolution = cma.CMAEvolutionStrategy(
        self._x0,
        self._sigma0,
        options,
    )
    # Ensures that the number of completed trials fed to CMA-ES is a multiple
    # of the popsize as required.
    feed_size = int(
        (num_completed_trials // cma_evolution.popsize) * cma_evolution.popsize
    )
    if feed_size > 0:
      features, labels = self._converter.to_xy(
          self._all_completed_trials[-feed_size:]
      )
      # CMA-ES expects a minimization problem by default, but the converter
      # outputs maximization metrics, so we sign-flip the converted labels.
      cma_evolution.feed_for_resume(features, -labels)
    cma_suggestions = np.array(cma_evolution.ask(count))
    return [
        vz.TrialSuggestion(params)
        for params in self._converter.to_parameters(cma_suggestions)
    ]
