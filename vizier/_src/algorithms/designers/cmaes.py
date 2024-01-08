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

"""CMA-ES designer."""
import json
import queue
from typing import Optional, Sequence

from evojax.algo import cma_jax
import jax.numpy as jnp
import numpy as np

from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier.pyvizier import converters
from vizier.utils import json_utils


class CMAESDesigner(vza.PartiallySerializableDesigner):
  """CMA-ES designer wrapping evo-jax.

  NOTE: Since the base version of CMA-ES expects the entire population size to
  be evaluated before an update, we must use temporary queues to hold partially
  finished populations.
  """

  def __init__(self, problem_statement: vz.ProblemStatement, **cma_kwargs):
    """Init.

    Args:
      problem_statement: Must use a flat DOUBLE-only search space.
      **cma_kwargs: Keyword arguments for the CMA_ES_JAX class.
    """
    self._problem_statement = problem_statement
    self._metric_name = self._problem_statement.metric_information.item().name

    self._search_space = self._problem_statement.search_space
    if self._search_space.is_conditional:
      raise ValueError(
          f'This designer {self} does not support conditional search.')
    for parameter_config in self._search_space.parameters:
      if not parameter_config.type.is_continuous():
        raise ValueError(
            f'This designer {self} only supports continuous parameters.')
    self._num_params = len(self._search_space.parameters)
    if self._num_params < 2:
      raise ValueError(
          'CMA-ES only supports search spaces with >=2 parameters. Current'
          f' number of parameters: {self._num_params}'
      )

    # CMA-ES expects a maximization problem by default, so we flip signs for
    # minimization metrics.
    self._converter = converters.TrialToArrayConverter.from_study_config(
        self._problem_statement,
        scale=True,
        flip_sign_for_minimization_metrics=True,
    )
    self._cma_es_jax = cma_jax.CMA_ES_JAX(
        param_size=self._num_params, **cma_kwargs)
    self._trial_population = queue.Queue(
        maxsize=self._cma_es_jax.hyper_parameters.pop_size)

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    completed_trials = list(completed.trials)

    # Keep inserting completed trials into population. If population is full,
    # a CMA-ES update and queue clear are triggered.
    while completed_trials:
      self._trial_population.put(completed_trials.pop())

      if self._trial_population.full():
        # Once full, make a full CMA-ES update.
        features, labels = self._converter.to_xy(
            list(self._trial_population.queue))
        # CMA-ES expects fitness to be shape (pop_size,) and solutions of shape
        # (pop_size, num_params).
        self._cma_es_jax.tell(
            fitness=jnp.array(labels[:, 0]), solutions=jnp.array(features))
        self._trial_population.queue.clear()

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
    cma_suggestions = np.array(self._cma_es_jax.ask(count))

    # Convert CMA suggestions to suggestions.
    return [
        vz.TrialSuggestion(params)
        for params in self._converter.to_parameters(cma_suggestions)
    ]

  def load(self, metadata: vz.Metadata) -> None:
    cma_state = json.loads(
        metadata.ns('cma')['state'], object_hook=json_utils.numpy_hook)
    self._cma_es_jax.load_state(cma_state)

  def dump(self) -> vz.Metadata:
    cma_state = self._cma_es_jax.save_state()
    metadata = vz.Metadata()
    metadata.ns('cma')['state'] = json.dumps(
        cma_state, cls=json_utils.NumpyEncoder)
    return metadata
