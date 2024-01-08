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

"""Wraps Designer as a gradient-free optimizer."""

from typing import Callable, List, TypeVar, Sequence

from absl import logging
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.algorithms.core import abstractions
from vizier._src.algorithms.optimizers import base

_Features = TypeVar('_Features')


class DesignerAsOptimizer(base.GradientFreeOptimizer):
  """Wraps a Designer into GradientFreeOptimizer."""

  def __init__(self,
               designer_factory: Callable[[vz.ProblemStatement],
                                          abstractions.Designer],
               *,
               batch_size: int = 100,
               num_evaluations: int = 15000):
    """Init.

    Args:
      designer_factory:
      batch_size: In each iteration, ask the designer to generate this many
        candidate trials.
      num_evaluations: Total number of trials to be evaluated on `score_fn`.
    """
    self._designer_factory = designer_factory
    self._batch_size = batch_size
    self._num_evaluations = num_evaluations

  def optimize(
      self,
      score_fn: base.BatchTrialScoreFunction,
      problem: vz.ProblemStatement,
      *,
      count: int = 1,
      budget_factor: float = 1.0,
      seed_candidates: Sequence[vz.TrialSuggestion] = tuple(),
  ) -> List[vz.Trial]:
    # Use the in-ram supporter as a pseudo-client for running a study in RAM.
    study = pythia.InRamPolicySupporter(problem)

    # Save the designer for debugging purposes only.
    self._designer = self._designer_factory(problem)
    num_iterations = max(
        int(self._num_evaluations * budget_factor) // self._batch_size, 1)
    logging.info(
        'Optimizing the acquisition for %s iterations of %s trials each',
        num_iterations, self._batch_size)

    for _ in range(num_iterations):
      trials = study.AddSuggestions(self._designer.suggest(self._batch_size))
      if not trials:
        break
      scores = score_fn(trials)
      # Check that scores are (N, 1) arrays as in BatchTrialScoreFunction.
      for k, v in scores.items():
        if v.shape != (len(trials), 1):
          raise ValueError(
              f'Incorrect shape {v.shape} in scores {scores[k]}\n'
              f'Expected shape is {(len(trials), 1)}'
          )
      for i, trial in enumerate(trials):
        # TODO: Decide what to do with NaNs scores.
        trial.complete(
            vz.Measurement({k: v[i].item() for k, v in scores.items()}))
      self._designer.update(
          abstractions.CompletedTrials(trials), abstractions.ActiveTrials()
      )
    logging.info(
        'Finished running the optimization study. Extracting the best trials...'
    )
    return study.GetBestTrials(count=count)
