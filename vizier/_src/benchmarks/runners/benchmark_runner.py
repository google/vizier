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

"""Benchmark runners to run benchmarks via modularized subroutines.

Modularization provides flexibility and clarity. Also, this allows easy
configuration of the protocol routine, which then can be applied to multiple
benchmarks via runner.run().

Ex: Typical Suggest + Evaluate loop all repeated for 7 iterations.

  # Define protocol (i.e. what should the benchmark do?).
  runner = benchmark_runner.BenchmarkRunner(
          benchmark_subroutines=[
              benchmark_runner.GenerateSuggestions(),
              benchmark_runner.EvaluateActiveTrials()
          ],
          num_repeats=7)

  # Initialize state (i.e. what problem and algorithm are we using?).
  benchmark_state =
  benchmark_runner.benchmark_state.BenchmarkState.from_designer_factory(
          designer_factory=designer_factory, experimenter=experimenter)

  # Runs benchmark protocol(s) and updates state.
  runner.run(benchmark_state)

Ex2: The power of modularization allows for complicated subroutines, such as
one that simulates the stochasticity of natural evaluation processes.

  # Allow for fixed-size batched Suggests and random-size evaluations.
  subroutines = []
  for _ in range(10):
    subroutines.append(benchmark_runner.GenerateSuggestions(num_suggestions=3)
    subroutines.append(benchmark_runner.EvaluateActiveTrial(
        num_trials_to_evaluate = np.randomInt(0, 10))
  runner = benchmark_runner.BenchmarkRunner(benchmark_subroutines=subroutines)
"""

import abc
import time
from typing import Optional, Sequence

from absl import logging
import attr
from vizier import pyvizier as vz
from vizier._src.benchmarks.runners import benchmark_state


class BenchmarkSubroutine(abc.ABC):
  """Abstraction for core benchmark routines.

  Benchmark protocols are modular alterations of BenchmarkState by reference.
  """

  @abc.abstractmethod
  def run(self, state: benchmark_state.BenchmarkState) -> None:
    """Abstraction to alter BenchmarkState by reference."""


@attr.define
class GenerateAndEvaluate(BenchmarkSubroutine):
  """Generate a fixed number of Suggestions and Evaluate them immediately.

  Use this Subroutine for simple benchmarks as it is computationally efficient.
  For benchmarks with complicated evaluation procedures, use a combination
  of other Subroutines.
  """

  # Number of total suggestions as a batch.
  batch_size: int = attr.field(
      default=1, validator=attr.validators.instance_of(int)
  )

  def run(self, state: benchmark_state.BenchmarkState) -> None:
    suggestions = state.algorithm.suggest(self.batch_size)
    if not suggestions:
      logging.info(
          'Algorithm returned 0 suggestions. Expected: %s.',
          self.batch_size,
      )
    logging.info('Generated %s suggestions.', len(suggestions))
    state.experimenter.evaluate(list(suggestions))
    for t in suggestions:
      logging.info('Trial %s: %s', t.id, t.final_measurement)


@attr.define
class GenerateSuggestions(BenchmarkSubroutine):
  """Generate a fixed number of Suggestions as Active Trials."""

  # Number of total suggestions as a batch.
  batch_size: int = attr.field(
      default=1, validator=attr.validators.instance_of(int)
  )

  def run(self, state: benchmark_state.BenchmarkState) -> None:
    suggestions = state.algorithm.suggest(self.batch_size)
    if not suggestions:
      logging.info(
          (
              'Suggestions did not generate %d suggestions'
              'because designer returned nothing.'
          ),
          self.batch_size,
      )


@attr.define
class FillActiveTrials(BenchmarkSubroutine):
  """Generate a number of Suggestions up to a limit of Active Trials."""

  # Upper limit of active Trials to be filled with new Suggestions.
  num_active_trials_limit: int = attr.field(
      default=1, validator=attr.validators.instance_of(int)
  )

  def run(self, state: benchmark_state.BenchmarkState) -> None:
    active_trials = state.algorithm.supporter.GetTrials(
        status_matches=vz.TrialStatus.ACTIVE
    )
    if (
        num_suggestions := self.num_active_trials_limit - len(active_trials)
    ) > 0:
      suggestions = state.algorithm.suggest(num_suggestions)
      if not suggestions:
        logging.info(
            (
                'Suggestions did not generate %d suggestions'
                'to fill active Trials to %d '
                'because designer returned nothing.'
            ),
            num_suggestions,
            self.num_active_trials_limit,
        )


@attr.define
class EvaluateActiveTrials(BenchmarkSubroutine):
  """Evaluate a fixed number of Active Trials as Completed Trials."""

  # If None, this Evaluates all Active Trials.
  num_evaluations: Optional[int] = attr.field(default=None)

  def run(self, state: benchmark_state.BenchmarkState) -> None:
    active_trials = state.algorithm.supporter.GetTrials(
        status_matches=vz.TrialStatus.ACTIVE
    )
    if self.num_evaluations is None:
      evaluated_trials = active_trials
    else:
      evaluated_trials = active_trials[: self.num_evaluations]

    state.experimenter.evaluate(evaluated_trials)
    logging.info('Evaluated %s trials.', len(evaluated_trials))
    for t in evaluated_trials:
      logging.info('Trial %s: %s', t.id, t.final_measurement)


@attr.define
class EvaluateAndAddPriorStudy(BenchmarkSubroutine):
  """Evaluate a fixed number of Active Trials as Completed Trials."""

  # Runner to use to mutate study.
  benchmark_runner: BenchmarkSubroutine = attr.field(
      kw_only=True,
      validator=attr.validators.instance_of(BenchmarkSubroutine),
  )
  # State factory.
  benchmark_state_factory: benchmark_state.BenchmarkStateFactory = attr.field(
      kw_only=True,
      validator=attr.validators.instance_of(
          benchmark_state.BenchmarkStateFactory
      ),
  )
  study_guid: Optional[str] = attr.field(
      kw_only=True,
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(str)),
  )
  seed: Optional[int] = attr.field(
      kw_only=True,
      default=None,
      validator=attr.validators.optional(attr.validators.instance_of(int)),
  )

  def run(self, state: benchmark_state.BenchmarkState) -> None:
    prior_state = self.benchmark_state_factory(seed=self.seed)
    self.benchmark_runner.run(prior_state)
    prior_problem = prior_state.algorithm.supporter.GetStudyConfig()
    prior_trials = prior_state.algorithm.supporter.GetTrials()
    prior_study = vz.ProblemAndTrials(
        problem=prior_problem, trials=prior_trials
    )

    state.algorithm.supporter.SetPriorStudy(
        study=prior_study, study_guid=self.study_guid
    )


@attr.define
class BenchmarkRunner(BenchmarkSubroutine):
  """Run a sequence of subroutines, all repeated for a few iterations."""

  # A sequence of benchmark subroutines that alter BenchmarkState.
  benchmark_subroutines: Sequence[BenchmarkSubroutine] = attr.field()
  # Number of times to repeat applying benchmark_subroutines.
  num_repeats: int = attr.field(
      default=1, validator=attr.validators.instance_of(int)
  )

  def run(self, state: benchmark_state.BenchmarkState) -> None:
    """Run algorithm with benchmark subroutines with repetitions."""
    for iteration in range(self.num_repeats):
      start = time.time()
      for subroutine in self.benchmark_subroutines:
        subroutine.run(state)
      total_time = time.time() - start
      logging.info(
          'Completed BenchmarkRunner iteration %d out of %d (%f seconds)',
          iteration,
          self.num_repeats,
          total_time,
      )
