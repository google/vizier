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
  benchmark_state = benchmark_runner.BenchmarkState.from_designer_factory(
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
from typing import Callable, Optional, Protocol, Sequence

from absl import logging
import attr
from vizier import algorithms as vza
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters.experimenter import Experimenter
from vizier._src.benchmarks.runners import runner_protocol

DesignerFactory = Callable[[vz.ProblemStatement], vza.Designer]
PolicyFactory = Callable[[vz.ProblemStatement], pythia.Policy]


@attr.define
class BenchmarkState:
  """State of a benchmark run. It is altered via benchmark protocols."""

  experimenter: Experimenter
  algorithm: runner_protocol.AlgorithmRunnerProtocol


@attr.define(frozen=True)
class BenchmarkStateFactory(abc.ABC):
  """Factory class to generate new BenchmarkState."""

  experimenter: Experimenter

  @abc.abstractmethod
  def create(self) -> BenchmarkState:
    """Creates a new instance of BenchmarkFactory."""
    pass


@attr.define(frozen=True)
class DesignerBenchmarkStateFactory(BenchmarkStateFactory):
  """Factory class to generate new BenchmarkState from designer factory."""

  designer_factory: DesignerFactory

  def create(self) -> BenchmarkState:
    """Create a BenchmarkState from designer factory."""
    problem = self.experimenter.problem_statement()
    return BenchmarkState(
        experimenter=self.experimenter,
        algorithm=runner_protocol.DesignerRunnerProtocol(
            designer=self.designer_factory(problem),
            local_supporter=pythia.InRamPolicySupporter(problem)))


@attr.define(frozen=True)
class PolicyBenchmarkStateFactory(BenchmarkStateFactory):

  policy_factory: PolicyFactory

  def create(self) -> BenchmarkState:
    """Create a BenchmarkState from policy factory."""
    problem = self.experimenter.problem_statement()
    return BenchmarkState(
        experimenter=self.experimenter,
        algorithm=runner_protocol.PolicyRunnerProtocol(
            policy=self.policy_factory(problem),
            local_supporter=pythia.InRamPolicySupporter(problem)))


class BenchmarkSubroutine(Protocol):
  """Abstraction for core benchmark routines.

  Benchmark protocols are modular alterations of BenchmarkState by reference.
  """

  def run(self, state: BenchmarkState) -> None:
    """Abstraction to alter BenchmarkState by reference."""


@attr.define
class GenerateAndEvaluate(BenchmarkSubroutine):
  """Generate a fixed number of Suggestions and Evaluate them immediately.

  Use this Subroutine for simple benchmarks as it is computationally efficient.
  For benchmarks with complicated evaluation procedures, use a combination
  of other Subroutines.
  """

  # Number of total suggestions as a batch.
  num_suggestions: int = attr.field(
      default=1, validator=attr.validators.instance_of(int))

  def run(self, state: BenchmarkState) -> None:
    suggestions = state.algorithm.suggest(self.num_suggestions)
    if not suggestions:
      logging.info(
          'Algorithm did not generate %d suggestions'
          'because it returned nothing.', self.num_suggestions)
    state.experimenter.evaluate(list(suggestions))
    # Only needed for Designers.
    state.algorithm.post_completion_callback(vza.CompletedTrials(suggestions))


@attr.define
class GenerateSuggestions(BenchmarkSubroutine):
  """Generate a fixed number of Suggestions as Active Trials."""

  # Number of total suggestions as a batch.
  num_suggestions: int = attr.field(
      default=1, validator=attr.validators.instance_of(int))

  def run(self, state: BenchmarkState) -> None:
    suggestions = state.algorithm.suggest(self.num_suggestions)
    if not suggestions:
      logging.info(
          'Suggestions did not generate %d suggestions'
          'because designer returned nothing.', self.num_suggestions)


@attr.define
class EvaluateActiveTrials(BenchmarkSubroutine):
  """Evaluate a fixed number of Active Trials as Completed Trials."""

  # If None, this Evaluates all Pending Trials.
  num_evaluations: Optional[int] = attr.field(default=None)

  def run(self, state: BenchmarkState) -> None:
    active_trials = state.algorithm.supporter.GetTrials(
        status_matches=vz.TrialStatus.ACTIVE)

    if self.num_evaluations is None:
      evaluated_trials = active_trials
    else:
      evaluated_trials = active_trials[:self.num_evaluations]

    state.experimenter.evaluate(evaluated_trials)
    # Only needed for Designers.
    state.algorithm.post_completion_callback(
        vza.CompletedTrials(evaluated_trials))


@attr.define
class BenchmarkRunner(BenchmarkSubroutine):
  """Run a sequence of subroutines, all repeated for a few iterations."""

  # A sequence of benchmark subroutines that alter BenchmarkState.
  benchmark_subroutines: Sequence[BenchmarkSubroutine] = attr.field()
  # Number of times to repeat applying benchmark_subroutines.
  num_repeats: int = attr.field(
      default=1, validator=attr.validators.instance_of(int))

  def run(self, state: BenchmarkState) -> None:
    """Run algorithm with benchmark subroutines with repetitions."""
    for _ in range(self.num_repeats):
      for subroutine in self.benchmark_subroutines:
        subroutine.run(state)
