"""Convergence test for Pythia policies and deisngers.

The convergence test uses the BenchmarkRunner and essentially runs multiple
loops of Suggest-Evaluate to gauge if the suggested values have converged to
the expected optimum value. The decision of whether the convergence test has
passed or failed is determined by the percentage of converged individual
checks. To ensure that each check is isolated from other, the ConvergenceRunner
uses BenchmarkStateFactory to generate a fresh BenchmarkState for each check.

Ex: Typical Convergence Test on Designer
----------------------------------------
class MyDesignerConvegenceTest(absltest.TestCase):

  # Test function assuming experimenter and designer factory.
  def test_convergence(self):

    # Create a benchmark state factory based on a designer factory
    benchmark_state_factory = benchmarks.DesignerBenchmarkStateFactory(
        designer_factory=designer_factory,
        experimenter=experimenter,
    )

    # Define the convergence test.
    convergence_test = convergence_runner.BenchmarkConvergenceRunner(
      benchmark_state_factory=benchmark_state_factory,
      trials_per_check=5000,
      repeated_checks=5,
      success_rate_threshold=0.6,
      tolerance=1.0)

    # Run the convergence test which will raise exception if fail.
    convergence_test.assert_converges()
"""

import enum
import logging
from typing import Tuple

import attr
from vizier import benchmarks
from vizier import pyvizier as vz


class ConvergeStatus(enum.Enum):
  FAILED = 'FAILED'
  PASSED = 'PASSEd'


class FailedConvergenceTestError(Exception):
  """Exception raised for convergence test fails."""


def _generate_convergence_report(
    test_status: ConvergeStatus,
    trials_per_check: int,
    repeated_checks: int,
    success_rate: float,
    success_rate_threshold: float,
    best_metrics: list[float],
    best_trials: list[vz.Trial],
    check_statuses: list[ConvergeStatus],
    tolerance: float,
    experimenter: benchmarks.Experimenter,
) -> str:
  """Generates a convergence report."""
  message = (
      f'Convergence test {test_status}.\nPerformed {repeated_checks} repeated '
      f'convergence checks with {trials_per_check} trials each.'
      f'\nSuccess rate : {success_rate}. Success rate threshold: '
      f'{success_rate_threshold}. Tolerance: {tolerance}. Experimenter: '
      f'{str(experimenter)}')

  for idx in range(repeated_checks):
    trimmed_parameters = {
        k: f'{v:.3f}' for k, v in best_trials[idx].parameters.as_dict().items()
    }
    test_desc = (
        f'\nConvergence Check {idx} - {check_statuses[idx]}. Best trial '
        f'metric: {best_metrics[idx]:.6f} .Best trial parameters: '
        f'{trimmed_parameters}.')
    message = message + test_desc
  return message


@attr.define
class BenchmarkConvergenceRunner:
  """Convergence test for designers.

  Important note: the convergence test currently only support float parameters.
  """
  benchmark_state_factory: benchmarks.BenchmarkStateFactory
  trials_per_check: int
  repeated_checks: int
  success_rate_threshold: float
  tolerance: float

  def assert_converges(self) -> None:
    """Runs the full convergence test.

    Performs the convergence test `repeated_checks` times and raises
    `FailedConvergenceTestError` if the percentage of converged test is below
    `success_rate_threshold`, otherwise logs the convergence results.

    Raises:
      FailedConvergenceTestError: if the convergence test failed.
    """
    num_converged_checks = 0
    best_metrics, best_trials, convergence_check_statuses = [], [], []

    for _ in range(self.repeated_checks):
      check_status, best_metric, best_trial = self._run_one_convergence_check()
      if check_status == ConvergeStatus.PASSED:
        num_converged_checks += 1
      best_metrics.append(best_metric)
      best_trials.append(best_trial)
      convergence_check_statuses.append(check_status)

    success_rate = num_converged_checks / self.repeated_checks
    if success_rate >= self.success_rate_threshold:
      convergence_test_status = ConvergeStatus.PASSED
    else:
      convergence_test_status = ConvergeStatus.FAILED

    message = _generate_convergence_report(
        test_status=convergence_test_status,
        trials_per_check=self.trials_per_check,
        repeated_checks=self.repeated_checks,
        success_rate=success_rate,
        success_rate_threshold=self.success_rate_threshold,
        best_metrics=best_metrics,
        best_trials=best_trials,
        check_statuses=convergence_check_statuses,
        tolerance=self.tolerance,
        experimenter=self.benchmark_state_factory.experimenter)

    if success_rate < self.success_rate_threshold:
      raise FailedConvergenceTestError(message)
    else:
      logging.info(message)

  def _run_one_convergence_check(
      self) -> Tuple[ConvergeStatus, float, vz.Trial]:
    """Runs a single convergence test.

    Returns:
      1. The convergence status (FAILED or PASSED).
      2. The value of the best metric found during the benchmark run check.
      3. The best trial (associated with 2.) found during the benchmark check.
    """
    benchmark_state = self.benchmark_state_factory.create()

    runner = benchmarks.BenchmarkRunner(
        benchmark_subroutines=[benchmarks.GenerateAndEvaluate()],
        num_repeats=self.trials_per_check)

    runner.run(benchmark_state)
    best_metric, best_trial = self._get_best_metric_and_trial(benchmark_state)

    # TODO: 1. Determine what is the "ideal" convergence criteria.
    # TODO: 2. Should we get the optimum value from the experimenter
    # to compare against? (currently it assumes 0.0).
    if best_metric < self.tolerance:
      check_status = ConvergeStatus.PASSED
    else:
      check_status = ConvergeStatus.FAILED
    return check_status, best_metric, best_trial

  def _get_best_metric_and_trial(
      self,
      benchmark_state: benchmarks.BenchmarkState) -> Tuple[float, vz.Trial]:
    """Returns the best trial and metric value after the benchmark has run."""
    best_trial = benchmark_state.algorithm.supporter.GetBestTrials(count=1)[0]
    metric_name = self.benchmark_state_factory.experimenter.problem_statement(
    ).single_objective_metric_name
    best_metric = best_trial.final_measurement.metrics[metric_name].value
    return best_metric, best_trial
