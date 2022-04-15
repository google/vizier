"""Test runners for algorithms."""
from typing import Any, Optional, Sequence

from absl import logging
import numpy as np

from vizier import algorithms as vza

from vizier import pyvizier as vz


def run_with_random_metrics(
    designer: vza.Designer,
    config: vz.StudyConfig,
    iters: int = 5,
    *,
    batch_size: Optional[int] = 1,
    seed: Any = None,
    verbose: int = 0,
    validate_parameters: bool = False) -> Sequence[vz.Trial]:
  """Generates completed `Trial`s with random metrics.

  EXAMPLE: This method can be used for smoke testing a `Designer`.
  ```
  from vizier._src.algorithms.testing import test_runners
  from vizier.testing import test_studies
  from vizier import pyvizier as vz

  study_config = vz.StudyConfig(
      test_studies.flat_space_with_all_types(),
      [vz.MetricInformation(
          'objective',
          goal=vz.ObjectiveMetricGoal.MAXIMIZE)],
      validate_parameters=True)
  test_runners.run_with_random_metrics(my_designer, study_config)
  ```

  EXAMPLE: This method can be used for generating a large number of trials
  to be used as a test dataset.
  ```
  # (Continued from the above code block)
  from vizier._src.algorithms.designers import random
  trials = test_runners.run_with_random_metrics(
      random.RandomDesigner(study_config.search_space),
      study_config,
      validate_parameters=False)
  ```

  Args:
    designer: Designer object.
    config: Study config.
    iters: Number of suggest-update iterations.
    batch_size: Number of suggestions to ask in each suggest() call.
    seed: Random seed for generating metrics.
    verbose: Increase the verbosity to see more logs.
    validate_parameters: If True, check if the suggested trials are valid in the
      search space.

  Returns:
    This method runs a suggest-update loop after completing suggestions with
    random metric values, and then returns all generated trials.
  """
  rng = np.random.RandomState(seed)
  all_trials = []
  for it in range(iters):
    suggestions = designer.suggest(batch_size)
    if not suggestions:
      logging.info(
          'Preemptively finished at iteration %s'
          'because designer returned nothing.', it)
      break
    trials = []
    for suggestion in designer.suggest(batch_size):
      if validate_parameters:
        config.search_space.assert_contains(suggestion.parameters)
      measurement = vz.Measurement()
      for mi in config.metric_information:
        measurement.metrics[mi.name] = rng.uniform(
            mi.min_value_or(lambda: -10.), mi.max_value_or(lambda: 10.))
      trials.append(suggestion.to_trial(len(trials) + 1).complete(measurement))
      if verbose:
        logging.info('At iteration %s, trials suggested and evaluated:\n%s', it,
                     trials)
      designer.update(vza.CompletedTrials(trials))
    all_trials.extend(trials)

  return all_trials
