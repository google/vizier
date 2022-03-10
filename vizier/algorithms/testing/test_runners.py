"""Test runners for algorithms."""
from typing import Optional, Sequence, Any

from absl import logging
import numpy as np

from vizier.algorithms import core as vza

from vizier.pyvizier import pythia as vz


def run_with_random_metrics(
    designer: vza.Designer,
    config: vz.StudyConfig,
    iters: int = 5,
    *,
    batch_size: Optional[int] = 1,
    seed: Any = None,
    verbose: int = 0,
    validate_parameters: bool = False) -> Sequence[vz.Trial]:
  """Generate completed trials with random metrics."""
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
