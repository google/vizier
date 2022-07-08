"""Example of a Vizier Client, which can be run on multiple machines.

This is meant to be used after the Vizier Server (see `run_vizier_server.py`)
has been launched and provided an address to connect to. Example of a launch
command:

```
python run_vizier_client.py --address="localhost:[PORT]"
```

where `address` was provided by the server.
"""

from typing import Sequence

from absl import app
from absl import flags
from absl import logging

from vizier.service import pyvizier as vz
from vizier.service import vizier_client

flags.DEFINE_string(
    'address', '',
    "Address of the Vizier Server which will be used by this demo. Should be of the form e.g. 'localhost:6006' if running on the same machine, or `[IP]:[PORT]` if running on a remote machine."
)
flags.DEFINE_integer(
    'max_num_iterations', 10,
    'Maximum number of possible iterations / calls to get suggestions.')
flags.DEFINE_integer(
    'suggestion_count', 5,
    'Number of suggestions to evaluate per iteration. Useful for batched evaluations.'
)
flags.DEFINE_boolean(
    'multiobjective', True,
    'Whether to demonstrate multiobjective or single-objective capabilities and API.'
)

FLAGS = flags.FLAGS


def evaluate_trial(trial: vz.Trial) -> vz.Measurement:
  """Dummy evaluator used as an example."""
  learning_rate = trial.parameters.get_value('learning_rate')
  num_layers = trial.parameters.get_value('num_layers')
  m = vz.Measurement()
  m.metrics = {'accuracy': learning_rate * num_layers}  # dummy accuracy
  if FLAGS.multiobjective:
    m.metrics['latency'] = 0.5 * num_layers
  return m


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not FLAGS.address:
    logging.info(
        'You did not specify the server address. Please see the documentation on the `address` FLAGS.'
    )

  study_config = vz.StudyConfig()  # Search space, metrics, and algorithm.
  root = study_config.search_space.select_root()
  root.add_float_param(
      'learning_rate',
      min_value=1e-4,
      max_value=1e-2,
      scale_type=vz.ScaleType.LOG)
  root.add_int_param('num_layers', min_value=1, max_value=5)
  study_config.metric_information.append(
      vz.MetricInformation(
          name='accuracy',
          goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          min_value=0.0,
          max_value=1.0))

  if FLAGS.multiobjective:
    # No need to specify min/max values.
    study_config.metric_information.append(
        vz.MetricInformation(
            name='latency', goal=vz.ObjectiveMetricGoal.MINIMIZE))

  if FLAGS.multiobjective:
    study_config.algorithm = vz.Algorithm.RANDOM_SEARCH
  else:
    study_config.algorithm = vz.Algorithm.EMUKIT_GP_EI

  client = vizier_client.create_or_load_study(
      service_endpoint=FLAGS.address,
      owner_id='my_name',
      client_id='my_client_id',
      study_id='cifar10',
      study_config=study_config)
  logging.info('Client created with study name: %s', client.study_resource_name)

  for _ in range(FLAGS.max_num_iterations):
    # Evaluate the suggestion(s) and report the results to Vizier.
    suggestions = client.get_suggestions(
        suggestion_count=FLAGS.suggestion_count)
    for trial in suggestions:
      measurement = evaluate_trial(trial)
      client.complete_trial(trial_id=trial.id, final_measurement=measurement)
      logging.info('Trial %d completed with metrics: %s', trial.id,
                   measurement.metrics)

  all_trials = client.list_trials()
  logging.info('%d trials have been completed.', len(all_trials))

  optimal_trials = client.list_optimal_trials()
  for optimal_trial in optimal_trials:
    logging.info(
        'Pareto-optimal trial found so far has parameters %s and metrics %s',
        optimal_trial.parameters, optimal_trial.final_measurement.metrics)


if __name__ == '__main__':
  app.run(main)
