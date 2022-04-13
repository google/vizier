"""Example of a Vizier Client, which can be run on multiple machines.

This is meant to be used after the Vizier Server (see `run_vizier_server.py`)
has been launched and provided an address to connect to.
"""

from typing import Sequence

from absl import app
from absl import flags

from vizier.service import pyvizier as vz
from vizier.service import vizier_client

flags.DEFINE_string('address', 'localhost:6006',
                    'Address of the Vizier Server.')
flags.DEFINE_integer(
    'max_num_iterations', 10,
    'Maximum number of possible iterations / calls to get suggestions.')
flags.DEFINE_integer(
    'suggestion_count', 5,
    'Number of suggestions to evaluate per iteration. Useful for batched evaluations.'
)

FLAGS = flags.FLAGS


def evaluate_trial(trial: vz.Trial) -> vz.Measurement:
  """Dummy evaluator used as an example."""
  learning_rate = trial.parameters.get_value('learning_rate')
  num_layers = trial.parameters.get_value('num_layers')
  m = vz.Measurement()
  m.metrics = {'accuracy': learning_rate * num_layers}  # dummy accuracy
  return m


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

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
  study_config.algorithm = vz.Algorithm.RANDOM_SEARCH

  client = vizier_client.create_or_load_study(
      service_endpoint=FLAGS.address,
      owner_id='my_name',
      client_id='my_client_id',
      study_display_name='cifar10',
      study_config=study_config)

  for _ in range(FLAGS.max_num_iterations):
    # Evaluate the suggestion(s) and report the results to Vizier.
    suggestions = client.get_suggestions(
        suggestion_count=FLAGS.suggestion_count)
    for trial in suggestions:
      measurement = evaluate_trial(trial)
      client.complete_trial(trial_id=trial.id, final_measurement=measurement)


if __name__ == '__main__':
  app.run(main)
