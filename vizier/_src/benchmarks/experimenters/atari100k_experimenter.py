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

"""Atari100k benchmarks from Dopamine.

Reference on benchmark noise and settings:

`Deep RL at the Edge of the Statistical Precipice.`
(https://arxiv.org/abs/2108.13264)

"""
# pylint:disable=dangerous-default-value
from typing import Dict, Optional, Sequence, Union

from absl import logging
from dopamine.discrete_domains import iteration_statistics
from dopamine.labs.atari_100k import atari_100k_rainbow_agent
from dopamine.labs.atari_100k import eval_run_experiment
import gin

from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter


def create_agent_fn(
    sess,  # pylint: disable=unused-argument
    environment,
    seed: Optional[int] = None,
    summary_writer=None) -> atari_100k_rainbow_agent.Atari100kRainbowAgent:
  """Helper function for creating full rainbow-based Atari 100k agent."""
  return atari_100k_rainbow_agent.Atari100kRainbowAgent(
      num_actions=environment.action_space.n,
      seed=seed,
      summary_writer=summary_writer)


@gin.configurable
class WrappedRunner(eval_run_experiment.MaxEpisodeEvalRunner):
  """Wraps the original Dopamine Runner for Vizier's convenience."""

  def __init__(self, base_dir: str = '/tmp/'):
    super().__init__(base_dir=base_dir, create_agent_fn=create_agent_fn)

  def run_trial(self) -> iteration_statistics.IterationStatistics:
    """Replacement of `run_experiment()` to avoid TensorBoard summary logging."""

    statistics = iteration_statistics.IterationStatistics()

    if self._num_iterations <= self._start_iteration:
      logging.warning('num_iterations (%d) < start_iteration(%d)',
                      self._num_iterations, self._start_iteration)
      return statistics

    for _ in range(self._start_iteration, self._num_iterations):
      self._run_train_phase(statistics)
      self._run_eval_phase(statistics)

    return statistics


GinParameterType = Union[float, int, str]


def default_search_space() -> pyvizier.SearchSpace:
  """Produces a reasonable SearchSpace for tuning the Rainbow training process."""
  ss = pyvizier.SearchSpace()
  ss.root.add_float_param(
      'JaxDQNAgent.gamma',
      0.7,
      0.999999,
      scale_type=pyvizier.ScaleType.REVERSE_LOG)
  ss.root.add_int_param('JaxDQNAgent.update_horizon', 1, 20)
  ss.root.add_int_param('JaxDQNAgent.update_period', 1, 10)
  ss.root.add_int_param('JaxDQNAgent.target_update_period', 1, 10000)
  ss.root.add_int_param('JaxDQNAgent.min_replay_history', 100, 100000)
  ss.root.add_float_param(
      'JaxDQNAgent.epsilon_train',
      0.0000001,
      1.0,
      scale_type=pyvizier.ScaleType.LOG)
  ss.root.add_int_param('JaxDQNAgent.epsilon_decay_period', 1000, 10000)
  ss.root.add_bool_param('JaxFullRainbowAgent.noisy')
  ss.root.add_bool_param('JaxFullRainbowAgent.dueling')
  ss.root.add_bool_param('JaxFullRainbowAgent.double_dqn')
  ss.root.add_int_param('JaxFullRainbowAgent.num_atoms', 1, 100)
  ss.root.add_bool_param('Atari100kRainbowAgent.data_augmentation')
  ss.root.add_float_param(
      'create_optimizer.learning_rate',
      0.0000001,
      1.0,
      scale_type=pyvizier.ScaleType.LOG)
  ss.root.add_float_param(
      'create_optimizer.eps', 0.0000001, 1.0, scale_type=pyvizier.ScaleType.LOG)

  return ss


class Atari100kExperimenter(experimenter.Experimenter):
  """Atari100k Experimenter."""

  def __init__(self,
               game_name: str = 'Pong',
               agent_name: str = 'DER',
               initial_gin_bindings: Dict[str, GinParameterType] = {}):
    """Initializes the Atari100k Experimenter.

    Args:
      game_name: Atari game name. Can be one of 57 games.
      agent_name: Name of the base config file to use. Note that the
        corresponding gin files contain reported/default values. To run them,
        just send in a trivial study_config, which leads to a separate XM run
        containing baseline results.
      initial_gin_bindings: Initial gin bindings to use. Useful for making tests
        small. Will be overridden by the trial's gin bindings.
    """
    self._game_name = game_name
    assert agent_name in ['DER', 'DrQ', 'DrQ_eps', 'OTRainbow']
    self._gin_file = f'vizier/_src/benchmarks/experimenters/atari100k_configs/{agent_name}.gin'
    self._initial_gin_bindings = initial_gin_bindings

  def problem_statement(self) -> pyvizier.ProblemStatement:
    ss = default_search_space()
    problem_statement = pyvizier.ProblemStatement(search_space=ss)
    problem_statement.metric_information.append(
        pyvizier.MetricInformation(
            name='eval_average_return',
            goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE))
    return problem_statement

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    for trial in suggestions:
      with gin.unlock_config():
        # Lock in initial values.
        gin.parse_config_file(self._gin_file)
        gin.bind_parameter('atari_lib.create_atari_environment.game_name',
                           self._game_name)
        for parameter_name in self._initial_gin_bindings:
          gin.bind_parameter(parameter_name,
                             self._initial_gin_bindings[parameter_name])

        # Lock in trial parameters.
        for parameter_name in trial.parameters:
          gin.bind_parameter(parameter_name,
                             trial.parameters[parameter_name].value)

      # Run actual training + inference.
      runner = WrappedRunner()
      statistics = runner.run_trial()
      logging.info('Statistics: %s', statistics.data_lists)

      # Add Intermediate Measurements.
      num_intermediate_measurements = len(
          statistics.data_lists['eval_average_return'])
      for i in range(num_intermediate_measurements):
        measurement = pyvizier.Measurement()
        measurement.metrics['train_average_return'] = statistics.data_lists[
            'train_average_return'][i]
        measurement.metrics[
            'train_average_steps_per_second'] = statistics.data_lists[
                'train_average_steps_per_second'][i]
        measurement.metrics['eval_average_return'] = statistics.data_lists[
            'eval_average_return'][i]
        trial.measurements.append(measurement)

      # Final Measurement.
      trial.complete(trial.measurements[-1])
