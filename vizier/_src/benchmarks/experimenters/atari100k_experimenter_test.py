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

"""Tests for atari100k."""
from absl import logging

from vizier import pyvizier
from vizier._src.algorithms.designers import random

from absl.testing import absltest
from absl.testing import parameterized


class Atari100KTest(parameterized.TestCase):

  @absltest.skip("ALE ROMS must be installed manually.")
  @parameterized.parameters('DER', 'DrQ', 'DrQ_eps', 'OTRainbow')
  @absltest.skip('Jax versioning not updated in Dopamine.')
  def test_e2e_evaluation(self, agent_name):
    from vizier._src.benchmarks.experimenters import atari100k_experimenter  # pylint: disable=g-import-not-at-top

    initial_gin_bindings = {
        'Runner.training_steps': 2,
        'MaxEpisodeEvalRunner.num_eval_episodes': 2,
        'Runner.num_iterations': 2,
        'Runner.max_steps_per_episode': 2,
        'JaxDQNAgent.min_replay_history': 2,
        'OutOfGraphPrioritizedReplayBuffer.replay_capacity': 1000,
    }
    experimenter = atari100k_experimenter.Atari100kExperimenter(
        game_name='Pong',
        agent_name=agent_name,
        initial_gin_bindings=initial_gin_bindings,
    )

    designer = random.RandomDesigner(
        experimenter.problem_statement().search_space, seed=None
    )

    suggestions = designer.suggest(2)
    trials = [suggestion.to_trial() for suggestion in suggestions]
    experimenter.evaluate(trials)
    # Trials should be completed in place.
    for trial in trials:
      self.assertEqual(trial.status, pyvizier.TrialStatus.COMPLETED)
      logging.info('Evaluated Trial: %s', trial)
      if trial.final_measurement:
        value = trial.final_measurement.metrics['eval_average_return'].value
        self.assertGreaterEqual(value, 0.0)


if __name__ == '__main__':
  absltest.main()
