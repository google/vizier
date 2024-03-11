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

"""Tests for combo_experimenter."""
# pylint:disable=g-long-lambda
import functools
from absl import logging

from vizier import pyvizier
from vizier._src.algorithms.designers import random
from vizier._src.benchmarks.experimenters import combo_experimenter

from absl.testing import absltest
from absl.testing import parameterized


class ComboExperimenterTest(parameterized.TestCase):
  """Default lamda values found in README of https://github.com/QUVA-Lab/COMBO."""

  @parameterized.named_parameters(
      ('contamination',
       functools.partial(
           combo_experimenter.ContaminationExperimenter,
           lamda=0.01), 20.0, 30.0),
      ('ising',
       functools.partial(combo_experimenter.IsingExperimenter,
                         lamda=0.01), 0.0, 50.0),
      ('centroid', combo_experimenter.CentroidExperimenter, 0.0, 150.0),
      ('pest_control', combo_experimenter.PestControlExperimenter, 12.0, 25.0),
  )
  def test_experimenters(self, experimenter_class, objective_min,
                         objective_max):
    """Tests if entire pipeline works and objective values make sense."""
    experimenter = experimenter_class()

    problem_statement = experimenter.problem_statement()
    logging.info(problem_statement)
    designer = random.RandomDesigner(
        search_space=problem_statement.search_space, seed=None)

    suggestions = designer.suggest(5)
    trials = [suggestion.to_trial() for suggestion in suggestions]
    experimenter.evaluate(trials)
    for trial in trials:
      logging.info('Evaluated Trial: %s', trial)
      self.assertEqual(trial.status, pyvizier.TrialStatus.COMPLETED)
      metric_name = problem_statement.metric_information.item().name
      eval_objective = trial.final_measurement_or_die.metrics[metric_name].value
      self.assertLessEqual(eval_objective, objective_max)
      self.assertGreaterEqual(eval_objective, objective_min)


if __name__ == '__main__':
  absltest.main()
