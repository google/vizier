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

"""Tests for surrogate_experimenter."""
from typing import Sequence, Optional
import jax

from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
from vizier._src.benchmarks.experimenters import surrogate_experimenter
from vizier.testing import test_studies

from absl.testing import absltest


class DummyPredictor(vza.Predictor):

  def predict(
      self,
      trials: Sequence[vz.TrialSuggestion],
      rng: Optional[jax.Array] = None,
      num_samples: Optional[int] = None,
  ) -> vza.Prediction:
    num_trials = len(trials)
    mean = jax.random.normal(key=rng, shape=(num_trials,))
    stddev = jax.random.normal(key=rng, shape=(num_trials,))
    return vza.Prediction(mean=mean, stddev=stddev)


class PredictorExperimenterTest(absltest.TestCase):

  def test_e2e(self):
    problem = vz.ProblemStatement(test_studies.flat_space_with_all_types())
    problem.metric_information.append(
        vz.MetricInformation(
            name='metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    predictor = DummyPredictor()
    experimenter = surrogate_experimenter.PredictorExperimenter(
        predictor=predictor, problem_statement=problem
    )

    quasi_random_sampler = quasi_random.QuasiRandomDesigner(
        problem.search_space
    )
    suggestions = quasi_random_sampler.suggest(count=7)
    trials = [suggestion.to_trial() for suggestion in suggestions]
    experimenter.evaluate(trials)

    for trial in trials:
      self.assertEqual(trial.status, vz.TrialStatus.COMPLETED)
      self.assertContainsSubset(
          trial.final_measurement_or_die.metrics.keys(), ['metric']
      )


if __name__ == '__main__':
  absltest.main()
