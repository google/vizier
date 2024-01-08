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

"""Tests for nsga2."""

import datetime
from typing import Optional

from absl import logging

import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz

from vizier._src.algorithms.evolution import nsga2
from vizier._src.algorithms.evolution import templates
from vizier.testing import test_studies

from absl.testing import absltest

np.set_printoptions(precision=3)


def nsga2_on_all_types(
    population_size: int = 50, eviction_limit: Optional[int] = None
) -> templates.CanonicalEvolutionDesigner[nsga2.Population, nsga2.Offspring]:
  problem = vz.ProblemStatement(
      search_space=test_studies.flat_space_with_all_types()
  )
  problem.metric_information.extend([
      vz.MetricInformation(name='m1', goal=vz.ObjectiveMetricGoal.MAXIMIZE),
      vz.MetricInformation(name='m2', goal=vz.ObjectiveMetricGoal.MINIMIZE),
      vz.MetricInformation(
          name='s1',
          goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          safety_threshold=1.0,
      ),
      vz.MetricInformation(
          name='s2',
          goal=vz.ObjectiveMetricGoal.MINIMIZE,
          safety_threshold=1.0,
      ),
  ])

  algorithm = nsga2.NSGA2Designer(
      problem,
      population_size,
      first_survival_after=population_size,
      eviction_limit=eviction_limit,
      seed=0,
  )
  return algorithm


class Nsga2Test(absltest.TestCase):

  def test_survival_by_pareto_rank(self):
    algorithm = nsga2_on_all_types(3)
    # Trial 0 is the only point on the frontier.
    trial0 = vz.Trial(id=0)
    trial0.complete(
        vz.Measurement({'m1': 1.0, 'm2': 0.0, 's1': 2.0, 's2': 0.0})
    )

    # 4 safe trials with the same pareto rank. Crowding distance is computed
    # among them to break ties. Trial 3 is less "crowded" than Trial 2.
    trial1 = vz.Trial(id=1)
    trial1.complete(
        vz.Measurement({'m1': 0.0, 'm2': -1.0, 's1': 2.0, 's2': 0.0})
    )
    trial2 = vz.Trial(id=2)
    trial2.complete(
        vz.Measurement({'m1': 0.5, 'm2': -0.5, 's1': 2.0, 's2': 0.0})
    )
    trial3 = vz.Trial(id=3)
    trial3.complete(
        vz.Measurement({'m1': 0.2, 'm2': -0.2, 's1': 2.0, 's2': 0.0})
    )
    trial4 = vz.Trial(id=4)
    trial4.complete(
        vz.Measurement({'m1': 0.3, 'm2': -0.3, 's1': 2.0, 's2': 0.0})
    )

    trials = vza.CompletedTrials([trial0, trial1, trial2, trial3, trial4])
    algorithm.update(trials, vza.ActiveTrials())
    self.assertSetEqual(set(algorithm.population.trial_ids), {0, 1, 2})

  def test_survival_by_crowding_distance(self):
    algorithm = nsga2_on_all_types(4)
    # Trial 0 is the only point on the frontier.
    trial0 = vz.Trial(id=0)
    trial0.complete(
        vz.Measurement({'m1': 1.001, 'm2': -1.001, 's1': 2.0, 's2': 0.0})
    )

    # 4 safe trials with the same pareto rank. Crowding distance is computed
    # among them to break ties. Trial 3 is less "crowded" than Trial 2.
    trial1 = vz.Trial(id=1)
    trial1.complete(
        vz.Measurement({'m1': 1.0, 'm2': 0.0, 's1': 2.0, 's2': 0.9})
    )
    trial2 = vz.Trial(id=2)
    trial2.complete(
        vz.Measurement({'m1': 0.9, 'm2': -0.1, 's1': 2.0, 's2': 0.9})
    )
    trial3 = vz.Trial(id=3)
    trial3.complete(
        vz.Measurement({'m1': 0.5, 'm2': -0.5, 's1': 2.0, 's2': 0.9})
    )
    trial4 = vz.Trial(id=4)
    trial4.complete(
        vz.Measurement({'m1': 0.0, 'm2': -1.0, 's1': 2.0, 's2': 0.9})
    )

    trials = vza.CompletedTrials([trial0, trial1, trial2, trial3, trial4])
    algorithm.update(trials, vza.ActiveTrials())
    self.assertSetEqual(set(algorithm.population.trial_ids), {0, 1, 3, 4})

  def test_survival_by_safety(self):
    algorithm = nsga2_on_all_types(3)

    # Trial 1 violates 's1'.
    trial1 = vz.Trial(id=1)
    trial1.complete(
        vz.Measurement({'m1': 1.0, 'm2': 1.0, 's1': 0.0, 's2': 0.0})
    )
    # Trial 2 violates no constraints.
    trial2 = vz.Trial(id=2)
    trial2.complete(
        vz.Measurement({'m1': 0.9, 'm2': 0.9, 's1': 2.0, 's2': 0.0})
    )
    # Trial 3 and 4 violate both 's1' and 's2'.
    trial3 = vz.Trial(id=3)
    trial3.complete(
        vz.Measurement({'m1': 0.5, 'm2': 0.5, 's1': 0.0, 's2': 2.0})
    )
    trial4 = vz.Trial(id=4)
    trial4.complete(
        vz.Measurement({'m1': 0.0, 'm2': 0.0, 's1': 0.0, 's2': 2.0})
    )

    trials = vza.CompletedTrials([trial1, trial2, trial3, trial4])
    algorithm.update(trials, vza.ActiveTrials())
    self.assertSetEqual(set(algorithm.population.trial_ids), {1, 2, 4})

  def test_comprehensive_sanity_check(self):
    algorithm = nsga2_on_all_types(5, eviction_limit=3)

    tid = 1
    for i in range(10):
      if i == 8:
        dumped = algorithm.dump()
      tick = datetime.datetime.now()
      suggestions = algorithm.suggest()
      tock = datetime.datetime.now()
      logging.info('Iteration %s: Suggestion took %s.', i, tock - tick)
      trials = []
      for t in suggestions:
        trials.append(
            t.to_trial(tid).complete(
                vz.Measurement(
                    metrics={
                        'm1': np.random.random(),
                        'm2': np.random.random(),
                        's1': np.random.uniform(1.0, 2.0),
                        's2': np.random.uniform(
                            0.0, 1.0
                        ),  # fails with 25% probability
                    }
                )
            )
        )
        tid += 1
      tick = datetime.datetime.now()
      logging.info(
          'Suggesitons evaluated: %s', '\n'.join(repr(t) for t in trials)
      )
      algorithm.update(vza.CompletedTrials(trials), vza.ActiveTrials())
      tock = datetime.datetime.now()
      logging.info(
          (
              'Iteration %s: Update took %s.\nPopulation(in array'
              ' format):%s\nAges:%s'
          ),
          i,
          tock - tick,
          algorithm.population.xs,
          algorithm.population.ages,
      )
      tick = tock

    self.assertTrue(np.all(algorithm.population.ages <= 3))

    ys = algorithm.population.ys
    pareto = algorithm.population[nsga2._pareto_rank(ys) == 0]
    logging.info('Pareto frontier %s %s', pareto.xs, pareto.ys)

    # Smoke test dump-load.
    algorithm.load(dumped)

  def test_seeding(self):
    algorithm_1 = nsga2_on_all_types(10)
    algorithm_2 = nsga2_on_all_types(10)

    self.assertEqual(algorithm_1.suggest(10), algorithm_2.suggest(10))

    completed_trials = []

    for tid in range(1, 11):
      trial = vz.Trial(id=tid)

      trial.complete(
          vz.Measurement({
              'm1': np.random.random(),
              'm2': np.random.random(),
              's1': np.random.random(),
              's2': np.random.random(),
          })
      )

      completed_trials.append(trial)

    trials = vza.CompletedTrials(completed_trials)
    algorithm_1.update(trials, vza.ActiveTrials())
    algorithm_2.update(trials, vza.ActiveTrials())

    self.assertEqual(algorithm_1.suggest(10), algorithm_2.suggest(10))


if __name__ == '__main__':
  absltest.main()
