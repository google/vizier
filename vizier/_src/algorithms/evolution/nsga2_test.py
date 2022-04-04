"""Tests for nsga2."""

import datetime
from typing import Optional

from absl import logging
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.core import deltas
from vizier._src.algorithms.evolution import nsga2
from vizier.testing import test_studies

from absl.testing import absltest

np.set_printoptions(precision=3)


def nsga2_on_all_types(population_size: int = 50,
                       eviction_limit: Optional[int] = None) -> nsga2.NSGA2:
  study_config = vz.StudyConfig(
      search_space=test_studies.flat_space_with_all_types())
  study_config.metric_information.extend([
      vz.MetricInformation(name='m1', goal=vz.ObjectiveMetricGoal.MAXIMIZE),
      vz.MetricInformation(name='m2', goal=vz.ObjectiveMetricGoal.MINIMIZE),
      vz.MetricInformation(
          name='s1', goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          safety_threshold=1.0),
      vz.MetricInformation(
          name='s2', goal=vz.ObjectiveMetricGoal.MINIMIZE, safety_threshold=1.0)
  ])

  algorithm = nsga2.NSGA2(
      study_config.search_space,
      study_config.metric_information,
      population_size,
      first_survival_after=population_size,
      eviction_limit=eviction_limit)
  return algorithm


class Nsga2Test(absltest.TestCase):

  def test_survival_by_pareto_rank(self):
    algorithm = nsga2_on_all_types(3)
    # Trial 0 is the only point on the frontier.
    trial0 = vz.Trial(id=0)
    trial0.complete(vz.Measurement({'m1': 1., 'm2': 0., 's1': 2., 's2': .0}))

    # 4 safe trials with the same pareto rank. Crowding distance is computed
    # among them to break ties. Trial 3 is less "crowded" than Trial 2.
    trial1 = vz.Trial(id=1)
    trial1.complete(vz.Measurement({'m1': 0., 'm2': -1., 's1': 2., 's2': .0}))
    trial2 = vz.Trial(id=2)
    trial2.complete(vz.Measurement({'m1': .5, 'm2': -.5, 's1': 2., 's2': .0}))
    trial3 = vz.Trial(id=3)
    trial3.complete(vz.Measurement({'m1': .2, 'm2': -.2, 's1': 2., 's2': .0}))
    trial4 = vz.Trial(id=4)
    trial4.complete(vz.Measurement({'m1': .3, 'm2': -.3, 's1': 2., 's2': .0}))

    trials = deltas.CompletedTrials([trial0, trial1, trial2, trial3, trial4])
    algorithm.update(trials)
    self.assertSetEqual({t.id for t in algorithm._pool.trials}, {0, 1, 2})

  def test_survival_by_crowding_distance(self):
    algorithm = nsga2_on_all_types(4)
    # Trial 0 is the only point on the frontier.
    trial0 = vz.Trial(id=0)
    trial0.complete(
        vz.Measurement({
            'm1': 1.001,
            'm2': -1.001,
            's1': 2.,
            's2': 0.
        }))

    # 4 safe trials with the same pareto rank. Crowding distance is computed
    # among them to break ties. Trial 3 is less "crowded" than Trial 2.
    trial1 = vz.Trial(id=1)
    trial1.complete(vz.Measurement({'m1': 1., 'm2': 0., 's1': 2., 's2': .9}))
    trial2 = vz.Trial(id=2)
    trial2.complete(vz.Measurement({'m1': .9, 'm2': -.1, 's1': 2., 's2': .9}))
    trial3 = vz.Trial(id=3)
    trial3.complete(vz.Measurement({'m1': .5, 'm2': -.5, 's1': 2., 's2': .9}))
    trial4 = vz.Trial(id=4)
    trial4.complete(vz.Measurement({'m1': .0, 'm2': -1., 's1': 2., 's2': .9}))

    trials = deltas.CompletedTrials([trial0, trial1, trial2, trial3, trial4])
    algorithm.update(trials)
    self.assertSetEqual({t.id for t in algorithm._pool.trials}, {0, 1, 3, 4})

  def test_survival_by_safety(self):
    algorithm = nsga2_on_all_types(3)

    # Trial 1 violates 's1'.
    trial1 = vz.Trial(id=1)
    trial1.complete(vz.Measurement({'m1': 1., 'm2': 1., 's1': .0, 's2': .0}))
    # Trial 2 violates no constraints.
    trial2 = vz.Trial(id=2)
    trial2.complete(vz.Measurement({'m1': .9, 'm2': .9, 's1': 2., 's2': .0}))
    # Trial 3 and 4 violate both 's1' and 's2'.
    trial3 = vz.Trial(id=3)
    trial3.complete(vz.Measurement({'m1': .5, 'm2': .5, 's1': .0, 's2': 2.}))
    trial4 = vz.Trial(id=4)
    trial4.complete(vz.Measurement({'m1': .0, 'm2': .0, 's1': .0, 's2': 2.}))

    trials = deltas.CompletedTrials([trial1, trial2, trial3, trial4])
    algorithm.update(trials)
    self.assertSetEqual({t.id for t in algorithm._pool.trials}, {1, 2, 4})

  def test_comprehensive_sanity_check(self):
    algorithm = nsga2_on_all_types(5, eviction_limit=3)

    tid = 1
    for i in range(10):
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
                        'm1':
                            np.random.random(),
                        'm2':
                            np.random.random(),
                        's1':
                            np.random.uniform(1., 2.),
                        's2':
                            np.random.uniform(.0, 1.
                                             )  # fails with 25% probability
                    })))
        tid += 1
      tick = datetime.datetime.now()
      logging.info('Suggesitons evaluated: %s',
                   algorithm._trials_to_population(trials).xs.asarray())
      algorithm.update(deltas.CompletedTrials(trials))
      tock = datetime.datetime.now()
      logging.info(
          'Iteration %s: Update took %s.\nPopulation(in array format):%s\nAges:%s',
          i, tock - tick, algorithm._pool.xs.asarray(), algorithm._pool.ages)
      tick = tock

    self.assertTrue(np.all(algorithm._pool.ages <= 3))

    ys = algorithm._pool.ys
    pareto = algorithm._pool[nsga2._pareto_rank(ys) == 0]
    logging.info('Pareto frontier %s %s', pareto.xs.asarray(), pareto.ys)


if __name__ == '__main__':
  absltest.main()
