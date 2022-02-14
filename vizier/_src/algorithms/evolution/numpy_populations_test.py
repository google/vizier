from absl import logging
from vizier import pyvizier as vz
from vizier._src.algorithms.designers import random
from vizier._src.algorithms.evolution import numpy_populations
from vizier._src.algorithms.testing import test_runners
from vizier.testing import test_studies

from absl.testing import absltest


class NumpyPopulationsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    problem = vz.ProblemStatement(
        test_studies.flat_space_with_all_types(),
        metric_information=test_studies.metrics_all_unconstrained())
    converter = numpy_populations.PopulationConverter(
        problem.search_space, problem.metric_information)

    trials = test_runners.run_with_random_metrics(
        random.RandomDesigner(problem.search_space, seed=0),
        problem,
        iters=1,
        batch_size=10,
        validate_parameters=False)
    population = converter.to_population(trials)

    self._problem = problem
    self._converter = converter
    self._trials = trials
    self._population = population

  def test_convert_to_population(self):
    logging.info('Population: %s', self._population)
    self.assertLen(self._population, 10)

  def test_population_serialization(self):
    dumped = self._population.dump()
    redumped = self._population.recover(dumped).dump()
    self.assertEqual(dumped, redumped)

  def test_convert_from_offspring(self):
    """TODO."""

  def test_offspring_serialization(self):
    """TODO."""

  def test_uniform_random_sampler(self):
    """TODO."""

  def test_linf_mutation(self):
    """TODO."""


if __name__ == '__main__':
  absltest.main()
