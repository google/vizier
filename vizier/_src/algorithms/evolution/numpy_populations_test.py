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

import copy

from absl import logging
import attr
import numpy as np
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

  def test_linf_mutation_no_state_modification(self):
    population_copy = copy.deepcopy(self._population)

    mutation = numpy_populations.LinfMutation()
    mutation.mutate(self._population, 0)

    population_copy_dict = attr.asdict(population_copy)
    population_dict = attr.asdict(self._population)
    for k, v in population_copy_dict.items():
      np.testing.assert_array_equal(v, population_dict[k], err_msg=k)

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
