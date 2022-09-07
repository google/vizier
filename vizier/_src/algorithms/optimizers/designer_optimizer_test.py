"""Tests for designer_optimizer."""

from vizier import pyvizier as vz
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.optimizers import designer_optimizer
from vizier._src.algorithms.testing import optimizer_test_utils

from absl.testing import absltest


class DesignerOptimizerTest(absltest.TestCase):

  def test_smoke(self):
    problem = vz.ProblemStatement()
    problem.search_space.root.add_float_param('a', 0, 1)

    designer_factory = quasi_random.QuasiRandomDesigner.from_problem
    optimizer_test_utils.assert_passes_on_random_single_metric_function(
        self,
        problem.search_space,
        designer_optimizer.DesignerAsOptimizer(designer_factory),
        np_random_seed=1)


if __name__ == '__main__':
  absltest.main()
