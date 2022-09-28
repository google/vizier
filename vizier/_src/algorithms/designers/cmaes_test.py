"""Tests for cmaes."""

from vizier import benchmarks
from vizier._src.algorithms.designers import cmaes
from vizier._src.algorithms.testing import test_runners
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest


class CmaesTest(absltest.TestCase):

  def test_e2e(self):
    problem = bbob.DefaultBBOBProblemStatement(2)
    experimenter = benchmarks.NumpyExperimenter(bbob.Sphere, problem)
    designer = cmaes.CMAESDesigner(experimenter.problem_statement())

    trials = test_runners.run_with_random_metrics(
        designer,
        experimenter.problem_statement(),
        iters=10,
        batch_size=3,
        verbose=1,
        validate_parameters=True)
    self.assertLen(trials, 30)


if __name__ == '__main__':
  absltest.main()
