"""Tests for cmaes."""
from vizier import benchmarks
from vizier._src.algorithms.designers import cmaes
from vizier._src.algorithms.testing import test_runners
from vizier._src.benchmarks.experimenters.synthetic import bbob

from absl.testing import absltest


class CmaesTest(absltest.TestCase):

  def setUp(self):
    self.problem = bbob.DefaultBBOBProblemStatement(2)
    self.experimenter = benchmarks.NumpyExperimenter(bbob.Sphere, self.problem)
    super().setUp()

  def test_e2e_and_serialization(self):
    designer = cmaes.CMAESDesigner(self.experimenter.problem_statement())

    trials = test_runners.run_with_random_metrics(
        designer,
        self.experimenter.problem_statement(),
        iters=10,
        batch_size=3,
        verbose=1,
        validate_parameters=True)
    self.assertLen(trials, 30)

    new_designer = cmaes.CMAESDesigner(self.experimenter.problem_statement())
    new_designer.load(designer.dump())

    suggestions = designer.suggest(10)
    same_suggestions = new_designer.suggest(10)

    self.assertEqual(suggestions, same_suggestions)


if __name__ == '__main__':
  absltest.main()
