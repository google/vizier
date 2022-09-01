"""Tests for bocs."""
from vizier._src.algorithms.designers import bocs
from vizier._src.algorithms.testing import test_runners
from vizier._src.benchmarks.experimenters import combo_experimenter

from absl.testing import absltest
from absl.testing import parameterized


class BocsTest(parameterized.TestCase):

  @parameterized.parameters((bocs.SemiDefiniteProgramming,),
                            (bocs.SimulatedAnnealing,))
  def test_make_suggestions(self, acquisition_optimizer_factory):
    experimenter = combo_experimenter.IsingExperimenter(lamda=0.01)
    designer = bocs.BOCSDesigner(
        experimenter.problem_statement(),
        acquisition_optimizer_factory=acquisition_optimizer_factory,
        num_initial_randoms=1)

    trials = test_runners.run_with_random_metrics(
        designer,
        experimenter.problem_statement(),
        iters=5,
        batch_size=1,
        verbose=1,
        validate_parameters=True)
    self.assertLen(trials, 5)


if __name__ == '__main__':
  absltest.main()
