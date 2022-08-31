"""Tests for bocs."""
from vizier._src.algorithms.designers import harmonica
from vizier._src.algorithms.testing import test_runners
from vizier._src.benchmarks.experimenters import combo_experimenter

from absl.testing import absltest


class HarmonicaTest(absltest.TestCase):

  def test_make_suggestions(self):
    experimenter = combo_experimenter.IsingExperimenter(lamda=0.01)
    designer = harmonica.HarmonicaDesigner(
        experimenter.problem_statement(), num_init_samples=1)

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
