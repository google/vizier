"""Categorical benchmarks from https://github.com/QUVA-Lab/COMBO."""
from typing import Optional, Sequence, Tuple
import numpy as np

from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter
from vizier._src.benchmarks.experimenters.combo import common


class IsingExperimenter(experimenter.Experimenter):
  """Ising Sparisification Problem."""

  def __init__(self,
               lamda: float,
               ising_grid_h: int = 4,
               ising_grid_w: int = 4,
               ising_n_edges: int = 24,
               random_seed: Optional[int] = None):
    self._lamda = lamda
    self._ising_grid_h = ising_grid_h
    self._ising_grid_w = ising_grid_w
    self._ising_n_edges = ising_n_edges
    self._interaction = common.generate_ising_interaction(
        self._ising_grid_h, self._ising_grid_w, random_seed)
    self._covariance, self._partition_original = common.spin_covariance(
        self._interaction, (self._ising_grid_h, self._ising_grid_w))

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    for suggestion in suggestions:
      # TODO: Switch to using StudyConfig.
      x = np.array([
          int(suggestion.parameters[f'x_{i}'].value == 'True')
          for i in range(self._ising_n_edges)
      ])
      x_h, x_v = self._bocs_consistency_mapping(x)
      interaction_sparsified = x_h * self._interaction[
          0], x_v * self._interaction[1]
      log_partition_sparsified = common.log_partition(
          interaction_sparsified, (self._ising_grid_h, self._ising_grid_w))
      evaluation = common.ising_dense(
          ising_grid_h=self._ising_grid_h,
          interaction_original=self._interaction,
          interaction_sparsified=interaction_sparsified,
          covariance=self._covariance,
          log_partition_original=np.log(self._partition_original),
          log_partition_new=log_partition_sparsified)
      evaluation += self._lamda * float(np.sum(x))

      suggestion.complete(
          pyvizier.Measurement(metrics={'main_objective': evaluation}))

  def problem_statement(self) -> pyvizier.ProblemStatement:
    problem_statement = pyvizier.ProblemStatement()
    root = problem_statement.search_space.select_root()
    for i in range(self._ising_n_edges):
      root.add_bool_param(name=f'x_{i}')
    problem_statement.metric_information.append(
        pyvizier.MetricInformation(
            name='main_objective', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE))
    return problem_statement

  def _bocs_consistency_mapping(self,
                                x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    horizontal_ind = [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 22, 23]
    vertical_ind = sorted(
        [elm for elm in range(24) if elm not in horizontal_ind])
    return x[horizontal_ind].reshape(
        (self._ising_grid_h, self._ising_grid_w - 1)), x[vertical_ind].reshape(
            (self._ising_grid_h - 1, self._ising_grid_w))


class ContaminationExperimenter(experimenter.Experimenter):
  """Contamination Control Problem."""

  def __init__(self,
               lamda: float,
               contamination_n_stages: int = 25,
               random_seed: Optional[int] = None):
    self._lamda = lamda
    self._contamination_n_stages = contamination_n_stages
    self._init_z, self._lambdas, self._gammas = self._generate_contamination_dynamics(
        random_seed)

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    for suggestion in suggestions:
      x = np.array([
          int(suggestion.parameters[f'x_{i}'].value == 'True')
          for i in range(self._contamination_n_stages)
      ])
      evaluation = self._contamination(
          x=x,
          cost=np.ones(x.size),
          init_z=self._init_z,
          lambdas=self._lambdas,
          gammas=self._gammas,
          u=0.1,
          epsilon=0.05)
      evaluation += self._lamda * float(np.sum(x))
      suggestion.complete(
          pyvizier.Measurement(metrics={'main_objective': evaluation}))

  def problem_statement(self) -> pyvizier.ProblemStatement:
    problem_statement = pyvizier.ProblemStatement()
    root = problem_statement.search_space.select_root()
    for i in range(self._contamination_n_stages):
      root.add_bool_param(name=f'x_{i}')
    problem_statement.metric_information.append(
        pyvizier.MetricInformation(
            name='main_objective', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE))
    return problem_statement

  def _contamination(self, x: np.ndarray, cost: np.ndarray, init_z: np.ndarray,
                     lambdas: np.ndarray, gammas: np.ndarray, u: float,
                     epsilon: float) -> float:
    assert x.size == self._contamination_n_stages

    rho = 1.0
    n_simulations = 100

    z = np.zeros((x.size, n_simulations))
    z[0] = lambdas[0] * (1.0 - x[0]) * (1.0 - init_z) + (
        1.0 - gammas[0] * x[0]) * init_z
    for i in range(1, self._contamination_n_stages):
      z[i] = lambdas[i] * (1.0 - x[i]) * (1.0 - z[i - 1]) + (
          1.0 - gammas[i] * x[i]) * z[i - 1]

    below_threshold = z < u
    constraints = np.mean(below_threshold, axis=1) - (1.0 - epsilon)

    return np.sum(x * cost - rho * constraints)

  def _generate_contamination_dynamics(
      self, random_seed=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_stages = self._contamination_n_stages
    n_simulations = 100

    init_alpha = 1.0
    init_beta = 30.0
    contam_alpha = 1.0
    contam_beta = 17.0 / 3.0
    restore_alpha = 1.0
    restore_beta = 3.0 / 7.0
    init_z = np.random.RandomState(random_seed).beta(
        init_alpha, init_beta, size=(n_simulations,))
    lambdas = np.random.RandomState(random_seed).beta(
        contam_alpha, contam_beta, size=(n_stages, n_simulations))
    gammas = np.random.RandomState(random_seed).beta(
        restore_alpha, restore_beta, size=(n_stages, n_simulations))

    return init_z, lambdas, gammas
