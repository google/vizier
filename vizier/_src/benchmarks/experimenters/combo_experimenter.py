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

"""Categorical benchmarks from https://github.com/QUVA-Lab/COMBO.

Lines with 'ATTENTION' mean that code was modified from original Github, in
order to fix bugs with the original code.
"""

# pyformat: disable
from typing import List, Optional, Sequence, Tuple
import numpy as np

from vizier import pyvizier
from vizier._src.benchmarks.experimenters import experimenter
from vizier._src.benchmarks.experimenters.combo import common

fileOpen = open


class IsingExperimenter(experimenter.Experimenter):
  """Ising Sparisification Problem."""

  def __init__(self,
               lamda: float = 1e-2,
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

    self._problem_statement = self.problem_statement()

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
          pyvizier.Measurement(metrics={
              self._problem_statement.single_objective_metric_name: evaluation
          }))

  def problem_statement(self) -> pyvizier.ProblemStatement:
    problem_statement = pyvizier.ProblemStatement()
    root = problem_statement.search_space.root
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
               lamda: float = 1e-2,
               contamination_n_stages: int = 25,
               random_seed: Optional[int] = None):
    self._lamda = lamda
    self._contamination_n_stages = contamination_n_stages
    self._init_z, self._lambdas, self._gammas = self._generate_contamination_dynamics(
        random_seed)
    self._problem_statement = self.problem_statement()

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    for suggestion in suggestions:
      # TODO: Switch to using StudyConfig.
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
          pyvizier.Measurement(metrics={
              self._problem_statement.single_objective_metric_name: evaluation
          }))

  def problem_statement(self) -> pyvizier.ProblemStatement:
    problem_statement = pyvizier.ProblemStatement()
    root = problem_statement.search_space.root
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


class CentroidExperimenter(experimenter.Experimenter):
  """General parameter variant of the Ising Sparisification Problem."""

  def __init__(self,
               centroid_n_choice=3,
               centroid_grid=(4, 4),
               ising_grid_h: int = 4,
               random_seed: Optional[int] = None):
    self._centroid_n_choice = centroid_n_choice
    self._centroid_grid = centroid_grid
    self._centroid_n_edges = centroid_grid[0] * (centroid_grid[1] - 1) + (
        centroid_grid[0] - 1) * centroid_grid[1]
    self._ising_grid_h = ising_grid_h

    self._interaction_list = []
    self._covariance_list = []
    self._partition_original_list = []
    self._n_ising_models = 3
    ising_seeds = np.random.RandomState(random_seed).randint(
        0, 10000, (self._n_ising_models,))
    for i in range(self._n_ising_models):
      interaction = common.generate_ising_interaction(self._centroid_grid[0],
                                                      self._centroid_grid[1],
                                                      ising_seeds[i])
      covariance, partition_original = common.spin_covariance(
          interaction, self._centroid_grid)
      self._interaction_list.append(interaction)
      self._covariance_list.append(covariance)
      self._partition_original_list.append(partition_original)

    self._problem_statement = self.problem_statement()

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    for suggestion in suggestions:
      # TODO: Switch to using StudyConfig.
      x = np.array([
          int(suggestion.parameters[f'x_{i}'].value)
          for i in range(self._centroid_n_edges)
      ])
      interaction_mixed = self._edge_choice(x, self._interaction_list)
      log_partition_mixed = common.log_partition(
          interaction_mixed, self._centroid_grid)  # ATTENTION
      kld_sum = 0

      for i in range(self._n_ising_models):
        kld = common.ising_dense(
            ising_grid_h=self._centroid_grid[0],  # ATTENTION
            interaction_original=self._interaction_list[i],
            interaction_sparsified=interaction_mixed,
            covariance=self._covariance_list[i],
            log_partition_original=np.log(
                self._partition_original_list[i]),  # ATTENTION
            log_partition_new=log_partition_mixed)  # ATTENTION
        kld_sum += kld
      evaluation = float(kld_sum / float(self._n_ising_models))

      suggestion.complete(
          pyvizier.Measurement(metrics={
              self._problem_statement.single_objective_metric_name: evaluation
          }))

  def problem_statement(self) -> pyvizier.ProblemStatement:
    problem_statement = pyvizier.ProblemStatement()
    root = problem_statement.search_space.root
    for i in range(self._centroid_n_edges):
      root.add_categorical_param(
          name='x_{}'.format(i),
          feasible_values=[str(j) for j in range(self._centroid_n_choice)])
    problem_statement.metric_information.append(
        pyvizier.MetricInformation(
            name='main_objective', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE))
    return problem_statement

  def _edge_choice(
      self, x: np.ndarray, interaction_list: List[Tuple[np.ndarray, np.ndarray]]
  ) -> Tuple[np.ndarray, np.ndarray]:
    edge_weight = np.zeros(x.shape)
    for i in range(len(interaction_list)):
      edge_weight[x == i] = np.hstack([
          interaction_list[i][0].reshape(-1), interaction_list[i][1].reshape(-1)
      ])[x == i]
    grid_h, grid_w = self._centroid_grid
    split_ind = grid_h * (grid_w - 1)
    return edge_weight[:split_ind].reshape(
        (grid_h, grid_w - 1)), edge_weight[split_ind:].reshape(
            (grid_h - 1, grid_w))


class PestControlExperimenter(experimenter.Experimenter):
  """Pest Control Problem."""

  def __init__(self,
               pest_control_n_choice: int = 5,
               pest_control_n_stages: int = 25,
               random_seed: Optional[int] = None):
    self._pest_control_n_choice = pest_control_n_choice
    self._pest_control_n_stages = pest_control_n_stages
    self._random_seed = random_seed
    self._problem_statement = self.problem_statement()

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    for suggestion in suggestions:
      # TODO: Switch to using StudyConfig.
      x = np.array([
          int(suggestion.parameters[f'x_{i}'].value)
          for i in range(self._pest_control_n_stages)
      ])
      evaluation = self._pest_control_score(x)
      suggestion.complete(
          pyvizier.Measurement(metrics={
              self._problem_statement.single_objective_metric_name: evaluation
          }))

  def problem_statement(self) -> pyvizier.ProblemStatement:
    problem_statement = pyvizier.ProblemStatement()
    root = problem_statement.search_space.root
    for i in range(self._pest_control_n_stages):
      root.add_categorical_param(
          name='x_{}'.format(i),
          feasible_values=[str(j) for j in range(self._pest_control_n_choice)])
    problem_statement.metric_information.append(
        pyvizier.MetricInformation(
            name='main_objective', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE))
    return problem_statement

  def _pest_spread(self, curr_pest_frac: float, spread_rate: float,
                   control_rate: float, apply_control: bool):
    if apply_control:
      next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
      next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac

  def _pest_control_score(self, x: np.ndarray) -> float:
    u = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {
        1: 1.0 / 7.0,
        2: 2.5 / 7.0,
        3: 2.0 / 7.0,
        4: 0.5 / 7.0
    }
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0

    init_pest_frac = np.random.RandomState(self._random_seed).beta(
        init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
      spread_rate = np.random.RandomState(self._random_seed).beta(
          spread_alpha, spread_beta, size=(n_simulations,))
      do_control = x[i] > 0
      if do_control:
        control_rate = np.random.RandomState(self._random_seed).beta(
            control_alpha, control_beta[x[i]], size=(n_simulations,))
        next_pest_frac = self._pest_spread(curr_pest_frac, spread_rate,
                                           control_rate, True)
        # Tolerance has been developed for pesticide type 1.
        control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
        # You will get a discount.
        payed_price = control_price[x[i]] * (
            1.0 - control_price_max_discount[x[i]] / float(n_stages) *
            float(np.sum(x == x[i])))
      else:
        next_pest_frac = self._pest_spread(curr_pest_frac, spread_rate, 0,
                                           False)
        payed_price = 0
      payed_price_sum += payed_price
      above_threshold += np.mean(curr_pest_frac > u)
      curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold


# MAXSAT Files can be found in
# https://github.com/QUVA-Lab/COMBO/tree/master/COMBO/experiments/MaxSAT/maxsat2018_data.

MAXSAT28_FILE = 'maxsat2018_data/maxcut-johnson8-2-4.clq.wcnf'
MAXSAT43_FILE = 'maxsat2018_data/maxcut-hamming8-2.clq.wcnf'
MAXSAT60_FILE = 'maxsat2018_data/frb-frb10-6-4.wcnf'


class MAXSATExperimenter(experimenter.Experimenter):
  """MAXSAT Problem."""

  def __init__(self, data_filename: str):
    self._data_filename = data_filename
    f = fileOpen(self._data_filename, 'rt')
    line_str = f.readline()
    while line_str[:2] != 'p ':
      line_str = f.readline()
    self._n_variables = int(line_str.split(' ')[2])
    self._n_clauses = int(line_str.split(' ')[3])
    self._n_vertices = np.array([2] * self._n_variables)
    raw_clauses = [(float(clause_str.split(' ')[0]),
                    clause_str.split(' ')[1:-1])
                   for clause_str in f.readlines()]
    f.close()
    weights = np.array([elm[0] for elm in raw_clauses]).astype(np.float32)
    weight_mean = np.mean(weights)
    weight_std = np.std(weights)
    self._weights = (weights - weight_mean) / weight_std
    self._clauses = []
    for _, clause in raw_clauses:
      pair = ([abs(int(elm)) - 1 for elm in clause],
              [int(elm) > 0 for elm in clause])
      self._clauses.append(pair)

    self._problem_statement = self.problem_statement()

  def evaluate(self, suggestions: Sequence[pyvizier.Trial]):
    for suggestion in suggestions:
      # TODO: Switch to using StudyConfig.
      bools = [
          int(suggestion.parameters[f'x_{i}'].value == 'True')
          for i in range(self._n_variables)
      ]
      x = np.array(bools, dtype=bool)
      satisfied = np.array([
          (x[clause[0]] == clause[1]).any() for clause in self._clauses
      ])
      evaluation = -np.sum(self._weights * satisfied)
      suggestion.complete(
          pyvizier.Measurement(metrics={
              self._problem_statement.single_objective_metric_name: evaluation
          }))

  def problem_statement(self) -> pyvizier.ProblemStatement:
    problem_statement = pyvizier.ProblemStatement()
    root = problem_statement.search_space.root
    for i in range(self._n_variables):
      root.add_bool_param(name=f'x_{i}')
    problem_statement.metric_information.append(
        pyvizier.MetricInformation(
            name='main_objective', goal=pyvizier.ObjectiveMetricGoal.MINIMIZE))
    return problem_statement
