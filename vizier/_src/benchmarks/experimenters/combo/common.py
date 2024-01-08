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

"""Common utility functions for COMBO benchmarks."""
# pylint:disable = missing-function-docstring
import itertools
from typing import Tuple
import numpy as np


def spin_covariance(interaction: Tuple[np.ndarray, np.ndarray],
                    grid_shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
  horizontal_interaction, vertical_interaction = interaction
  n_vars = horizontal_interaction.shape[0] * vertical_interaction.shape[1]
  spin_cfgs = np.array(list(itertools.product(*([[-1, 1]] * n_vars))))
  density = np.zeros(spin_cfgs.shape[0])
  for i in range(spin_cfgs.shape[0]):
    spin_cfg = spin_cfgs[i].reshape(grid_shape)
    h_comp = spin_cfg[:, :-1] * horizontal_interaction * spin_cfg[:, 1:] * 2
    v_comp = spin_cfg[:-1] * vertical_interaction * spin_cfg[1:] * 2
    log_interaction_energy = np.sum(h_comp) + np.sum(v_comp)
    density[i] = np.exp(log_interaction_energy)
  interaction_partition = np.sum(density)
  density = density / interaction_partition

  covariance = spin_cfgs.T.dot(spin_cfgs * density.reshape((-1, 1)))
  return covariance, interaction_partition


def partition(interaction: Tuple[np.ndarray, np.ndarray],
              grid_shape: Tuple[int, int]) -> float:
  horizontal_interaction, vertical_interaction = interaction
  n_vars = horizontal_interaction.shape[0] * vertical_interaction.shape[1]
  spin_cfgs = np.array(list(itertools.product(*([[-1, 1]] * n_vars))))
  interaction_partition = 0.0
  for i in range(spin_cfgs.shape[0]):
    spin_cfg = spin_cfgs[i].reshape(grid_shape)
    h_comp = spin_cfg[:, :-1] * horizontal_interaction * spin_cfg[:, 1:] * 2
    v_comp = spin_cfg[:-1] * vertical_interaction * spin_cfg[1:] * 2
    log_interaction_energy = np.sum(h_comp) + np.sum(v_comp)
    interaction_partition += np.exp(log_interaction_energy)

  return interaction_partition


def log_partition(interaction: Tuple[np.ndarray, np.ndarray],
                  grid_shape: Tuple[int, int]) -> float:
  horizontal_interaction, vertical_interaction = interaction
  n_vars = horizontal_interaction.shape[0] * vertical_interaction.shape[1]
  spin_cfgs = np.array(list(itertools.product(*([[-1, 1]] * n_vars))))
  log_interaction_energy_list = []
  for i in range(spin_cfgs.shape[0]):
    spin_cfg = spin_cfgs[i].reshape(grid_shape)
    h_comp = spin_cfg[:, :-1] * horizontal_interaction * spin_cfg[:, 1:] * 2
    v_comp = spin_cfg[:-1] * vertical_interaction * spin_cfg[1:] * 2
    log_interaction_energy = np.sum(h_comp) + np.sum(v_comp)
    log_interaction_energy_list.append(log_interaction_energy)

  log_interaction_energy_list = np.array(log_interaction_energy_list)
  max_log_interaction_energy = np.max(log_interaction_energy_list)
  interaction_partition = np.sum(
      np.exp(log_interaction_energy_list - max_log_interaction_energy))

  return np.log(interaction_partition) + max_log_interaction_energy


def generate_ising_interaction(
    grid_h: int,
    grid_w: int,
    random_seed=None) -> Tuple[np.ndarray, np.ndarray]:
  np.random.seed(random_seed)
  horizontal_interaction = (
      (np.random.randint(0, 2,
                         (grid_h * (grid_w - 1),)) * 2 - 1).astype(float) *
      (np.random.rand(grid_h * (grid_w - 1)) * (5 - 0.05) + 0.05)).reshape(
          grid_h, grid_w - 1)
  vertical_interaction = ((np.random.randint(0, 2, (
      (grid_h - 1) * grid_w,)) * 2 - 1).astype(float) * (np.random.rand(
          (grid_h - 1) * grid_w) * (5 - 0.05) + 0.05)).reshape(
              grid_h - 1, grid_w)
  return horizontal_interaction, vertical_interaction


def ising_dense(ising_grid_h: int, interaction_original: Tuple[np.ndarray,
                                                               np.ndarray],
                interaction_sparsified: Tuple[np.ndarray, np.ndarray],
                covariance: np.ndarray, log_partition_original: float,
                log_partition_new: float) -> float:
  diff_horizontal = interaction_original[0] - interaction_sparsified[0]
  diff_vertical = interaction_original[1] - interaction_sparsified[1]

  kld = 0
  n_spin = covariance.shape[0]
  for i in range(n_spin):
    i_h, i_v = int(i / ising_grid_h), int(i % ising_grid_h)
    for j in range(i, n_spin):
      j_h, j_v = int(j / ising_grid_h), int(j % ising_grid_h)
      if i_h == j_h and abs(i_v - j_v) == 1:
        kld += diff_horizontal[i_h, min(i_v, j_v)] * covariance[i, j]
      elif abs(i_h - j_h) == 1 and i_v == j_v:
        kld += diff_vertical[min(i_h, j_h), i_v] * covariance[i, j]

  return kld * 2 + log_partition_new - log_partition_original
