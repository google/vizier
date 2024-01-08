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

"""Sparse experimenter.

The SparseExperimenter class wraps another experimenter and expands its search
space with placeholder parameters that don't impact the evaluation.

This experimenter allows testing that a designer/policy is able to optimize an
objective function when only a subset of its parameters affect the function
values.
"""

import copy
from typing import Optional, Sequence

import attr
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter


# These default values are used for defining the placeholder parameters.
_DEFAULT_FLOAT_BOUNDS = (-5.0, 5.0)
_DEFAULT_INT_BOUNDS = (-5, 5)

_DEFAULT_DISCRETE_FEASIBLE_VALUES = (0, 1, 2, 3, 4)
_DEFAULT_CATEGORICAL_FEASIBLE_VALUES = ('a', 'b', 'c', 'd', 'e', 'f')


class SparseExperimenter(experimenter.Experimenter):
  """Sparse experimenter.

  The sparse experimenter allows expanding an experimenter's search space with
  additional placeholder parameters that don't have affect during evaluation.
  """

  def __init__(
      self,
      experiment: experimenter.Experimenter,
      search_space: vz.SearchSpace,
      prefix: str = '_SPARSE',
  ):
    """Initializes a sparse experimenter.

    The sparse experimenter is constructed by adding a copy of parameters from
    $search_space to the experimenter search space. Newly added parameter names
    are prefixed with $prefix.

    Arguments:
      experiment: An experimenter to add sparse parameters to.
      search_space: A search space to use for the sparse parameters.
      prefix: The sparse parameter name prefix (shouldn't already exist in the
        search space of 'experiment').
    """
    super().__init__()
    self._sparse_param_prefix = prefix
    self._experimenter = experiment
    self._search_space = copy.deepcopy(search_space)
    self._prefix = prefix

    problem = experiment.problem_statement()
    for pc in search_space.parameters:
      # Add a copy of the parameter config with a modified name.
      problem.search_space.add(attr.evolve(pc, name=prefix + '_' + pc.name))
    self._problem_statement = problem

  def evaluate(self, suggestions: Sequence[vz.Trial]):
    """Evaluates and completes trials on sparse subset parameters."""
    # Evaluates trials on non-sparse parameters only determined by the prefix.
    original_params = []
    for trial in suggestions:
      original_params.append(trial.parameters)
      trial.parameters = {
          param_name: param_value
          for param_name, param_value in trial.parameters.items()
          if not param_name.startswith(self._sparse_param_prefix)
      }
    self._experimenter.evaluate(suggestions)
    # Restores the original parameters with the sparse parameters.
    for trial, params in zip(suggestions, original_params):
      trial.parameters = params

  def problem_statement(self) -> vz.ProblemStatement:
    return copy.deepcopy(self._problem_statement)

  @classmethod
  def create(
      cls,
      experiment: experimenter.Experimenter,
      float_count: int,
      int_count: int,
      discrete_count: int,
      categorical_count: int,
      float_min_value: Optional[float] = None,
      float_max_value: Optional[float] = None,
      int_min_value: Optional[int] = None,
      int_max_value: Optional[int] = None,
      discrete_feasible_values: Optional[list[int]] = None,
      categorical_feasible_values: Optional[list[str]] = None,
  ) -> 'SparseExperimenter':
    """Creates a sparse experimenter with different parameter types.

    Arguments:
      experiment: The experimenter to add sparsity to.
      float_count: The number of FLOAT sparse parameters to add.
      int_count: The number of INT sparse parameters to add.
      discrete_count: The number of DISCRETE sparse parameters to add.
      categorical_count: The number of CATEGORICAL sparse parameters to add.
      float_min_value: The FLOAT sparse parameter min value.
      float_max_value: The INTEGER sparse parameter max value.
      int_min_value: The FLOAT sparse parameter min value.
      int_max_value: The INTEGER sparse parameter max value.
      discrete_feasible_values: The DISCRETE sparse parameter feasible values.
      categorical_feasible_values: The CTEGORICAL sparse parameter feasible
        values.

    Returns:
      The sparse experimenter.
    """
    sparse_search_space = vz.SearchSpace()
    for idx in range(float_count):
      sparse_search_space.root.add_float_param(
          name='FLOAT' + str(idx),
          min_value=float_min_value or _DEFAULT_FLOAT_BOUNDS[0],
          max_value=float_max_value or _DEFAULT_FLOAT_BOUNDS[1],
      )
    for idx in range(int_count):
      sparse_search_space.root.add_int_param(
          name='INT' + str(idx),
          min_value=float_min_value or _DEFAULT_INT_BOUNDS[0],
          max_value=float_max_value or _DEFAULT_INT_BOUNDS[1],
      )
    for idx in range(discrete_count):
      sparse_search_space.root.add_discrete_param(
          name='DISCRETE' + str(idx),
          feasible_values=categorical_feasible_values
          or _DEFAULT_DISCRETE_FEASIBLE_VALUES,
      )
    for idx in range(categorical_count):
      sparse_search_space.root.add_categorical_param(
          name='CATEGORICAL' + str(idx),
          feasible_values=categorical_feasible_values
          or _DEFAULT_CATEGORICAL_FEASIBLE_VALUES,
      )
    return cls(experiment, sparse_search_space)
