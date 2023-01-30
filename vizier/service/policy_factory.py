# Copyright 2023 Google LLC.
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

"""Policy Factory API."""
# pylint:disable=g-import-not-at-top
import functools
import random
from typing import Protocol

from vizier import pythia
from vizier._src.algorithms.policies import designer_policy as dp
from vizier.service import pyvizier as vz


class PolicyFactory(Protocol):
  """Protocol (PEP-544) for a Policy Factory."""

  def __call__(
      self,
      problem_statement: vz.ProblemStatement,
      algorithm: str,
      policy_supporter: pythia.PolicySupporter,
      study_name: str,
  ) -> pythia.Policy:
    """Creates a Pythia Policy."""


class DefaultPolicyFactory(PolicyFactory):
  """Default Policy Factory used in Pythia."""

  def __call__(
      self,
      problem_statement: vz.ProblemStatement,
      algorithm: str,
      policy_supporter: pythia.PolicySupporter,
      study_name: str,
  ) -> pythia.Policy:
    del study_name
    if algorithm in (
        'DEFAULT',
        'ALGORITHM_UNSPECIFIED',
        'GAUSSIAN_PROCESS_BANDIT',
    ):
      from vizier._src.algorithms.designers import gp_bandit

      return dp.DesignerPolicy(policy_supporter, gp_bandit.VizierGPBandit)
    elif algorithm == 'RANDOM_SEARCH':
      from vizier._src.algorithms.policies import random_policy

      return random_policy.RandomPolicy(policy_supporter)
    elif algorithm == 'QUASI_RANDOM_SEARCH':
      from vizier._src.algorithms.designers import quasi_random

      return dp.PartiallySerializableDesignerPolicy(
          problem_statement,
          policy_supporter,
          quasi_random.QuasiRandomDesigner.from_problem,
      )
    elif algorithm == 'GRID_SEARCH':
      from vizier._src.algorithms.designers import grid

      return dp.PartiallySerializableDesignerPolicy(
          problem_statement,
          policy_supporter,
          grid.GridSearchDesigner.from_problem,
      )
    elif algorithm == 'SHUFFLED_GRID_SEARCH':
      from vizier._src.algorithms.designers import grid

      shuffle_seed = random.randint(0, 2147483647)
      grid_factory = functools.partial(
          grid.GridSearchDesigner.from_problem, shuffle_seed=shuffle_seed
      )
      return dp.PartiallySerializableDesignerPolicy(
          problem_statement,
          policy_supporter,
          grid_factory,
      )
    elif algorithm == 'NSGA2':
      from vizier._src.algorithms.evolution import nsga2

      return dp.PartiallySerializableDesignerPolicy(
          problem_statement, policy_supporter, nsga2.create_nsga2
      )
    elif algorithm == 'EMUKIT_GP_EI':
      from vizier._src.algorithms.designers import emukit

      return dp.DesignerPolicy(policy_supporter, emukit.EmukitDesigner)
    elif algorithm == 'BOCS':
      from vizier._src.algorithms.designers import bocs

      return dp.DesignerPolicy(policy_supporter, bocs.BOCSDesigner)
    elif algorithm == 'HARMONICA':
      from vizier._src.algorithms.designers import harmonica

      return dp.DesignerPolicy(policy_supporter, harmonica.HarmonicaDesigner)
    elif algorithm == 'CMA_ES':
      from vizier._src.algorithms.designers import cmaes

      return dp.PartiallySerializableDesignerPolicy(
          problem_statement, policy_supporter, cmaes.CMAESDesigner
      )
    else:
      raise ValueError(f'Algorithm {algorithm} is not registered.')
