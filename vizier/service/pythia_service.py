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

"""Separate Pythia service for handling algorithmic logic."""
# pylint:disable=g-import-not-at-top
from typing import Optional, Protocol
from absl import logging
import grpc

from vizier import pythia
from vizier._src.algorithms.policies import designer_policy as dp
from vizier.service import pythia_service_pb2
from vizier.service import pythia_service_pb2_grpc
from vizier.service import pyvizier as vz
from vizier.service import service_policy_supporter
from vizier.service import stubs_util
from vizier.service import types


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


def default_policy_factory(
    problem_statement: vz.ProblemStatement,
    algorithm: str,
    policy_supporter: pythia.PolicySupporter,
    study_name: str,
) -> pythia.Policy:
  """Creates a policy."""
  del study_name

  if algorithm in ('ALGORITHM_UNSPECIFIED', 'RANDOM_SEARCH'):
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


class PythiaServicer(pythia_service_pb2_grpc.PythiaServiceServicer):
  """Implements the GRPC functions outlined in pythia_service.proto."""

  def __init__(
      self,
      vizier_service: Optional[types.VizierService] = None,
      policy_factory: PolicyFactory = default_policy_factory,
  ):
    """Initialization.

    Args:
      vizier_service: Can be either an actual VizierService object or a stub. An
        actual VizierService should be passed only for local testing/local
        development.
      policy_factory: Factory for creating policies. Defaulted to OSS
        Vizier-specific policies, but allows use of external package (e.g.
        PyGlove) poliices.
    """
    self._vizier_service = vizier_service
    self._policy_factory = policy_factory

  def connect_to_vizier(self, endpoint: str) -> None:
    """Only needs to be called if VizierService wasn't passed in init."""
    if self._vizier_service:
      raise ValueError('Vizier Service was already set:', self._vizier_service)
    self._vizier_service = stubs_util.create_vizier_server_stub(endpoint)

  def Suggest(
      self,
      request: pythia_service_pb2.SuggestRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> pythia_service_pb2.SuggestDecision:
    """Performs Suggest RPC call."""
    # Setup Policy Supporter.
    study_config = vz.SuggestConverter.from_request_proto(request).study_config
    study_name = request.study_descriptor.guid
    policy_supporter = service_policy_supporter.ServicePolicySupporter(
        study_name, self._vizier_service
    )
    pythia_policy = self._policy_factory(
        study_config, request.algorithm, policy_supporter, study_name
    )

    # Perform algorithmic computation.
    suggest_request = vz.SuggestConverter.from_request_proto(request)
    try:
      suggest_decision = pythia_policy.suggest(suggest_request)
    # Leaving a broad catch for now since Pythia can raise any exception.
    # TODO: Be more specific about exception raised,
    # e.g. AttributeError, ModuleNotFoundError, SyntaxError
    except Exception as e:  # pylint: disable=broad-except
      logging.error(
          'Failed to request trials from Pythia for request: %s', request
      )
      raise RuntimeError('Pythia has encountered an error: ' + str(e)) from e

    return vz.SuggestConverter.to_decision_proto(suggest_decision)

  def EarlyStop(
      self,
      request: pythia_service_pb2.EarlyStopRequest,
      context: Optional[grpc.ServicerContext] = None,
  ) -> pythia_service_pb2.EarlyStopDecisions:
    """Performs EarlyStop RPC call."""
    # Setup Policy Supporter.
    study_config = vz.EarlyStopConverter.from_request_proto(
        request
    ).study_config
    study_name = request.study_descriptor.guid
    policy_supporter = service_policy_supporter.ServicePolicySupporter(
        study_name, self._vizier_service
    )
    pythia_policy = self._policy_factory(
        study_config, request.algorithm, policy_supporter, study_name
    )

    # Perform algorithmic computation.
    early_stop_request = vz.EarlyStopConverter.from_request_proto(request)
    early_stopping_decisions = pythia_policy.early_stop(early_stop_request)

    return vz.EarlyStopConverter.to_decisions_proto(early_stopping_decisions)
