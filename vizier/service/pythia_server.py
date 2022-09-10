"""Separate Pythia service for handling algorithmic logic."""
from typing import Optional
from absl import logging
import grpc

from vizier import pythia

from vizier._src.algorithms.designers import bocs
from vizier._src.algorithms.designers import emukit
from vizier._src.algorithms.designers import grid
from vizier._src.algorithms.designers import harmonica
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.evolution import nsga2
from vizier._src.algorithms.policies import designer_policy as dp
from vizier._src.algorithms.policies import random_policy

from vizier.service import pythia_service_pb2
from vizier.service import pythia_service_pb2_grpc
from vizier.service import pyvizier as vz
from vizier.service import service_policy_supporter
from vizier.service import study_pb2


def policy_creator(
    problem_statement: vz.ProblemStatement,
    algorithm: study_pb2.StudySpec.Algorithm,
    policy_supporter: service_policy_supporter.ServicePolicySupporter
) -> pythia.Policy:
  """Creates a policy."""
  if algorithm in (study_pb2.StudySpec.Algorithm.ALGORITHM_UNSPECIFIED,
                   study_pb2.StudySpec.Algorithm.RANDOM_SEARCH):
    return random_policy.RandomPolicy(policy_supporter)
  elif algorithm == study_pb2.StudySpec.Algorithm.QUASI_RANDOM_SEARCH:
    return dp.PartiallySerializableDesignerPolicy(
        problem_statement, policy_supporter,
        quasi_random.QuasiRandomDesigner.from_problem)
  elif algorithm == study_pb2.StudySpec.Algorithm.GRID_SEARCH:
    return dp.PartiallySerializableDesignerPolicy(
        problem_statement, policy_supporter,
        grid.GridSearchDesigner.from_problem)
  elif algorithm == study_pb2.StudySpec.Algorithm.NSGA2:
    return dp.PartiallySerializableDesignerPolicy(problem_statement,
                                                  policy_supporter,
                                                  nsga2.create_nsga2)
  elif algorithm == study_pb2.StudySpec.Algorithm.EMUKIT_GP_EI:
    return dp.DesignerPolicy(policy_supporter, emukit.EmukitDesigner)
  elif algorithm == study_pb2.StudySpec.Algorithm.BOCS:
    return dp.DesignerPolicy(policy_supporter, bocs.BOCSDesigner)
  elif algorithm == study_pb2.StudySpec.Algorithm.HARMONICA:
    return dp.DesignerPolicy(policy_supporter, harmonica.HarmonicaDesigner)
  else:
    raise ValueError(
        f'Algorithm {study_pb2.StudySpec.Algorithm.Name(algorithm)} '
        'is not registered.')


class PythiaService(pythia_service_pb2_grpc.PythiaServiceServicer):
  """Implements the GRPC functions outlined in pythia_service.proto."""

  def __init__(self):
    self._vizier_service_endpoint: Optional[str] = None

  def connect_to_vizier(self, vizier_service_endpoint: str) -> None:
    self._vizier_service_endpoint = vizier_service_endpoint

  def Suggest(
      self,
      request: pythia_service_pb2.SuggestRequest,
      context: Optional[grpc.ServicerContext] = None
  ) -> pythia_service_pb2.SuggestDecision:
    """Performs Suggest RPC call."""
    # Setup Policy Supporter.
    study_config = vz.SuggestConverter.from_request_proto(request).study_config
    policy_supporter = service_policy_supporter.ServicePolicySupporter(
        request.study_descriptor.guid)
    policy_supporter.connect_to_vizier(self._vizier_service_endpoint)
    pythia_policy = policy_creator(study_config, request.algorithm,
                                   policy_supporter)

    # Perform algorithmic computation.
    suggest_request = vz.SuggestConverter.from_request_proto(request)
    try:
      suggest_decision = pythia_policy.suggest(suggest_request)
    # Leaving a broad catch for now since Pythia can raise any exception.
    # TODO: Be more specific about exception raised,
    # e.g. AttributeError, ModuleNotFoundError, SyntaxError
    except Exception as e:  # pylint: disable=broad-except
      logging.error('Failed to request trials from Pythia for request: %s',
                    request)
      raise RuntimeError('Pythia has encountered an error: ' + str(e)) from e

    return vz.SuggestConverter.to_decision_proto(suggest_decision)

  def EarlyStop(
      self,
      request: pythia_service_pb2.EarlyStopRequest,
      context: Optional[grpc.ServicerContext] = None
  ) -> pythia_service_pb2.EarlyStopDecisions:
    """Performs EarlyStop RPC call."""
    # Setup Policy Supporter.
    study_config = vz.EarlyStopConverter.from_request_proto(
        request).study_config
    policy_supporter = service_policy_supporter.ServicePolicySupporter(
        request.study_descriptor.guid)
    policy_supporter.connect_to_vizier(self._vizier_service_endpoint)
    pythia_policy = policy_creator(study_config, request.algorithm,
                                   policy_supporter)

    # Perform algorithmic computation.
    early_stop_request = vz.EarlyStopConverter.from_request_proto(request)
    early_stopping_decisions = pythia_policy.early_stop(early_stop_request)

    return vz.EarlyStopConverter.to_decisions_proto(early_stopping_decisions)
