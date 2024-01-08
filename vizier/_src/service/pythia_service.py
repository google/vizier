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

"""Separate Pythia service for handling algorithmic logic."""
from typing import Optional

from absl import logging
import attr
import grpc
from vizier import pythia
from vizier._src.service import policy_factory as service_policy_factory
from vizier._src.service import pythia_service_pb2
from vizier._src.service import pythia_service_pb2_grpc
from vizier._src.service import service_policy_supporter
from vizier._src.service import stubs_util
from vizier._src.service import types
from vizier.service import pyvizier as vz

from google.protobuf import empty_pb2


@attr.define
class PythiaServicer(pythia_service_pb2_grpc.PythiaServiceServicer):
  """Implements the GRPC functions outlined in pythia_service.proto."""

  # Can be either an actual VizierService object or a stub. An actual
  # VizierService should be passed only for local testing/local development.
  _vizier_service: Optional[types.VizierService] = attr.field(
      init=True, default=None
  )
  # Factory for creating policies. Defaulted to OSS Vizier-specific policies,
  # but allows use of external package (e.g. PyGlove) poliices.
  _policy_factory: pythia.PolicyFactory = attr.field(
      init=True, factory=service_policy_factory.DefaultPolicyFactory
  )

  def __attrs_post_init__(self):
    try:
      # If Jax is installed, always use float64 for all policies.
      import jax  # pylint:disable=g-import-not-at-top

      jax.config.update('jax_enable_x64', True)
    except ImportError:
      pass

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

  def Ping(
      self,
      request: empty_pb2.Empty,
      context: Optional[grpc.ServicerContext] = None,
  ) -> empty_pb2.Empty:
    return empty_pb2.Empty()
