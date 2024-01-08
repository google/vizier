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

"""Tuner implementation based on Open Source Vizier."""

from concurrent import futures
import os
import threading
from typing import Optional, Union

from absl import logging
import attr
import grpc
import portpicker
import pyglove as pg
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.pyglove import algorithms
from vizier._src.pyglove import backend
from vizier._src.pyglove import client
from vizier._src.pyglove import converters
from vizier._src.service import clients as pyvizier_clients
from vizier._src.service import constants
from vizier._src.service import policy_factory as service_policy_factory_lib
from vizier._src.service import pythia_service
from vizier._src.service import pythia_service_pb2_grpc
from vizier._src.service import resources
from vizier._src.service import service_policy_supporter
from vizier._src.service import stubs_util
from vizier._src.service import types as vizier_types
from vizier._src.service import vizier_client
from vizier.client import client_abc
from vizier.service import pyvizier as svz

from google.protobuf import empty_pb2

BuiltinAlgorithm = algorithms.BuiltinAlgorithm
ExpandedStudyName = client.ExpandedStudyName
PolicyCache = client.PolicyCache
StudyKey = client.StudyKey


class PyGlovePolicyFactory(pythia.PolicyFactory):
  """PolicyFactory for OSSVizierTuner."""

  def __init__(self, policy_cache: PolicyCache):
    self._policy_cache = policy_cache

  def __call__(
      self, problem_statement, algorithm, policy_supporter, study_name
  ):
    study_resource = resources.StudyResource.from_name(study_name)
    study_key = StudyKey(
        study_resource.owner_id, ExpandedStudyName(study_resource.study_id)
    )
    if study_key in self._policy_cache:
      logging.info(
          'StudyKey %s was found in cache. Using it as the policy.', study_key
      )
      return self._policy_cache[study_key]

    # Use default Vizier algorithms if not using PyGlove poliices.
    logging.info(
        'StudyKey %s was not found in cache. Using default policy factory.'
    )

    default_policy_factory = service_policy_factory_lib.DefaultPolicyFactory()
    return default_policy_factory(
        problem_statement, algorithm, policy_supporter, study_name
    )


@attr.define
class _VizierServices:
  """Vizier services hub.

  This class is intended to be used as a singleton and not directly exposed to
  the user. Instead, users should always call `vizier.pyglove.init` to set up
  the vizier services stub, which calls `_VizierServices.use_vizier_service` and
  `_VizierServices.set_pythia_port` underlying.
  """

  _vizier_endpoint: Optional[str] = None
  _vizier_service: Optional[vizier_types.VizierService] = None
  _pythia_port: Optional[int] = None
  _pythia_server: Optional[grpc.Server] = None
  _pythia_servicer: Optional[pythia_service.PythiaServicer] = None

  def reset_for_testing(self) -> None:
    """Resets the services for testing purpose."""
    self._vizier_endpoint = None
    self._vizier_service = None
    self._pythia_port = None
    self._pythia_server = None
    self._pythia_servicer = None

  def use_vizier_service(self, endpoint: Optional[None]) -> None:
    """Uses vizier service specified by endpoint."""
    endpoint = endpoint or constants.NO_ENDPOINT
    if self._vizier_endpoint is not None and self._vizier_endpoint != endpoint:
      raise ValueError(
          'Cannot use different vizier endpoints in the same process. '
          f'Previous={self._vizier_endpoint}, New={endpoint}.'
      )
    self._vizier_endpoint = endpoint
    if self._vizier_service is not None:
      return
    if endpoint != constants.NO_ENDPOINT:
      pyvizier_clients.environment_variables.server_endpoint = (
          self._vizier_endpoint
      )
    self._vizier_service = vizier_client.create_vizier_servicer_or_stub()

  @property
  def vizier_service(self) -> vizier_types.VizierService:
    """Returns current vizier service."""
    assert self._vizier_service, 'call `use_vizier_service` first.'
    return self._vizier_service

  def set_pythia_port(self, pythia_port: Optional[int] = None) -> None:
    if (
        self._pythia_port is not None
        and pythia_port is not None
        and self._pythia_port != pythia_port
    ):
      raise ValueError(
          'Cannot use different pythia ports in the same process. '
          f'Previous={self._pythia_port}, New={pythia_port}'
      )
    self._pythia_port = pythia_port or portpicker.pick_unused_port()

  @property
  def pythia_endpoint(self) -> str:
    """Returns the endpoint for Pythia service."""
    assert self._pythia_port is not None, 'Please call `set_pythia_port` first.'
    return f'{os.uname()[1]}:{self._pythia_port}'

  def start_pythia_service(self, policy_cache: PolicyCache) -> None:
    """Start pythia service with a policy cache."""
    if self._pythia_server is not None:
      return

    policy_factory = PyGlovePolicyFactory(policy_cache)
    self._pythia_servicer = pythia_service.PythiaServicer(
        self._vizier_service, policy_factory
    )
    self._pythia_server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pythia_service_pb2_grpc.add_PythiaServiceServicer_to_server(
        self._pythia_servicer, self._pythia_server
    )
    self._pythia_server.add_insecure_port(self.pythia_endpoint)
    self._pythia_server.start()


# Global vizier services hub.
_services = _VizierServices()


class _OSSVizierTuner(client.VizierTuner):
  """OSS Vizier tuner for pyglove."""

  def get_tuner_id(self, algorithm: pg.DNAGenerator) -> str:
    """See parent class."""
    del algorithm
    # We use hostname plus thread ID as tuner ID, so we could test
    # distributed tuning scenarios using multi-threading.
    return f'{threading.get_ident()}@{_services.pythia_endpoint}'

  def _start_pythia_service(self, policy_cache: PolicyCache) -> None:
    """See parent class."""
    _services.start_pythia_service(policy_cache)

  def load_prior_study(self, resource_name: str) -> client_abc.StudyInterface:
    """See parent class."""
    return pyvizier_clients.Study.from_resource_name(resource_name)

  @classmethod
  def load_study(
      cls, owner: str, name: ExpandedStudyName
  ) -> client_abc.StudyInterface:
    """See parent class."""
    return pyvizier_clients.Study.from_owner_and_id(owner, name)

  def _configure_algorithm(
      self, study_config: svz.StudyConfig, algorithm: pg.DNAGenerator
  ) -> None:
    """Configure algorithm for a study."""
    if isinstance(algorithm, algorithms.BuiltinAlgorithm):
      study_config.algorithm = algorithm.name
    else:
      study_config.algorithm = 'EXTERNAL_PYTHIA_SERVICE'
    study_config.pythia_endpoint = _services.pythia_endpoint

  def create_study(
      self,
      problem: vz.ProblemStatement,
      converter: converters.VizierConverter,
      owner: str,
      name: str,
      algorithm: pg.DNAGenerator,
      stopping_policy: Optional[pg.tuning.EarlyStoppingPolicy] = None,
  ) -> client_abc.StudyInterface:
    """See parent class."""
    study_config = svz.StudyConfig.from_problem(problem)
    if converter.vizier_conversion_error:
      study_config.observation_noise = svz.ObservationNoise.HIGH
    self._configure_algorithm(study_config, algorithm)
    logging.info(
        'Created OSS Vizier study with owner: %s, name: %s', owner, name
    )
    return pyvizier_clients.Study.from_study_config(
        study_config, owner=owner, study_id=name
    )

  def get_group_id(self, group_id: Union[None, int, str] = None) -> str:
    """See parent class."""
    if group_id is None:
      hostname = os.uname()[1]
      thread_id = threading.get_ident()
      return f'{thread_id}@{hostname}'
    elif isinstance(group_id, int):
      return f'group:{group_id}'
    elif isinstance(group_id, str):
      return group_id  # pytype: disable=bad-return-type

  def ping_tuner(self, tuner_id: str) -> bool:
    # We treat `tuner_id` as the Pythia endpoint.
    assert '@' in tuner_id, tuner_id
    pythia_endpoint = tuner_id.split('@')[1]
    try:
      stubs_util.create_pythia_server_stub(pythia_endpoint, timeout=3).Ping(
          empty_pb2.Empty()
      )
      return True
    except (grpc.RpcError, grpc.FutureTimeoutError):
      return False

  def pythia_supporter(
      self, study: client_abc.StudyInterface
  ) -> pythia.PolicySupporter:
    return service_policy_supporter.ServicePolicySupporter(
        study.resource_name, _services.vizier_service
    )

  def use_pythia_for_study(self, study: client_abc.StudyInterface) -> None:
    pythia_endpoint = _services.pythia_endpoint
    metadata = svz.StudyConfig.pythia_endpoint_metadata(pythia_endpoint)
    study.update_metadata(metadata)


def init(
    study_prefix: Optional[str] = None,
    vizier_endpoint: Optional[str] = None,
    pythia_port: Optional[int] = None,
) -> None:
  """Init OSS Vizier backend.

  Args:
    study_prefix: An optional string that will be used as the prefix for the
      study names created by `pg.sample` throughout the application. This allows
      users to change the study names across multiple runs of the same binary
      through this single venue, instead of modifying the `name` argument of
      every `pg.sample` invocation.
    vizier_endpoint: An optional string in format of <hostname>:<port>, as the
      Vizier service address to connect to. If None, an in-process Vizier
      service will be created for local tuning scenarios.
    pythia_port: An optional port used for hosting the Pythia service. If None,
      the port will be automatically picked.
  """
  _services.use_vizier_service(vizier_endpoint)
  _services.set_pythia_port(pythia_port)
  backend.VizierBackend.use_study_prefix(study_prefix)
  pg.tuning.set_default_backend('oss_vizier')


@pg.tuning.add_backend('oss_vizier')
class OSSVizierBackend(backend.VizierBackend):
  """PyGlove backend that uses OSS Vizier."""

  tuner_cls = _OSSVizierTuner
