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

"""Abstraction for Vizier Tuner."""

import abc
import threading
from typing import Dict, NewType, Optional, Union

import attrs
import pyglove as pg
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.pyglove import converters
from vizier._src.pyglove import pythia as pg_pythia
from vizier.client import client_abc


# Pyglove-Vizier interactions only accept ExpandedStudyName. It is distinguished
# from the name that is passed into Backend.__init__ which does not include
# prefix.
ExpandedStudyName = NewType('ExpandedStudyName', str)


@attrs.define(eq=True, frozen=True)
class StudyKey:
  """Simple immutable structure used for looking up policies in a dict."""
  _owner: str
  _name: ExpandedStudyName


PolicyCache = Dict[StudyKey, pg_pythia.TunerPolicy]


class VizierTuner(abc.ABC):
  """Abstraction for Vizier tuner.

  Each platform (e.g. Google, OSS, Vertex) should have their own implementation
  of this abstraction.
  """
  # NOTE(chansoo): All of the methods can technically become a classmethod.

  def __init__(self):
    self._pythia_lock = threading.Lock()

  @abc.abstractmethod
  def get_tuner_id(self, algorithm: pg.DNAGenerator) -> str:
    """Get identifier of this tuner instance."""

  def start_pythia_service(self, policy_cache: PolicyCache) -> None:
    """Start pythia service on current machine."""
    with self._pythia_lock:
      self._start_pythia_service(policy_cache)

  @abc.abstractmethod
  def _start_pythia_service(
      self, policy_cache: dict[StudyKey, pg_pythia.TunerPolicy]
  ) -> None:
    """Starts pythia service _only if_ there is no pythia service running.

    Args:
      policy_cache: The pythia service will have the reference to policies
        created by the backend. It is expected to use the study keys to find
        the policies, instead of creating new ones from scratch.
    """

  @abc.abstractmethod
  def load_prior_study(self, resource_name: str) -> client_abc.StudyInterface:
    """Loads prior study identified by the resource name."""

  @abc.abstractmethod
  def create_study(
      self,
      problem: vz.ProblemStatement,
      converter: converters.VizierConverter,
      owner: str,
      name: str,
      algorithm: pg.DNAGenerator,
      stopping_policy: Optional[pg.tuning.EarlyStoppingPolicy] = None,
  ) -> client_abc.StudyInterface:
    """Creates a new study."""

  @abc.abstractmethod
  def get_group_id(self, group_id: Union[None, int, str] = None) -> str:
    """Get worker ID."""

  @abc.abstractmethod
  def ping_tuner(self, tuner_id: str) -> bool:
    """See if the tuner is alive."""

  @abc.abstractmethod
  def pythia_supporter(
      self, study: client_abc.StudyInterface
  ) -> pythia.PolicySupporter:
    """Creates a pythia policy supporter for this study."""

  @abc.abstractmethod
  def use_pythia_for_study(self, study: client_abc.StudyInterface) -> None:
    """Uses current Pythia service to serve the input study."""

  @classmethod
  @abc.abstractmethod
  def load_study(
      cls, owner: str, name: ExpandedStudyName
  ) -> client_abc.StudyInterface:
    """Loads a study identified by (owner, name) pair."""
