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

"""Stateful algorithm suggesters with datastores to be used in benchmarks.

Definitions:
------------
- A 'suggester' generates suggestions and update its state based on the provided
suggestion results.

- A 'state' uses a suggester and experimenter to simulate a study. It calls
the suggester multiple times and evaluate the suggestions with the experimenter.
Note that the experimenter may be non-static.
"""

import abc
from typing import Optional, Protocol, Sequence

import attr
from vizier import algorithms as vza
from vizier import pythia
from vizier import pyvizier as vz
from vizier._src.algorithms.policies.designer_policy import InRamDesignerPolicy
from vizier._src.benchmarks.experimenters.experimenter import Experimenter
from vizier._src.benchmarks.experimenters.experimenter_factory import ExperimenterFactory


@attr.define
class PolicySuggester:
  """Wraps a Policy and a datastore/supporter as an algorithm.

  NOTE: Updates to datastore should interface directly with supporter API. The
  policy will simply use the dataset that is in the supporter. This is the most
  faithful replication of the datastore vs suggest interaction in prod. Note
  that the algorithm/policy is updated with new Trials upon the suggest call.

  Typical usage:
    suggester = PolicySuggester(policy, supporter)
    # Trial ids are assigned by the supporter.
    suggestions = suggestion.suggest(...)
    trials = evaluate_and_complete_trials(suggestions)
    suggester.supporter.AddTrials(trials)
  """

  _policy: pythia.Policy = attr.field()
  _local_supporter: pythia.InRamPolicySupporter = attr.field()

  def suggest(self, batch_size: Optional[int]) -> Sequence[vz.Trial]:
    return self._local_supporter.SuggestTrials(
        self._policy, count=batch_size or 1
    )

  @property
  def supporter(self) -> pythia.InRamPolicySupporter:
    return self._local_supporter

  @classmethod
  def from_designer_factory(
      cls,
      problem: vz.ProblemStatement,
      designer_factory: vza.DesignerFactory[vza.Designer],
      supporter: Optional[pythia.InRamPolicySupporter] = None,
      seed: Optional[int] = None,
  ) -> 'PolicySuggester':
    """Initializes a PolicySuggester from Designer."""
    if supporter is None:
      supporter = pythia.InRamPolicySupporter(problem)
    # Wrap as policy and Policy
    policy = InRamDesignerPolicy(
        problem_statement=problem,
        supporter=supporter,
        designer_factory=designer_factory,
        seed=seed,
    )
    return PolicySuggester(policy=policy, local_supporter=supporter)


@attr.define
class BenchmarkState:
  """State of a benchmark run. It is altered via benchmark protocols."""

  experimenter: Experimenter
  algorithm: PolicySuggester


@attr.define(frozen=True)
class BenchmarkStateFactory(abc.ABC):
  """Factory class to generate new BenchmarkState."""

  @abc.abstractmethod
  def __call__(self, seed: Optional[int] = None) -> BenchmarkState:
    """Creates a new instance of BenchmarkFactory."""
    pass


@attr.define(frozen=True)
class ExperimenterDesignerBenchmarkStateFactory(BenchmarkStateFactory):
  """Generate new BenchmarkState from exptr/designer factory."""

  experimenter_factory: ExperimenterFactory
  designer_factory: vza.DesignerFactory[vza.Designer]

  def __call__(self, seed: Optional[int] = None) -> BenchmarkState:
    """Create a BenchmarkState from experimenter and designer factory."""
    experimenter = self.experimenter_factory()
    factory = DesignerBenchmarkStateFactory(
        experimenter=experimenter, designer_factory=self.designer_factory
    )
    return factory(seed=seed)


@attr.define(frozen=True)
class DesignerBenchmarkStateFactory(BenchmarkStateFactory):
  """Factory class to generate new BenchmarkState from designer factory."""

  experimenter: Experimenter
  designer_factory: vza.DesignerFactory[vza.Designer]

  def __call__(self, seed: Optional[int] = None) -> BenchmarkState:
    """Create a BenchmarkState from designer factory."""
    problem = self.experimenter.problem_statement()

    return BenchmarkState(
        experimenter=self.experimenter,
        algorithm=PolicySuggester.from_designer_factory(
            problem=problem,
            designer_factory=self.designer_factory,
            seed=seed,
        ),
    )


class SeededPolicyFactory(Protocol):
  """Factory to generate a policy with a seed."""

  def __call__(self, problem: vz.ProblemStatement, seed: int) -> pythia.Policy:
    ...


@attr.define(frozen=True)
class PolicyBenchmarkStateFactory(BenchmarkStateFactory):
  """Factory class to generate new BenchmarkState from policy factory."""

  experimenter: Experimenter
  policy_factory: SeededPolicyFactory

  def __call__(self, seed: Optional[int] = None) -> BenchmarkState:
    """Create a BenchmarkState from policy factory."""
    problem = self.experimenter.problem_statement()
    return BenchmarkState(
        experimenter=self.experimenter,
        algorithm=PolicySuggester(
            policy=self.policy_factory(problem, seed),
            local_supporter=pythia.InRamPolicySupporter(problem),
        ),
    )
