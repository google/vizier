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

"""Wrapper around optproblems PyPI library for multiobjective problems.

See documentation:
https://ls11-www.cs.tu-dortmund.de/people/swessing/optproblems/doc/index.html#

WFG benchmark was used in MultiObjective TPE:
https://www.jair.org/index.php/jair/article/view/13188
"""

import json
import attrs
from optproblems import dtlz
from optproblems import wfg
from optproblems import zdt
from vizier import pyvizier as vz
from vizier._src.benchmarks.experimenters import experimenter_factory
from vizier._src.benchmarks.experimenters import numpy_experimenter


def _DefaultOptProblemStatement(
    dimension: int,
    num_objectives: int,
    *,
    min_value: float = 0.0,
    max_value: float = 1.0,
):
  """Returns default optproblems ProblemStatement."""
  problem = vz.ProblemStatement()
  for n in range(num_objectives):
    metric = vz.MetricInformation(
        name=f"f{n}", goal=vz.ObjectiveMetricGoal.MINIMIZE
    )
    problem.metric_information.append(metric)
  for d in range(dimension):
    problem.search_space.root.add_float_param(f"x{d}", min_value, max_value)
  return problem


EXPERIMENTER_FACTORY_KEY = "experimenter_factory"


@attrs.define
class WFGExperimenterFactory(
    experimenter_factory.SerializableExperimenterFactory
):
  """Multiobjective problem with variable number of objectives and dimensions.

  Reference:
    Huband, S.; Hingston, P.; Barone, L.; While, L. (2006). A review of
    multiobjective test problems and a scalable test problem toolkit. IEEE
    Transactions on Evolutionary Computation, vol.10, no.5, pp. 477-506.
  """

  name: str = attrs.field(
      default="", validator=attrs.validators.instance_of(str)
  )
  dim: int = attrs.field(
      default=1,
      validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
  )
  num_objectives: int = attrs.field(
      default=2,
      validator=[attrs.validators.instance_of(int), attrs.validators.ge(2)],
  )

  def __attrs_post_init__(self):
    # k = "Position-related parameters". Must be divisible by (num_obj-1).
    k = self.num_objectives - 1
    if (self.dim - k) % 2 != 0:
      raise ValueError(
          f"dimensions - k must be even, got {self.dim - k} for k={k}."
      )
    self.k = k

  def __call__(self) -> numpy_experimenter.MultiObjectiveNumpyExperimenter:
    optprob_factory = getattr(wfg, self.name, None)
    if optprob_factory is None:
      raise ValueError(f"{self.name} is not a valid WFG problem in wfg.py")
    optprob: wfg.WFGBaseProblem = optprob_factory(
        self.num_objectives, self.dim, self.k
    )
    impl = optprob.objective_function
    problem = _DefaultOptProblemStatement(self.dim, self.num_objectives)
    return numpy_experimenter.MultiObjectiveNumpyExperimenter(impl, problem)

  def dump(self) -> vz.Metadata:
    metadata = vz.Metadata()
    metadata_dict = {
        "name": self.name,
        "dim": self.dim,
        "num_objectives": self.num_objectives,
    }
    metadata[EXPERIMENTER_FACTORY_KEY] = json.dumps(metadata_dict)
    return metadata

  @classmethod
  def recover(cls, metadata: vz.Metadata) -> "WFGExperimenterFactory":
    metadata_dict = json.loads(metadata[EXPERIMENTER_FACTORY_KEY])
    return cls(**metadata_dict)


@attrs.define
class DTLZExperimenterFactory(
    experimenter_factory.SerializableExperimenterFactory
):
  """Multiobjective problem with variable number of objectives and dimensions.

  Reference:
    K. Deb, L. Thiele, M. Laumanns, E. Zitzler, A. Abraham, L. Jain, and
    R. Goldberg. Scalable test problems for evolutionary multi-objective
    optimization. Evolutionary Multiobjective Optimization, Springer-Verlag,
    pp. 105-145, 2005.
  """

  name: str = attrs.field(
      default="", validator=attrs.validators.instance_of(str)
  )
  dim: int = attrs.field(
      default=1,
      validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
  )
  num_objectives: int = attrs.field(
      default=2,
      validator=[attrs.validators.instance_of(int), attrs.validators.ge(2)],
  )

  def __call__(self) -> numpy_experimenter.MultiObjectiveNumpyExperimenter:
    optprob_factory = getattr(dtlz, self.name, None)
    if optprob_factory is None:
      raise ValueError(f"{self.name} is not a valid DTLZ problem in dtlz.py")
    optprob: dtlz.DTLZBaseProblem = optprob_factory(
        self.num_objectives, self.dim
    )
    impl = optprob.objective_function
    problem = _DefaultOptProblemStatement(self.dim, self.num_objectives)
    return numpy_experimenter.MultiObjectiveNumpyExperimenter(impl, problem)

  def dump(self) -> vz.Metadata:
    metadata = vz.Metadata()
    metadata_dict = {
        "name": self.name,
        "dim": self.dim,
        "num_objectives": self.num_objectives,
    }
    metadata[EXPERIMENTER_FACTORY_KEY] = json.dumps(metadata_dict)
    return metadata

  @classmethod
  def recover(cls, metadata: vz.Metadata) -> "DTLZExperimenterFactory":
    metadata_dict = json.loads(metadata[EXPERIMENTER_FACTORY_KEY])
    return cls(**metadata_dict)


@attrs.define
class ZDTExperimenterFactory(
    experimenter_factory.SerializableExperimenterFactory
):
  """Multiobjective problem with two objectives and d-dimensions.

  Reference:
    E. Zitzler, K. Deb, and L. Thiele. Comparison of multiobjective
    evolutionary algorithms: Empirical results. Evolutionary Computation, vol.
    8, no. 2,pp. 173-195, 2000.
  """

  name: str = attrs.field(
      default="", validator=attrs.validators.instance_of(str)
  )
  dim: int = attrs.field(
      default=1,
      validator=[attrs.validators.instance_of(int), attrs.validators.gt(0)],
  )

  def __call__(self) -> numpy_experimenter.MultiObjectiveNumpyExperimenter:
    optprob_factory = getattr(zdt, self.name, None)
    if optprob_factory is None:
      raise ValueError(f"{self.name} is not a valid ZDT problem in zdt.py")
    if optprob_factory == zdt.ZDT5:
      raise ValueError("ZDT5 does not allow variable dimensions.")
    optprob: zdt.ZDTBaseProblem = optprob_factory(self.dim)
    impl = optprob.objective_function
    problem = _DefaultOptProblemStatement(self.dim, 2)
    return numpy_experimenter.MultiObjectiveNumpyExperimenter(impl, problem)

  def dump(self) -> vz.Metadata:
    metadata = vz.Metadata()
    metadata_dict = {
        "name": self.name,
        "dim": self.dim,
    }
    metadata[EXPERIMENTER_FACTORY_KEY] = json.dumps(metadata_dict)
    return metadata

  @classmethod
  def recover(cls, metadata: vz.Metadata) -> "ZDTExperimenterFactory":
    metadata_dict = json.loads(metadata[EXPERIMENTER_FACTORY_KEY])
    return cls(**metadata_dict)
