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

"""Abstractions."""

import abc
from typing import Optional, Protocol, Sequence, TypeVar

import attr
import chex
import jax
from vizier import pyvizier as vz
from vizier.interfaces import serializable

_T = TypeVar('_T')


@attr.define(frozen=True)
class CompletedTrials:
  """A group of completed trials.

  Attributes:
    trials: Completed Trials.
  """

  def __attrs_post_init__(self):
    for trial in self.trials:
      if trial.status != vz.TrialStatus.COMPLETED:
        raise ValueError(f'All trials must be completed. Bad trial:\n{trial}')

  trials: Sequence[vz.Trial] = attr.field(
      converter=tuple,
      validator=attr.validators.deep_iterable(
          attr.validators.instance_of(vz.Trial)
      ),
  )


@attr.define(frozen=True)
class ActiveTrials:
  """A group of active (a.k.a pending) trials.

  Attributes:
    trials: Active Trials.
  """

  def __attrs_post_init__(self):
    for trial in self.trials:
      if trial.status != vz.TrialStatus.ACTIVE:
        raise ValueError(f'All trials must be active. Bad trial:\n{trial}')

  trials: Sequence[vz.Trial] = attr.field(
      converter=tuple,
      default=attr.Factory(list),
      validator=attr.validators.deep_iterable(
          attr.validators.instance_of(vz.Trial)
      ),
  )


class _SuggestionAlgorithm(abc.ABC):
  """Suggestion algorithm."""

  @abc.abstractmethod
  def suggest(
      self, count: Optional[int] = None
  ) -> Sequence[vz.TrialSuggestion]:
    """Make new suggestions.

    Args:
      count: Makes best effort to generate this many suggestions. If None,
        suggests as many as the algorithm wants.

    Returns:
      New suggestions.
    """


class Designer(_SuggestionAlgorithm):
  """Interface for sequential suggestion algorithms.

  A Designer can be wrapped into a pythia `Policy` via `DesignerPolicy`.
  Prefer implementing a `Designer` interface over `Policy` interface, for
  shared error handling, performance monitoring, logging, etc.


  A big limitation of vanilla `DesignerPolicy` is that it does not retain states
  between consecutive suggestion requests. It creates a new `Designer` from
  scratch and calls `Designer.update` with all trials of the study.
  Many algorithms with a compact state, such as evolutionary search algorithms,
  can gain a huge performance boost from incremental updates. (Side note:
  GP-UCB does not fit into this category because its state includes
  all previous observations, which is not compact).

  If your Designer can take advantage of a persistent state, implement
  `(Partially)SerializableDesigner` interface, which can be wrapped into
  `(Partially)SerializableDesignerPolicy`.
  These Policies serialize the Designer's state, store it in Vizier DB, and
  load it for the next suggest operation. `Designer.update()` is called only
  with the newly completed trials (delta).

  IMPORTANT: `Designer` should not change its state inside `suggest()` (e.g. to
  incorporate its own suggestions before completion). If it does, use
  (Partially)SerializableDesigner interface.

  NOTE: When run inside a service binary, a `Designer` instance does not
  persist during the lifetime of a `Study`. This goes true even for the
  serializable variants; the states are recovered into a new `Designer`
  instance.

  NOTE: `Designer`s are designed to be used directly in benchmarks without
  a `Policy` wrapper. Create a single `Designer` instance for the entire study,
  and incrementally update its state with delta only.
  """

  @abc.abstractmethod
  def update(
      self, completed: CompletedTrials, all_active: ActiveTrials
  ) -> None:
    """Incorporates trials into the designer's state.

    Example:
      [t1, t2] # CompletedTrials
      [t3, t4] # Active Trials
      designer.update([t1], [t3])  # state includes: t1 and t3.
      designer.update([t2])        # state includes: t1 and t2 (not t3).
      designer.update([], [t3,t4]) # state includes: t1, t2, t3, and t4.
      designer.update([], [t3])    # state includes: t1, t2, and t3.

    Arguments:
      completed: COMPLETED trials that this Designer should additionaly
        incorporate.
      all_active: All ACTIVE (aka PENDING) trials in the study from its
        beginning.
    """


@attr.define(frozen=True)
class Prediction:
  """Container to hold predictions.

  The shape of the 'mean' and 'stddev' depends on the number of predictions
  and the number of objectives/metrics.

  In the single-objective case the shape is (num_predictions,).
  In the mulit-objective case the shape is (num_predictions, num_metrics).

  The metadata could be used to supply additional information about the
  prediction.
  """

  mean: chex.Array
  stddev: chex.Array
  metadata: Optional[vz.Metadata] = None

  def __attrs_post_init__(self):
    if self.mean.shape != self.stddev.shape:
      raise ValueError('The shape of mean and stddev needs to be the same.')


class Predictor(abc.ABC):
  """Predicts objective values, given suggestions.

  For algorithms which involve the use of function regressors, this class also
  acts as a mixin to expose their prediction API.
  """

  @abc.abstractmethod
  def predict(
      self,
      trials: Sequence[vz.TrialSuggestion],
      rng: Optional[jax.Array] = None,
      num_samples: Optional[int] = None,
  ) -> Prediction:
    """Returns the mean and stddev for any given suggestions.

    Arguments:
      trials: The suggestions where the predictions will be made. Can also be
        completed trials for retrospective predictions.
      rng: The sampling random key used for approximation (if applicable).
      num_samples: The number of samples used for the approximation (if
        applicable).

    Returns:
      The predictions for the given suggestions.
    """


class DesignerFactory(Protocol[_T]):
  """Protocol (PEP-544) for a designer factory."""

  def __call__(self, problem: vz.ProblemStatement, **kwargs) -> _T:
    pass


class PartiallySerializableDesigner(
    Designer, serializable.PartiallySerializable
):
  pass


class SerializableDesigner(Designer, serializable.Serializable):
  pass
