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

"""Acquisition functions and builders implementations."""

import abc
import copy
from typing import Any, Callable, Optional, Protocol, Sequence

import attr
import chex
import jax
from jax import numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from vizier import pyvizier as vz
from vizier._src.jax import stochastic_process_model as sp
from vizier.pyvizier import converters


tfd = tfp.distributions
tfp_bo = tfp.experimental.bayesopt


class AcquisitionFunction(Protocol):

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[chex.ArrayTree] = None,
      labels: Optional[chex.Array] = None,
  ) -> chex.Array:
    pass


@attr.define
class UCB(AcquisitionFunction):
  """UCB AcquisitionFunction."""

  coefficient: float = attr.field(
      default=1.8, validator=attr.validators.instance_of(float)
  )

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[chex.ArrayTree] = None,
      labels: Optional[chex.Array] = None,
  ) -> chex.Array:
    del features, labels
    return dist.mean() + self.coefficient * dist.stddev()


@attr.define
class HyperVolumeScalarization(AcquisitionFunction):
  """HyperVolume Scalarization acquisition function."""

  coefficient: float = attr.field(
      default=1.0, validator=attr.validators.instance_of(float)
  )

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[chex.ArrayTree] = None,
      labels: Optional[chex.Array] = None,
  ) -> chex.Array:
    del features, labels
    # Uses scalarizations in https://arxiv.org/abs/2006.04655 for
    # non-convex biobjective optimization of mean vs stddev.
    return jnp.minimum(dist.mean(), self.coefficient * dist.stddev())


@attr.define
class EI(AcquisitionFunction):

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[chex.ArrayTree] = None,
      labels: Optional[chex.Array] = None,
  ) -> chex.Array:
    del features
    return tfp_bo.acquisition.GaussianProcessExpectedImprovement(dist, labels)()


class EI(AcquisitionFunction):

  def __call__(
      self,
      dist: tfd.Distribution,
      features: Optional[chex.ArrayTree] = None,
      labels: Optional[chex.Array] = None,
  ) -> chex.Array:
    del features
    return tfp_bo.acquisition.GaussianProcessExpectedImprovement(dist, labels)()


# TODO: Support discretes and categoricals.
# TODO: Support custom distances.
class TrustRegion:
  """L-inf norm based TrustRegion.

  Limits the suggestion within the union of small L-inf norm balls around each
  of the trusted points, which are in most cases observed points. The radius
  of the L-inf norm ball grows in the number of observed points.

  Assumes that all points are in the unit hypercube.

  The trust region can be used e.g. during acquisition optimization:
    converter = converters.TrialToArrayConverter.from_study_config(problem)
    features, labels = converter.to_xy(trials)
    tr = TrustRegion(features, converter.output_specs)
    # xs is a point in the search space.
    distance = tr.min_linf_distance(xs)
    if distance <= tr.trust_radius:
      print('xs in trust region')
  """

  def __init__(
      self, trusted: chex.Array, specs: Sequence[converters.NumpyArraySpec]
  ):
    """Init.

    Args:
      trusted: Array of shape (N, D) where each element is in [0, 1]. Each row
        is the D-dimensional vector representing a trusted point.
      specs: List of output specs of the `TrialToArrayConverter`.
    """
    self._trusted = trusted
    self._dof = len(specs)
    self._trust_radius = self._compute_trust_radius(self._trusted)

    max_distance = []
    for spec in specs:
      # Cap distances between one-hot encoded features so that they fall within
      # the trust region radius.
      if spec.type is converters.NumpyArraySpecType.ONEHOT_EMBEDDING:
        max_distance.extend([self._trust_radius] * spec.num_dimensions)
      else:
        max_distance.append(np.inf)
    self._max_distances = np.array(max_distance)

  def _compute_trust_radius(self, trusted: chex.Array) -> chex.Scalar:
    """Computes the trust region radius."""
    # TODO: Make hyperparameters configurable.
    min_radius = 0.2  # Hyperparameter
    dimension_factor = 5.0  # Hyperparameter

    # TODO: Discount the infeasible points.

    trust_level = (0.1 * trusted.shape[0] + 0.9 * trusted.shape[0]) / (
        dimension_factor * (self._dof + 1)
    )
    trust_region_radius = min_radius + (0.5 - min_radius) * trust_level
    return trust_region_radius

  @property
  def trust_radius(self) -> float:
    return self._trust_radius

  def min_linf_distance(self, xs: chex.Array) -> chex.Array:
    """l-inf norm distance to the closest trusted point.

    Caps distances between one-hot encoded features to the trust-region radius,
    so that the trust region cutoff does not discourage exploration of these
    features.

    Args:
      xs: (M, D) array where each element is in [0, 1].

    Returns:
      (M,) array of floating numbers, L-infinity distances to the nearest
      trusted point.
    """
    distances = jnp.abs(self._trusted - xs[..., jnp.newaxis, :])  # (M, N, D)
    distances_bounded = jnp.minimum(distances, self._max_distances)
    linf_distance = jnp.max(distances_bounded, axis=-1)  # (M, N)
    return jnp.min(linf_distance, axis=-1)  # (M,)


class AcquisitionBuilder(abc.ABC):
  """Acquisition/prediction builder.

  This builder takes in a Jax/Flax model, along with its hparams, and builds
  the usable predictive metrics, as well as the acquisition problem and jitted
  function (note that build may reuse cached jits).
  """

  @abc.abstractmethod
  def build(
      self,
      problem: vz.ProblemStatement,
      model: sp.StochasticProcessModel,
      state: chex.ArrayTree,
      features: chex.Array,
      labels: chex.Array,
      *args,
      **kwargs,
  ) -> None:
    """Builds the predict and acquisition functions.

    Args:
      problem: Initial problem.
      model: Jax/Flax model for predictions.
      state: State of trained hparams or precomputation to be applied in model.
      features: Features array for acquisition computations (i.e. TrustRegion).
      labels: Labels array for acquisition computation (i.e. EI)
      *args:
      **kwargs:
    """
    pass

  @property
  @abc.abstractmethod
  def metadata_dict(self) -> dict[str, Any]:
    """A dictionary of key-value pairs to be added to Suggestion metadata."""
    pass

  @property
  @abc.abstractmethod
  def acquisition_problem(self) -> vz.ProblemStatement:
    """Acquisition optimization problem statement."""
    pass

  @property
  @abc.abstractmethod
  def acquisition_on_array(self) -> Callable[[chex.Array], chex.Array]:
    """Acquisition function on features array."""
    pass

  @property
  @abc.abstractmethod
  def predict_on_array(self) -> Callable[[chex.Array], chex.ArrayTree]:
    """Prediction function on features array."""
    pass


@attr.define(slots=False)
class GPBanditAcquisitionBuilder(AcquisitionBuilder):
  """Acquisition/prediction builder for the GPBandit-type designers.

  This builder takes in a Jax/Flax model, along with its hparams, and builds
  the usable predictive metrics, as well as the acquisition.

  For example:

    acquisition_builder =
    GPBanditAcquisitionBuilder(acquisition_fn=acquisitions.UCB())
    acquisition_builder.build(
          problem_statement,
          model=model,
          state=state,
          features=features,
          labels=labels,
          converter=self._converter,
    )
    # Get the acquisition Callable.
    acq = acquisition_builder.acquisition_on_array
  """

  # Acquisition function that takes a TFP distribution and optional features
  # and labels.
  acquisition_fn: AcquisitionFunction = attr.field(factory=UCB, kw_only=True)
  use_trust_region: bool = attr.field(default=True, kw_only=True)

  def __attrs_post_init__(self):
    # Perform extra initializations.
    self._built = False

  def build(
      self,
      problem: vz.ProblemStatement,
      model: sp.StochasticProcessModel,
      state: chex.ArrayTree,
      features: chex.Array,
      labels: chex.Array,
      converter: converters.TrialToArrayConverter,
      use_vmap: bool = True,
  ) -> None:
    """Generates the predict and acquisition functions.

    Args:
      problem: See abstraction.
      model: See abstraction.
      state: See abstraction.
      features: See abstraction.
      labels: See abstraction.
      converter: TrialToArrayConverter for TrustRegion configuration.
      use_vmap: If True, applies Vmap across parameter ensembles.
    """

    def _predict_on_array_one_model(
        state: chex.ArrayTree, *, xs: chex.Array
    ) -> chex.ArrayTree:
      return model.apply(
          state,
          xs,
          features,
          labels,
          method=model.posterior_predictive,
      )

    # Vmaps and combines the predictive distribution over all models.
    def _get_predictive_dist(xs: chex.Array) -> tfd.Distribution:
      if not use_vmap:
        return _predict_on_array_one_model(state, xs=xs)

      def _predict_mean_and_stddev(state_: chex.ArrayTree) -> chex.ArrayTree:
        dist = _predict_on_array_one_model(state_, xs=xs)
        return {'mean': dist.mean(), 'stddev': dist.stddev()}  # pytype: disable=attribute-error  # numpy-scalars

      # Returns a dictionary with mean and stddev, of shape [M, N].
      # M is the size of the parameter ensemble and N is the number of points.
      pp = jax.vmap(_predict_mean_and_stddev)(state)
      batched_normal = tfd.Normal(pp['mean'].T, pp['stddev'].T)  # pytype: disable=attribute-error  # numpy-scalars

      return tfd.MixtureSameFamily(
          tfd.Categorical(logits=jnp.ones(batched_normal.batch_shape[1])),
          batched_normal,
      )

    # Could also allow injection of a prediction function that takes a
    # distribution and returns something else, or rename this to
    # 'predict_mean_and_stddev`, e.g.`
    @jax.jit
    def predict_on_array(xs: chex.Array) -> chex.ArrayTree:
      dist = _get_predictive_dist(xs)
      return {'mean': dist.mean(), 'stddev': dist.stddev()}

    self._predict_on_array = predict_on_array

    # Define acquisition.
    self._tr = TrustRegion(features, converter.output_specs)

    # This supports acquisition fns that do arbitrary computations with the
    # input distributions -- e.g. they could take samples or compute quantiles.
    @jax.jit
    def acquisition_on_array(xs):
      dist = _get_predictive_dist(xs)
      acquisition = self.acquisition_fn(dist, features, labels)
      if self.use_trust_region and self._tr.trust_radius < 0.5:
        distance = self._tr.min_linf_distance(xs)
        # Due to output normalization, acquisition can't be nearly as
        # low as -1e12.
        # We use a bad value that decreases in the distance to trust region
        # so that acquisition optimizer can follow the gradient and escape
        # untrusted regions.
        return jnp.where(
            distance <= self._tr.trust_radius, acquisition, -1e12 - distance
        )
      else:
        return acquisition

    self._acquisition_on_array = acquisition_on_array

    acquisition_problem = copy.deepcopy(problem)
    config = vz.MetricsConfig(
        metrics=[
            vz.MetricInformation(
                name='acquisition', goal=vz.ObjectiveMetricGoal.MAXIMIZE
            )
        ]
    )
    acquisition_problem.metric_information = config
    self._acquisition_problem = acquisition_problem
    self._built = True

  @property
  def acquisition_problem(self) -> vz.ProblemStatement:
    if not self._built:
      raise ValueError('Acquisition must be built first via build().')
    return self._acquisition_problem

  @property
  def acquisition_on_array(self) -> Callable[[chex.Array], chex.Array]:
    if not self._built:
      raise ValueError('Acquisition must be built first via build().')
    return self._acquisition_on_array

  @property
  def predict_on_array(self) -> Callable[[chex.Array], chex.ArrayTree]:
    if not self._built:
      raise ValueError('Acquisition must be built first via build().')
    return self._predict_on_array

  @property
  def metadata_dict(self) -> dict[str, Any]:
    if not self._built:
      raise ValueError('Acquisition must be built first via build().')
    return {'trust_radius': self._tr.trust_radius}
