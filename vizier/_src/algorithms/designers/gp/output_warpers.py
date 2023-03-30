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

"""Output warper."""

import abc
import copy
from typing import Sequence, Optional

import attr
import attrs
import chex
import jax.numpy as jnp
import numpy as np
from scipy import stats
from tensorflow_probability.substrates import jax as tfp


def _validate_labels(
    labels_arr: chex.Array, warping: bool = True
) -> chex.Array:
  """Checks and modifies the shape and values of the labels."""
  labels_arr = labels_arr.astype(float)
  if not (labels_arr.ndim == 2 and labels_arr.shape[-1] == 1):
    raise ValueError('Labels need to be an array of shape (num_points, 1).')
  if np.isposinf(labels_arr).any():
    raise ValueError('Inifinity metric value is not valid.')
  if np.isneginf(labels_arr).any():
    labels_arr[np.isneginf(labels_arr)] = np.nan
  if (
      np.unique(labels_arr[np.isfinite(labels_arr).flatten(), :]).size <= 1
      and np.isnan(labels_arr).sum() == 0
  ) and warping:
    raise ValueError(
        'Labels need to include at least two finite unique value in the absence'
        ' of infeaible points.'
    )
  return labels_arr


class OutputWarper(abc.ABC):
  """Interface for different output warper methods."""

  # TODO: implement the unwarp for different warper classes. Currently,
  # they return an error.

  @abc.abstractmethod
  def warp(self, labels_arr: chex.Array) -> chex.Array:
    """Runs the output warper of choice on an array of labels.

    Args:
      labels_arr: (num_points, 1) shaped array of unwarped labels. Note that
        each call accomodates one metric, which can be an objective or a safety
        metric. NaN and infinity values are allowed and can be warped based on
        the choice of warper. Labels are assumed to be maximizing. labels_arr
        will not be mutated.

    Returns:
      (num_points, 1) shaped array of finite warped mutated labels.
    """
    pass

  @abc.abstractmethod
  def unwarp(self, labels_arr: chex.Array) -> chex.Array:
    """Runs the inverse of output warper of choice on an array of  labels.

    Args:
      labels_arr: (num_points, 1) shaped array of warped labels. Note tha each
        call accomodates one metric, which can be an objective or a safety
        metric. NaN and infinity values are allowed and will untouched. Labels
        are assumed to be maximizing. labels_arr will not be mutated.
        Technically any array can be passed to the unwarp method, but we are
        interested in unwarping warped labels for purposes of GP training and
        prediction.

    Returns:
      (num_points, 1) shaped array of finite unwarped mutated labels.
    """
    pass


@attr.define
class OutputWarperPipeline(OutputWarper):
  """Performs a sequence of warpings on an input labels array."""
  warpers: Sequence[OutputWarper] = attr.ib(factory=list)

  def warp(self, labels_arr: chex.Array) -> chex.Array:
    """Sequntial warping of the labels.

    Note that if labels include one unique finite value, pipeline returns
    an array of zeros. Alternatively, if the labels include only infeasible
    values (eg. NaNs and -inf), pipeline returns an array of minus ones.
    Otherwise, it *sequntially* performs the warpings on labels. Warping is not
    done in place.

    Args:
      labels_arr: (num_points, 1) shaped array of labels.

    Returns:
      (num_points, 1) shaped array of warped labels.
    """
    labels_arr = copy.deepcopy(labels_arr)
    if np.isneginf(labels_arr).any():
      labels_arr[np.isneginf(labels_arr)] = np.nan
    if np.isfinite(labels_arr).all() and len(
        np.unique(labels_arr).flatten()) == 1:
      return np.zeros(labels_arr.shape)
    if np.isnan(labels_arr).all():
      return -1 * np.ones(shape=labels_arr.shape)
    for warper in self.warpers:
      labels_arr = warper.warp(labels_arr)
    return labels_arr

  def unwarp(self, labels_arr: chex.Array) -> chex.Array:
    """Sequential unwarping of the warped labels.

    Note that if labels include one unique finite value zero, the pipeline
    returns the array itself. Alternatively, if the labels include only minus
    ones, the pipeline returns an array of NaNs. Otherwise, it *sequentially*
    performs the unwarpings on labels. Unwarping is not done in place.

    Args:
      labels_arr: (num_points, 1) shaped array of warped labels.

    Returns:
      (num_points, 1) shaped array of unwarped labels.
    """
    labels_arr = copy.deepcopy(labels_arr)
    if (
        np.isfinite(labels_arr).all()
        and len(np.unique(labels_arr).flatten()) == 1
    ):
      if np.unique(labels_arr) == 0.0:
        return labels_arr
      if np.unique(labels_arr) == -1.0:
        return np.nan * np.ones(shape=labels_arr.shape)
    for warper in self.warpers:
      labels_arr = warper.unwarp(labels_arr)
    return labels_arr


def create_default_warper(
    *,
    half_rank_warp: bool = True,
    log_warp: bool = True,
    infeasible_warp: bool = False) -> OutputWarperPipeline:
  """Creates an output warper pipeline.

  Args:
    half_rank_warp: boolean indicating if half-rank warping to be performed.
    log_warp: boolean indicating if log warping to be performed.
    infeasible_warp: boolean indicating if infeasible warping to be performed.

  Returns:
    an instance of OutputWarperPipeline.
  """
  if not half_rank_warp and not log_warp and not infeasible_warp:
    raise ValueError(
        'At least one of "half_rank_warp", "log_warp" or "infeasible_warp" '
        'must be True.'
    )
  warpers = []
  if half_rank_warp:
    warpers.append(HalfRankComponent())
  if log_warp:
    warpers.append(LogWarperComponent())
  if infeasible_warp:
    warpers.append(InfeasibleWarperComponent())
  return OutputWarperPipeline(warpers)


def create_warp_outliers_warper(
    *,
    warp_outliers: bool = True,
    infeasible_warp: bool = True,
    transform_gaussian: bool = True) -> OutputWarperPipeline:
  """Creates an output warper outline which detects outliers and warps them."""
  warpers = []
  if warp_outliers:
    warpers.append(DetectOutliers())
  if infeasible_warp:
    warpers.append(InfeasibleWarperComponent())
  if transform_gaussian:
    warpers.append(TransformToGaussian())
  return OutputWarperPipeline(warpers)


@attr.define
class HalfRankComponent(OutputWarper):
  """Warps half of an array of labels to fit into a Gaussian distribution.

  Note that this warping is performed on finite values of the array and NaNs are
  untouched.
  """

  _median: Optional[float] = attr.field(default=None)
  _stddev: Optional[float] = attr.field(default=None)
  _dedup_median_index: Optional[int] = attr.field(default=None)
  _unique_labels: Optional[chex.Array] = attr.field(default=None)

  def _estimate_std_of_good_half(
      self, unique_labels: chex.Array, threshold: float
  ) -> float:
    """Estimates the standard devation of the good half of the array.

    Args:
      unique_labels: (num_points, 1) shaped array of unique labels.
      threshold: minimum label value to be considered in "good half".

    Returns:
      float estimated standard devation of the good half of the array.
    """
    good_half = unique_labels[unique_labels >= threshold]
    std = np.sqrt(((good_half - threshold)**2).sum() * (1 / good_half.shape[0]))
    if std > 0:
      return std
    std = np.sqrt(
        ((unique_labels - threshold)**2).sum() * (1 / unique_labels.shape[0]))
    if np.isfinite(std):
      return std
    std = (np.abs(unique_labels - threshold)).sum() * (1 /
                                                       unique_labels.shape[0])
    return std

  def warp(self, labels_arr: chex.Array) -> chex.Array:
    """See base class."""
    labels_arr = _validate_labels(labels_arr)
    if labels_arr.size == 1:
      return labels_arr
    labels_arr = labels_arr.flatten()
    # Compute median, unique labels, and ranks.
    median = np.nanmedian(labels_arr)
    self._median = median
    self._stddev = np.nanstd(labels_arr)
    unique_labels = np.unique(labels_arr[np.isfinite(labels_arr)])
    self._unique_labels = unique_labels
    ranks = stats.rankdata(labels_arr, method='dense')  # nans ranked last.

    dedup_median_index = unique_labels.searchsorted(median, 'left')
    self._dedup_median_index = dedup_median_index
    denominator = dedup_median_index + (unique_labels[dedup_median_index]
                                        == median) * .5
    estimated_std = self._estimate_std_of_good_half(unique_labels, median)
    # Apply transformation to points below median.
    for i, (yy, rank) in enumerate(zip(labels_arr, ranks)):
      if np.isfinite(yy) and yy < median:
        # FYI: a lot of effort went into choosing this arbitrary
        # denominator. rankdata(method='max') and simply using
        # rank_quantile = rank / np.isfinite(labels_arr).sum() should be
        # just as fine.
        rank_quantile = 0.5 * (rank - 0.5) / denominator
        # rank_ppf is always less than 0
        rank_ppf = stats.norm.ppf(rank_quantile)
        labels_arr[i] = rank_ppf * estimated_std + median
    return np.reshape(labels_arr, [-1, 1])

  def unwarp(self, labels_arr: chex.Array) -> chex.Array:
    labels_arr = _validate_labels(labels_arr, warping=False)
    if np.isnan(labels_arr).any():
      raise ValueError('Array passed to unwarp cannot include nans.')
    if self._dedup_median_index == 0:
      return self._median + self._stddev * labels_arr
    labels_arr[labels_arr >= 0.0] = (
        self._median + self._stddev * labels_arr[labels_arr >= 0.0]
    )
    rank_bad = np.array(
        [
            2 * stats.norm.cdf(y) * (self._dedup_median_index + 0.5) - 0.5
            for y in labels_arr[labels_arr < 0.0]
        ]
    )
    if (rank_bad < -0.5).any() or (
        rank_bad > 1.0001 * self._dedup_median_index
    ).any():
      raise ValueError('Rank needs to be within [-0.5, 1.0001 * median-index].')
    labels_bad = np.ones(labels_arr[labels_arr < 0.0].shape)
    scale = self._stddev + self._median - np.min(self._unique_labels)
    if scale < 0.0:
      raise ValueError('Scale needs to be non-negative.')
    r_ints, r_fracs = divmod(rank_bad[rank_bad >= 0.0], 1)
    labels_bad[rank_bad >= 0.0] = np.array(
        [
            self._unique_labels(int(r_int)) * (1 - r_frac)
            + (self._unique_labels(int(r_int) + 1) * r_frac)
            for r_int, r_frac in zip(r_ints, r_fracs)
        ]
    )
    labels_bad[rank_bad < 0.0] = (
        np.min(self._unique_labels) + scale * rank_bad[rank_bad < 0.0]
    )

    labels_arr[labels_arr < 0.0] = labels_bad
    return labels_arr


@attr.define
class LogWarperComponent(OutputWarper):
  """Warps an array of labels to highlght the difference between good values.

  Note that this warping is performed on finite values of the array and NaNs are
  untouched.
  """

  _labels_min: Optional[float] = attr.field(default=None)
  _labels_max: Optional[float] = attr.field(default=None)
  offset: float = attr.field(default=1.5, validator=attrs.validators.gt(0.0))

  def warp(self, labels_arr: chex.Array) -> chex.Array:
    """See base class."""
    labels_arr = _validate_labels(labels_arr)
    self._labels_min = np.nanmin(labels_arr)
    self._labels_max = np.nanmax(labels_arr)
    labels_arr = labels_arr.flatten()
    labels_range = np.nanmax(labels_arr) - np.nanmin(labels_arr)
    labels_arr[np.isfinite(labels_arr)] = 0.5 - (
        (
            np.log1p(
                (
                    (
                        np.nanmax(labels_arr)
                        - labels_arr[np.isfinite(labels_arr)]
                    )
                    / labels_range
                )
                * (self.offset - 1)
            )
        )
        / (np.log(self.offset))
    )
    return np.reshape(labels_arr, [-1, 1])

  def unwarp(self, labels_arr: chex.Array) -> chex.Array:
    if self._labels_max is None or self._labels_min is None:
      raise ValueError(' warp() needs to be called before unwarp() is called.')
    labels_arr = self._labels_max - (
        np.exp(np.log(self.offset) * (0.5 - labels_arr)) - 1
    ) * ((self._labels_max - self._labels_min) / (self.offset - 1))
    return np.reshape(labels_arr, [-1, 1])


@attr.define
class InfeasibleWarperComponent(OutputWarper):
  """Warps the infeasible/nan value to feasible/finite values."""

  def warp(self, labels_arr: chex.Array) -> chex.Array:
    labels_arr = _validate_labels(labels_arr)
    labels_arr = labels_arr.flatten()
    labels_range = np.nanmax(labels_arr) - np.nanmin(labels_arr)
    warped_bad_value = np.nanmin(labels_arr) - (0.5 * labels_range + 1)
    labels_arr[np.isnan(labels_arr)] = warped_bad_value
    return np.reshape(labels_arr, [-1, 1])

  def unwarp(self, labels_arr: chex.Array) -> chex.Array:
    raise ValueError(
        'InfeasibleWarperComponent does not support unwarp method.'
    )


class ZScoreLabels(OutputWarper):
  """Sandardizes finite label values, leaving the NaNs & infinities out."""

  def warp(self, labels_arr: chex.Array) -> chex.Array:
    """Sandardizes finite label values to scale the mean to 0 and std to 1.

    Args:
      labels_arr: (num_points, 1) shaped array of labels.

    Returns:
      (num_points, 1) shaped array of standardize labels.
    """
    labels_arr = _validate_labels(labels_arr)
    if np.isnan(labels_arr).all():
      raise ValueError('Labels need to have at least one non-NaN entry.')
    labels_finite_ind = np.isfinite(labels_arr)
    labels_arr_finite = labels_arr[labels_finite_ind]
    if np.nanstd(labels_arr_finite) == 0 or not np.isfinite(
        np.nanstd(labels_arr_finite)):
      return labels_arr
    labels_arr_finite_normalized = (
        labels_arr_finite -
        np.nanmean(labels_arr_finite)) / np.nanstd(labels_arr_finite)
    labels_arr[labels_finite_ind] = labels_arr_finite_normalized
    return labels_arr

  def unwarp(self, labels_arr: chex.Array) -> chex.Array:
    raise NotImplementedError(
        'unwarp  method for ZScoreLabels is not implemented yet.'
    )


class NormalizeLabels(OutputWarper):
  """Normalizes the finite label values, leaving the NaNs & infinities out."""

  def warp(self, labels_arr: chex.Array) -> chex.Array:
    """Normalizes the finite label values to bring them between 0 and 1.

    Args:
      labels_arr: (num_points, 1) shaped array of labels.

    Returns:
      (num_points, 1) shaped array of normalized labels.
    """
    labels_arr = _validate_labels(labels_arr)

    if np.isnan(labels_arr).all():
      raise ValueError('Labels need to have at least one non-NaN entry.')
    if np.nanmax(labels_arr) == np.nanmax(labels_arr):
      return labels_arr
    labels_finite_ind = np.isfinite(labels_arr)
    labels_arr_finite = labels_arr[labels_finite_ind]
    labels_arr_finite_normalized = (
        labels_arr_finite - np.nanmin(labels_arr_finite)) / (
            np.nanmax(labels_arr_finite) - np.nanmin(labels_arr_finite))
    labels_arr[labels_finite_ind] = labels_arr_finite_normalized
    return labels_arr

  def unwarp(self, labels_arr: chex.Array) -> chex.Array:
    raise NotImplementedError(
        'unwarp  method for NormalizeLabels is not implemented yet.'
    )


@attr.define
class DetectOutliers(OutputWarper):
  """Detects outliers from an array of labels.

  The goal of this warper is to detect the unreasonably bad labels, aka
  outliers. Eg, assuming a maximization problem where the `normal` range of
  labels is [1, 10], the occasionally observed value of -10**76 counts as an
  outlier. For this warper, we only consider the *finite* values in the labels
  and leave the NaN and infinity values untouched.

  The proposed warping finds the difference between the maximum label and the
  median of the labels.

  Attributes:
    min_zscore: number of stds below the median for variance estimation.
    max_zscore: number of stds above the median for outlier detection.
  """
  min_zscore: float = attr.field(kw_only=True, default=6.)
  max_zscore: float = attr.field(kw_only=True, default=None)

  def _estimate_variance(self, labels_arr: chex.Array) -> float:
    """Estimates the variance of labels array using the top half values.

    The estimation is a function of the size of the labels array. For details
    and derivations, see the following paper:
    `Estimating the mean and variance from the median, range, and the size of
    a sample` paper found below
    https://link.springer.com/content/pdf/10.1186/1471-2288-5-13.pdf.

    Args:
      labels_arr: (num_points, 1) shaped array of labels.

    Returns:
      The estiamated variance of the labels.
    """
    num_points = labels_arr.shape[0]
    labels_median = np.nanmedian(labels_arr)
    labels_max = np.nanmax(labels_arr)

    if not np.isfinite(labels_max) or np.isnan(labels_max):
      raise ValueError('The max label value should be finite.')

    if not np.isfinite(labels_median) or np.isnan(labels_median):
      raise ValueError('The median label value should be finite.')

    if self.max_zscore:
      return (labels_max - labels_median) / self.min_zscore

    if num_points >= 70:
      return (labels_max - labels_median) / 3
    elif num_points >= 15:
      return (labels_max - labels_median) / 2
    else:
      # We hallucinate the min labels value
      labels_min_hallucinated = labels_median - np.max(labels_arr)
      # Following the paper assumptions, we need to make sure the hallucinated
      # min is non-negative and shift the rest of the values accordingly.
      if labels_min_hallucinated < 0:
        labels_min_hallucinated = 0
        labels_max -= labels_min_hallucinated
        labels_median -= labels_min_hallucinated
      # eq. 12 in the paper mentioned above.
      return 1 / (num_points - 1) * (
          labels_min_hallucinated**2 + labels_median**2 + labels_max**2 +
          ((num_points - 3) / 2) *
          (((labels_min_hallucinated + labels_median)**2 +
            (labels_max + labels_median)**2) / 4) - num_points *
          ((labels_min_hallucinated + 2 * labels_median + labels_max) / 4 +
           (labels_min_hallucinated - 2 * labels_median + labels_max) /
           (4 * num_points))**2)

  def warp(self, labels_arr: chex.Array) -> chex.Array:
    labels_arr = _validate_labels(labels_arr)
    labels_finite_ind = np.isfinite(labels_arr)
    labels_arr_finite = labels_arr[labels_finite_ind]
    labels_median = np.median(labels_arr_finite)
    labels_std = np.sqrt(self._estimate_variance(labels_arr_finite))
    threshold = labels_median - self.min_zscore * labels_std
    labels_arr_finite[labels_arr_finite < threshold] = np.nan
    labels_arr[labels_finite_ind] = labels_arr_finite
    return labels_arr

  def unwarp(self, labels_arr: chex.Array) -> chex.Array:
    raise NotImplementedError(
        'unwarp  method for DetectOutliers is not implemented yet.'
    )


class TransformToGaussian(OutputWarper):
  """Transforms the labels into a Gaussian distribution .

  The goal of this warper is to transform the label into a Gaussian sample to
  better suit it for a Gaussian process. Here, we use a non-parametric warper
  called quantile transformer. Traditonally, this transformer relies on the
  relational rank of the labels, however this approach does not take into
  account the values of the labels. To address this issue, we also provide
  another option to use the normalized distances instead.

  Alternatives include a box-cox transformation (for non-negative labels) or a
  yeo-johnson transformation. Both these methods are parameteric and their
  parameters can be learned using a maximum likelihood approach.
  """

  def __init__(
      self,
      *,
      softclip_low: float = 1e-10,
      softclip_high: float = 1 - 1e-10,
      softclip_hinge_softness: float = 0.01,
      use_rank: bool = False,
  ):
    self.softclip_low = softclip_low
    self.softclip_high = softclip_high
    self.softclip_hinge_softness = softclip_hinge_softness
    self.use_rank = use_rank

  def warp(self, labels_arr: chex.Array) -> chex.Array:
    labels_arr = _validate_labels(labels_arr)
    labels_arr = np.asarray(labels_arr, dtype=np.float64)
    labels_arr_flattened = labels_arr.flatten()
    if self.use_rank:
      base_for_transform = np.argsort(labels_arr_flattened)
    else:
      base_for_transform = labels_arr_flattened
    base_for_transform_normalized = (
        base_for_transform - np.min(base_for_transform)) / (
            np.max(base_for_transform) - np.min(base_for_transform))
    clip = tfp.bijectors.SoftClip(
        low=jnp.array(self.softclip_low, dtype=labels_arr.dtype),
        high=jnp.array(self.softclip_high, dtype=labels_arr.dtype),
        hinge_softness=self.softclip_hinge_softness)
    base_for_transform_normalized_clipped = np.array(
        clip.forward(base_for_transform_normalized))
    normal_dist = tfp.distributions.Normal(0., 1)
    labels_arr_transformed = normal_dist.quantile(
        base_for_transform_normalized_clipped)
    labels_arr_transformed = np.reshape(labels_arr_transformed,
                                        labels_arr.shape)
    return labels_arr_transformed

  def unwarp(self, labels_arr: chex.Array) -> chex.Array:
    raise NotImplementedError(
        'unwarp  method for TransformToGaussian is not implemented yet.'
    )
