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

"""The file has utilities to regress on trial intermediate measurements.

This contains utilities to fit regression models that predict the objective
at a particular future step of ACTIVE trials. We notably support LightGBM
models and necessary datastructures.
"""

import copy
from typing import (Any, Callable, Dict, Optional, Tuple, Union)

from absl import logging
import attrs
import lightgbm.sklearn as lightgbm
import numpy as np
from scipy.interpolate.fitpack2 import InterpolatedUnivariateSpline
import six
from six.moves import range
from sklearn import model_selection
from vizier import algorithms as vza
from vizier import pyvizier
from vizier.pyvizier import converters


@attrs.define
class TrialData:
  """Light weight trial data class to be used for training regression models."""

  id: int
  learning_rate: float
  final_objective: float
  steps: list[int]
  objective_values: list[float]

  @classmethod
  def from_trial(
      cls,
      trial: pyvizier.Trial,
      learning_rate_param_name: str,
      metric_name: str,
      converter: converters.TimedLabelsExtractor,
  ):
    """Preprocess the pyvizier trial into an instance of the class.

    Args:
      trial: pyvizier.Trial containing trial to process.
      learning_rate_param_name: name of learning rate param
      metric_name: name of optimization metric
      converter: vizier tool to convert trials to times sequences

    Returns:
      returned_trial: the trial in TrialData format
    """

    learning_rate = trial.parameters.get(
        learning_rate_param_name, pyvizier.ParameterValue(0.0)
    ).value

    timedlabels = converter.convert([trial])[0]
    steps, values = (
        np.asarray(timedlabels.times, np.int32).reshape(-1).tolist(),
        timedlabels.labels[metric_name].reshape(-1).tolist(),
    )

    if trial.final_measurement and (
        metric_name in trial.final_measurement.metrics
    ):
      final_value = converter.metric_converters[0].convert(
          [trial.final_measurement]
      )[0]
    else:
      final_value = values[-1] if values else 0.0

    return cls(
        id=trial.id,
        learning_rate=learning_rate,
        final_objective=final_value,
        steps=steps,
        objective_values=values,
    )

  def extrapolate_trial_objective_value(self, max_num_steps: int):
    """Extend the measurements of self to max_num_steps.

    Args:
      max_num_steps: target steps to extend the measurement.
    """
    last_step = self.steps[-1]
    if last_step >= max_num_steps:
      return

    last_objective = self.objective_values[-1]
    self.steps.append(max_num_steps)
    self.objective_values.append(last_objective)


def _generate_interpolation_fn_from_trial(
    steps: list[int], values: list[float]
) -> Callable[[int], float]:
  """Generates an interpolation function from a trial's measurement data.

  Since different trials have evaluations at different step numbers,
  we need to be able to interpolate the objective value between steps
  in order to compare trials and regress against trial data. This function
  converts a trial into a function suitable for this use.

  Args:
    steps:  list of integers indicating the x-axis of the input data points.
    values: list of floats indicating the y-axis of the input data points. steps
      and values list contains the same number of elements.

  Returns:
    interpolation function that takes input a number t and returns
    interpolated value of objective function for this trial at t steps.
  """
  return InterpolatedUnivariateSpline(steps, values, k=1)


def _sort_dedupe_measurements(
    steps: list[Union[int, float]], values: list[float]
) -> Tuple[list[Union[int, float]], list[float]]:
  """Sort and remove duplicates in the trial's measurements.

  Args:
    steps: a list of integer measurement steps for a given trial.
    values: a list of objective values corresponding to the steps for a given
      trial.

  Returns:
    steps: a list of integer measurement steps after dedupe.
    values: a list of objective values corresponding to the steps after dedupe.
  """
  if isinstance(steps[0], float):
    # Dedupe is skipped when steps are not integers.
    return steps, values
  step_obj_dict = {}
  updated_steps = []
  updated_values = []
  for index in range(len(steps)):
    step_obj_dict[steps[index]] = values[index]
  last_step = None
  for step, value in sorted(six.iteritems(step_obj_dict)):
    if last_step is None or step > last_step:
      updated_steps.append(step)
      updated_values.append(value)
      last_step = step
  return updated_steps, updated_values


class GBMAutoRegressor(object):
  """Train and predict trial measurements using auto-regressive GBM model."""

  def __init__(
      self,
      target_step: Union[int, float],
      min_points: int,
      learning_rate_param_name: str,
      metric_name: str,
      converter: converters.TimedLabelsExtractor,
      gbdt_param_grid: Optional[Dict[str, Any]] = None,
      cv: int = 2,
      random_state: Optional[int] = None,
  ):
    """Initialize model params.

    Args:
      target_step: the step to compute the prediction according to existing data
        points.
      min_points: number of lag points in the auto-regressive model
      learning_rate_param_name: name of learning rate param
      metric_name: name of optimization metric
      converter: vizier tool to convert trials to times sequences
      gbdt_param_grid: parameter grid for CV grid search for lightGBM model
      cv: k in k-fold cross validation
      random_state: random state for GBDT model
    """
    self._target_step = target_step
    self._min_points = min_points
    self._converter = converter
    self._learning_rate_param_name = learning_rate_param_name
    self._metric_name = metric_name
    self._gbdt_param_grid = gbdt_param_grid or {
        "max_depth": [2, 3, 5],
        "n_estimators": [50, 100],
    }
    self._cv = cv
    self._random_state = random_state
    self._model: lightgbm.LGBMRegressor = None  # place holder for trained model
    self._best_params: Dict[str, Any] = None  # place holder for best parameters

  @property
  def is_trained(self) -> bool:
    return self._model is not None

  def train(self, completed: vza.CompletedTrials) -> None:
    """Trains a GBDT combined models for auto-regression given completed trials.

    Args:
      completed: Sequence of completed trials.

    Returns:
      Nothing. Updated `_model` and `_best_params` members of the class.
    """
    completed_trials = []
    for trial in completed.trials:
      completed_trials.append(
          TrialData.from_trial(
              trial,
              learning_rate_param_name=self._learning_rate_param_name,
              metric_name=self._metric_name,
              converter=self._converter,
          )
      )
    if len(completed_trials) < self._min_points + 1:
      logging.info(
          "Not enough completed trials (only %d) to train GBDT model.",
          len(completed_trials),
      )
      return
    feature_matrix = []
    targets = []
    for trialc in completed_trials:
      # only consider trials with at least min_points + 1 steps as otherwise
      # the features constructed are mostly interpolated
      if len(trialc.steps) < self._min_points + 1:
        continue
      trialc.extrapolate_trial_objective_value(self._target_step)
      tc_steps, tc_values = _sort_dedupe_measurements(
          trialc.steps, trialc.objective_values
      )
      trial_inter_func = _generate_interpolation_fn_from_trial(
          tc_steps, tc_values
      )
      for i, step in enumerate(trialc.steps):
        if i < self._min_points - 1 or step >= self._target_step:
          continue
        features = self._create_features_from_trial(trialc, i)
        feature_matrix.append(features)
        targets.append(trial_inter_func(self._target_step))
    feature_matrix = np.array(feature_matrix)
    logging.info("Feature matrix shape: %s", feature_matrix.shape)
    if feature_matrix.shape[0] <= (self._min_points + 1) / (
        1.0 - 1.0 / self._cv
    ):
      logging.info(
          "Not enough rows in feature matrix. "
          "This can happen when there are not enough measurements in "
          "the completed trials."
      )
      return
    targets = np.array(targets)
    gbdt_cv = model_selection.GridSearchCV(
        lightgbm.LGBMRegressor(random_state=self._random_state),
        self._gbdt_param_grid,
        cv=self._cv,
    )
    gbdt_cv = gbdt_cv.fit(feature_matrix, targets)
    self._best_params = gbdt_cv.best_params_
    gbm = lightgbm.LGBMRegressor(
        **self._best_params, random_state=self._random_state
    )
    self._model = gbm.fit(feature_matrix, targets)

  def predict(self, trial: pyvizier.Trial) -> Union[float, None]:
    """Estimate objective values at target_steps using autoregression algorithm.

    Args:
      trial: current pyvizier trial.

    Returns:
      prediction: the predicted objective value at target_steps.
      It returns None when the trial has less than min_points steps.
    """
    trial_data = TrialData.from_trial(
        trial,
        learning_rate_param_name=self._learning_rate_param_name,
        metric_name=self._metric_name,
        converter=self._converter,
    )
    if not self.is_trained:
      raise ValueError("Prediction cannot be performed before model training.")
    # Not enough features for prediction
    if len(trial_data.steps) < self._min_points:
      return None
    features = self._create_features_from_trial(
        trial_data, len(trial_data.steps) - 1
    )
    features = np.asarray(features).reshape(1, -1)
    return self._model.predict(features)[0]

  def _create_features_from_trial(
      self,
      trial: TrialData,
      end_index: int,
  ) -> list[float]:
    """Create feature vector for auto-regressive model from a trial.

    Args:
      trial: sequence of measurements as a trial
      end_index: last index in steps to be included in the feature set

    Returns:
      list of features as 1D list of size `min_points`.
    """
    if self._min_points > end_index + 1:
      raise ValueError("Not enough data before end_index for creating features")
    if end_index >= len(trial.steps):
      raise ValueError("Not enough indices in trials.steps")
    features = [trial.learning_rate]
    for j in range(self._min_points):
      # Difference from target_step and value at the step into `features`.
      features.append(self._target_step - trial.steps[end_index - j])
      features.append(trial.objective_values[end_index - j])
    return features


@attrs.define
class HallucinationOptions(object):
  """Options for hallucinated suggestions policy.

  Attributes:
    autoregressive_order: order of the auto regressive predictor.
    use_steps: if true we use `steps` for prediction, otherwise we use
      `elapsed_seconds`.
    learning_rate_param_name: name of learning rate parameter.
    gbdt_param_grid: param grid for tuning the gbdt regressor.
    min_completed_trials: only start hallucinations after this many completed
      trials.
    min_steps: only hallucinate a stopped trial which has greater than equal to
      these many steps.
    max_steps: maximum number of steps in a trial.
    random_state: random seed.
    elapsed_seconds_gap: When using steps, the hallucinated measurements
      `elapsed_seconds` is set to be the current elapsed_seconds` + this
      constant.
  """

  autoregressive_order: int = attrs.field(default=5)
  learning_rate_param_name: str = attrs.field(default="learning_rate")
  use_steps: bool = attrs.field(default=True)
  gbdt_param_grid: Dict[str, Any] = attrs.field(
      default={"max_depth": [2, 3, 5], "n_estimators": [50, 100]}
  )
  min_completed_trials: int = attrs.field(default=5)
  min_steps: int = attrs.field(default=5)
  max_steps: Optional[int] = attrs.field(default=None)
  random_state: Optional[int] = attrs.field(default=None)
  elapsed_seconds_gap: Optional[int] = attrs.field(default=0)


class GBMTrialHallucinator:
  """Regression based early stopping hallucinations."""

  def __init__(
      self,
      study_config: pyvizier.ProblemStatement,
      options: HallucinationOptions,
      verbose: int = 0,
  ):
    """Initialization.

    Args:
      study_config: pyvizier study config.
      options: hallucination options
      verbose: verbosity level. If set to > 0, then we log trial predictions.
    """
    self._options = options
    self._study_config = study_config
    if not self._study_config.metric_information.is_single_objective:
      raise ValueError("This policy only supports one objective.")
    self._metric = self._study_config.metric_information.item()
    self._converter = converters.TimedLabelsExtractor(
        [
            converters.DefaultModelOutputConverter(
                self._metric, flip_sign_for_minimization_metrics=False
            ),
        ],
        timestamp="steps" if self._options.use_steps else "elapsed_secs",
        value_extraction="raw",
    )
    self._max_steps = self._options.max_steps
    self._min_steps = self._options.min_steps
    # Place holder for model.
    self._model: Optional[GBMAutoRegressor] = None
    self._verbose = verbose

  def train(self, train_trials: vza.CompletedTrials):
    """Returns the selected trials to be tried out next.

    Args:
      train_trials: a collection of completed trials to be used for training.
    """

    # Decide whether to update stopped trials or not based on
    # `update_all_stopped_trials` flag and if there are enough train trials.
    # Finally, training an autoregressive model if the conditions are met.
    has_enough_data = len(train_trials.trials) >= max(
        self._options.autoregressive_order + 1,
        self._options.min_completed_trials,
    )
    if not has_enough_data:
      logging.info("Not enough train trials.")
      return

    self._max_steps = self._max_steps or int(
        np.percentile([len(t.measurements) for t in train_trials.trials], 95)
    )

    # If min_steps was not set, take 1/10th of the max_num_steps as its
    # value.
    self._min_steps = self._min_steps or int(self._max_steps / 10)

    if not self._max_steps:
      logging.info("Not updating stopped trials as max steps could not be set.")
      return

    global_autoregressive_model = GBMAutoRegressor(
        target_step=self._max_steps,
        min_points=self._options.autoregressive_order,
        learning_rate_param_name=self._options.learning_rate_param_name,
        metric_name=self._metric.name,
        converter=self._converter,
        gbdt_param_grid=self._options.gbdt_param_grid,
        random_state=self._options.random_state,
    )
    global_autoregressive_model.train(train_trials)
    logging.info("Finished Training global Auto-regressive GBDT model.")

    # If `global_autoregressive_model` could not be trained, then stopped trials
    # are not updated before making suggestions.
    if not global_autoregressive_model.is_trained:
      logging.info(
          "Not updating stopped trials as global GBDT model is not trained."
      )
      return

    self._model = global_autoregressive_model

  def update_stopped_trials(
      self, stopped_trials: list[pyvizier.Trial]
  ) -> list[pyvizier.Trial]:
    """Add hallucinated final measurements to stopped trials.

    Args:
      stopped_trials: stopped trials.

    Returns:
      updated stopped trials with final measurement potentially added.
    """
    if not self._model:
      logging.info("Not updating stopped trials as model was not trained.")
      return stopped_trials
    updated_trials = []  # Collect all stopped trials that need to be updated.
    for pytrial in stopped_trials:
      if pytrial.infeasible:
        updated_trials.append(pytrial)
        continue
      auto_prediction = self._model.predict(pytrial)
      if not auto_prediction:
        updated_trials.append(pytrial)
        continue
      logging.log_if(
          logging.INFO,
          "Trial generated prediction %f",
          auto_prediction,
          self._verbose >= 1,
      )
      self._create_final_measurement(
          pytrial,
          auto_prediction=auto_prediction,
      )
      logging.info("Updated stopped trial : %s, using GBDT model.", pytrial.id)
      updated_trials.append(pytrial)
    return updated_trials

  def _create_final_measurement(
      self, pytrial: pyvizier.Trial, auto_prediction: float
  ):
    """Creates a final measurement for stopped trials."""
    if pytrial.final_measurement:
      logging.info(
          "A pending trial somehow has a final measurement and will"
          "remain unchanged"
          "(trial_id=%s)",
          pytrial.id,
      )
      return
    final_measurement = copy.deepcopy(pytrial.measurements[-1])
    final_measurement.metrics[self._metric.name] = pyvizier.Metric(
        value=auto_prediction
    )
    if self._options.use_steps:
      final_measurement.steps = self._max_steps
      # Increase the elapsed_secs for the final_measurement to
      # make sure new measurements won't have newer timestamps.
      final_measurement.elapsed_secs = (
          pytrial.measurements[-1].elapsed_secs
          + self._options.elapsed_seconds_gap
      )
    else:
      final_measurement.elapsed_secs = self._max_steps
      # Increase the steps for the final_measurement to ensure it's the last
      # step. Note that the gap does not effect vizier's suggestion policy
      # and therefore is a safe operation.
      if len(pytrial.measurements) > 1:
        checkpoint_gap = (
            pytrial.measurements[-1].steps - pytrial.measurements[-2].steps
        )
      else:
        checkpoint_gap = pytrial.measurements[-1].steps
      final_measurement.steps = (
          pytrial.measurements[-1].steps + checkpoint_gap * 2
      )
    pytrial.complete(final_measurement)
