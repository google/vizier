# Copyright 2022 Google LLC.
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

"""The file has utilities to regress on trial intermediate measurements.

This contains utilities to fit regression models that predict the objective
at a particular future step of ACTIVE trials. We notably support LightGBM
models and necessary datastructures.
"""

from typing import (Any, Callable, Dict, List, Optional, Tuple, Union)

from absl import logging
import attr
import lightgbm.sklearn as lightgbm
import numpy as np
from scipy.interpolate.fitpack2 import InterpolatedUnivariateSpline
import six
from six.moves import range
from sklearn.model_selection import GridSearchCV
from vizier import algorithms as vza
from vizier import pyvizier
from vizier.pyvizier import converters


@attr.define
class TrialData:
  """Light weight trial data class to be used for training regression models."""
  id: int
  learning_rate: float
  final_objective: float
  steps: List[int]
  objective_values: List[float]

  @classmethod
  def from_trial(cls, trial: pyvizier.Trial, learning_rate_param_name: str,
                 metric_name: str, converter: converters.TimedLabelsExtractor):
    """Preprocess the pyvizier trial into an instance of the class.

    Args:
      trial: pyvizier.Trial containing trial to process.
      learning_rate_param_name: name of learning rate param
      metric_name: name of optimization metric
      converter: vizier tool to convert trials to times sequences

    Returns:
      returned_trial: the trial in TrialData format
    """

    learning_rate = trial.parameters.get(learning_rate_param_name,
                                         pyvizier.ParameterValue(0.0)).value

    timedlabels = converter.convert([trial])[0]
    steps, values = np.asarray(timedlabels.times, np.int32).reshape(
        -1).tolist(), timedlabels.labels[metric_name].reshape(-1).tolist()

    final_value = values[-1] if values else 0.0

    if trial.final_measurement and (metric_name
                                    in trial.final_measurement.metrics):
      final_value = converter.metric_converters[0].convert(
          [trial.final_measurement])[0]
    else:
      final_value = values[-1] if values else 0.0

    return cls(
        id=trial.id,
        learning_rate=learning_rate,
        final_objective=final_value,
        steps=steps,
        objective_values=values)

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
    steps: List[int], values: List[float]) -> Callable[[int], float]:
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
    steps: List[int], values: List[float]) -> (Tuple[List[int], List[float]]):
  """Sort and remove duplicates in the trial's measurements.

  Args:
    steps: a list of integer measurement steps for a given trial.
    values: a list of objective values corresponding to the steps for a given
      trial.

  Returns:
    steps: a list of integer measurement steps after dedupe.
    values: a list of objective values corresponding to the steps after dedupe.
  """
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

  def __init__(self,
               target_step: Union[int, float],
               min_points: int,
               learning_rate_param_name: str,
               metric_name: str,
               converter: converters.TimedLabelsExtractor,
               gbdt_param_grid: Optional[Dict[str, Any]] = None,
               cv: int = 2,
               random_state: Optional[int] = None):
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
        "n_estimators": [50, 100]
    }
    self._cv = cv
    self._random_state = random_state
    self._model: lightgbm.LGBMRegressor = None  # place holder for trained model
    self._best_params: Dict[str, Any] = None  # place holder for best parameters

  @property
  def is_trained(self) -> bool:
    return self._model is not None

  def train(self, trials: vza.CompletedTrials) -> None:
    """Trains a GBDT combined models for auto-regression given completed trials.

    Args:
      trials: Sequence of completed trials.

    Returns:
      Nothing. Updated `_model` and `_best_params` members of the class.
    """
    completed_trials = []
    for trial in trials.completed:
      completed_trials.append(
          TrialData.from_trial(
              trial,
              learning_rate_param_name=self._learning_rate_param_name,
              metric_name=self._metric_name,
              converter=self._converter))
    if len(completed_trials) < self._min_points + 1:
      logging.info("Not enough completed trials (only %d) to train GBDT model.",
                   len(completed_trials))
      return
    feature_matrix = []
    targets = []
    for trialc in completed_trials:
      # only consider trials with at least min_points + 1 steps as otherwise
      # the features constructed are mostly interpolated
      if len(trialc.steps) < self._min_points + 1:
        continue
      tc_steps, tc_values = _sort_dedupe_measurements(trialc.steps,
                                                      trialc.objective_values)
      trial_inter_func = _generate_interpolation_fn_from_trial(
          tc_steps, tc_values)
      for i, step in enumerate(trialc.steps):
        if i < self._min_points - 1 or step >= self._target_step:
          continue
        features = self._create_features_from_trial(trialc, i)
        feature_matrix.append(features)
        targets.append(trial_inter_func(self._target_step))
    feature_matrix = np.array(feature_matrix)
    targets = np.array(targets)
    gbdt_cv = GridSearchCV(
        lightgbm.LGBMRegressor(random_state=self._random_state),
        self._gbdt_param_grid,
        cv=self._cv)
    gbdt_cv = gbdt_cv.fit(feature_matrix, targets)
    self._best_params = gbdt_cv.best_params_
    gbm = lightgbm.LGBMRegressor(
        **self._best_params, random_state=self._random_state)
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
        converter=self._converter)
    if not self.is_trained:
      raise ValueError("Prediction cannot be performed before model training.")
    # Not enough features for prediction
    if len(trial_data.steps) < self._min_points:
      return None
    features = self._create_features_from_trial(trial_data,
                                                len(trial_data.steps) - 1)
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
