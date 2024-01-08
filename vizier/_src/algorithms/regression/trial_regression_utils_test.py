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

"""Tests for trial_regression_utils."""

import copy
from typing import Union

import lightgbm.sklearn as lightgbm
import numpy as np
from sklearn.model_selection import GridSearchCV
from vizier import algorithms as vza
from vizier import pyvizier
from vizier._src.algorithms.regression import trial_regression_utils
from vizier.pyvizier import converters

from absl.testing import absltest

_METRIC_NAME = 'objective_value'


def _create_trial_for_testing(
    learning_rate: float,
    steps: list[int],
    seconds: list[Union[int, float]],
    values: list[float],
    stop_reason: Union[None, str],
    trial_id: int,
    metric_name: str = _METRIC_NAME,
):
  measurements = []
  for i in range(len(steps)):
    measurements.append(
        pyvizier.Measurement(
            steps=steps[i],
            elapsed_secs=seconds[i],
            metrics={metric_name: values[i]},
        )
    )
  trial = pyvizier.Trial(
      id=trial_id,
      measurements=measurements,
      stopping_reason=stop_reason,
      parameters=pyvizier.ParameterDict({'learning_rate': learning_rate}),
      final_measurement=measurements[-1],
  )
  return trial


class TrialRegressionUtilsTest(absltest.TestCase):

  def test_preprocess_trial(self):
    steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    seconds = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    values = [0.1, 0.3, 0.4, 0.45, 0.48, 0.49, 0.5, 0.49, 0.51, 0.5]

    metric = pyvizier.MetricInformation(
        name=_METRIC_NAME, goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
    )
    conv1 = converters.TimedLabelsExtractor(
        [
            converters.DefaultModelOutputConverter(
                metric, flip_sign_for_minimization_metrics=True
            ),
        ],
        timestamp='steps',
        value_extraction='raw',
    )
    trial = _create_trial_for_testing(
        0.3, steps, seconds, values, None, 1, _METRIC_NAME
    )
    conv2 = converters.TimedLabelsExtractor(
        [
            converters.DefaultModelOutputConverter(
                metric, flip_sign_for_minimization_metrics=True
            ),
        ],
        timestamp='elapsed_secs',
        value_extraction='raw',
    )

    conv_trial1 = trial_regression_utils.TrialData.from_trial(
        trial=trial,
        learning_rate_param_name='learning_rate',
        metric_name=_METRIC_NAME,
        converter=conv1,
    )
    conv_trial2 = trial_regression_utils.TrialData.from_trial(
        trial=trial,
        learning_rate_param_name='learning_rate',
        metric_name=_METRIC_NAME,
        converter=conv2,
    )
    self.assertListEqual(steps, conv_trial1.steps)
    self.assertSequenceAlmostEqual(values, conv_trial1.objective_values)
    self.assertListEqual(seconds, conv_trial2.steps)
    self.assertSequenceAlmostEqual(values, conv_trial2.objective_values)

  def test_sort_dedupe_measurements(self):
    # Case 1: no duplicates.
    steps = [10, 20]
    values = [0.1, 0.2]
    actual_steps, actual_values = (
        trial_regression_utils._sort_dedupe_measurements(steps, values)
    )
    self.assertListEqual(steps, actual_steps)
    self.assertListEqual(values, actual_values)

    # Case 2: with duplicates, and measurement order needs correction.
    steps = [10, 20, 10]
    values = [0.1, 0.2, 0.2]
    expected_steps = [10, 20]
    expected_values = [0.2, 0.2]
    actual_steps, actual_values = (
        trial_regression_utils._sort_dedupe_measurements(steps, values)
    )
    self.assertListEqual(expected_steps, actual_steps)
    self.assertListEqual(expected_values, actual_values)

  def test_generate_interpolation_fn_from_trial(self):
    steps = [10, 20]
    values = [0.1, 0.5]
    fn = trial_regression_utils._generate_interpolation_fn_from_trial(
        steps, values
    )
    self.assertEqual(fn(steps[0]), values[0])
    self.assertEqual(fn(steps[1]), values[1])
    self.assertGreater(fn(15), values[0])
    self.assertLess(fn(15), values[1])

  def test_extrapolate_trial(self):
    steps = [10, 20, 30]
    values = [0.1, 0.3, 0.5]

    expected_steps = [10, 20, 30, 100]
    expected_values = [0.1, 0.3, 0.5, 0.5]

    trial = trial_regression_utils.TrialData(
        id=1,
        learning_rate=0.1,
        final_objective=values[-1],
        steps=steps,
        objective_values=values,
    )

    expected_trial = copy.deepcopy(trial)

    # Case 1: enough steps, no extension needed.
    trial.extrapolate_trial_objective_value(30)
    self.assertEqual(trial, expected_trial)

    # Case 2: fewer steps, extension needed.
    expected_trial.extrapolate_trial_objective_value(100)
    self.assertEqual(expected_trial.id, trial.id)
    self.assertEqual(expected_trial.learning_rate, trial.learning_rate)
    self.assertEqual(expected_trial.final_objective, trial.final_objective)
    self.assertListEqual(expected_steps, expected_trial.steps)
    self.assertListEqual(expected_values, expected_trial.objective_values)


class GBMAutoRegressorTest(absltest.TestCase):

  def test_create_features_from_trial(self):
    steps = [10, 20, 30]
    values = [0.1, 0.3, 0.5]

    trial = trial_regression_utils.TrialData(
        id=1,
        learning_rate=0.1,
        final_objective=values[-1],
        steps=steps,
        objective_values=values,
    )
    metric = pyvizier.MetricInformation(
        name=_METRIC_NAME, goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
    )
    conv = converters.TimedLabelsExtractor(
        [
            converters.DefaultModelOutputConverter(
                metric, flip_sign_for_minimization_metrics=True
            ),
        ],
        timestamp='steps',
        value_extraction='raw',
    )

    gbm = trial_regression_utils.GBMAutoRegressor(
        min_points=3,
        target_step=40,
        learning_rate_param_name='learning_rate',
        metric_name=_METRIC_NAME,
        converter=conv,
    )
    features = gbm._create_features_from_trial(trial, end_index=2)
    expected_features = [0.1, 10, 0.5, 20, 0.3, 30, 0.1]
    self.assertListEqual(features, expected_features)

  def test_gbmautoregressor_train_predict(self):
    metric = pyvizier.MetricInformation(
        name=_METRIC_NAME, goal=pyvizier.ObjectiveMetricGoal.MAXIMIZE
    )
    conv = converters.TimedLabelsExtractor(
        [
            converters.DefaultModelOutputConverter(
                metric, flip_sign_for_minimization_metrics=True
            ),
        ],
        timestamp='steps',
        value_extraction='raw',
    )
    gbm = trial_regression_utils.GBMAutoRegressor(
        target_step=50,
        min_points=2,
        learning_rate_param_name='learning_rate',
        metric_name=_METRIC_NAME,
        converter=conv,
        random_state=111,
    )
    with self.subTest('TestModelCreation'):
      steps = [10, 20, 30, 40, 50]
      values1 = [0.1, 0.3, 0.5, 0.6, 0.66]
      values2 = [0.1, 0.25, 0.4, 0.5, 0.55]
      values3 = [0.1, 0.2, 0.35, 0.48, 0.6]

      trial1 = trial_regression_utils.TrialData(
          id=1,
          learning_rate=0.1,
          final_objective=values1[-1],
          steps=steps,
          objective_values=values1,
      )
      pytrial1 = _create_trial_for_testing(
          learning_rate=0.1,
          steps=steps,
          seconds=steps,
          values=values1,
          trial_id=1,
          stop_reason=None,
          metric_name=_METRIC_NAME,
      )
      trial2 = trial_regression_utils.TrialData(
          id=2,
          learning_rate=0.3,
          final_objective=values2[-1],
          steps=steps,
          objective_values=values2,
      )
      pytrial2 = _create_trial_for_testing(
          learning_rate=0.3,
          steps=steps,
          seconds=steps,
          values=values2,
          trial_id=2,
          stop_reason=None,
          metric_name=_METRIC_NAME,
      )
      trial3 = trial_regression_utils.TrialData(
          id=3,
          learning_rate=0.2,
          final_objective=values3[-1],
          steps=steps,
          objective_values=values3,
      )
      pytrial3 = _create_trial_for_testing(
          learning_rate=0.2,
          steps=steps,
          seconds=steps,
          values=values3,
          trial_id=3,
          stop_reason=None,
          metric_name=_METRIC_NAME,
      )
      feat_mat = []
      targets = []
      for trial in [trial1, trial2, trial3]:
        for i, step in enumerate(trial.steps):
          if i < 1 or step >= 50:
            continue
          features = gbm._create_features_from_trial(trial, 2)
          feat_mat.append(features)
          targets.append(trial.objective_values[-1])
      feat_mat = np.array(feat_mat)
      targets = np.array(targets)
      gbdt_param_grid = {'max_depth': [2, 3], 'n_estimators': [50, 100]}
      gbdt_cv = GridSearchCV(
          lightgbm.LGBMRegressor(random_state=111), gbdt_param_grid, cv=2
      )
      gbdt_cv = gbdt_cv.fit(feat_mat, targets)
      best_params = gbdt_cv.best_params_
      ideal_model = lightgbm.LGBMRegressor(**best_params, random_state=111)
      ideal_model = ideal_model.fit(feat_mat, targets)
      gbm.train(vza.CompletedTrials(trials=[pytrial1, pytrial2, pytrial3]))
      pred = gbm._model.predict(feat_mat[[1], :])
      pred_expected = ideal_model.predict(feat_mat[[1], :])
      self.assertAlmostEqual(pred, pred_expected)

    with self.subTest('TestPrediction'):
      steps = [10, 20, 30, 40]
      values = [0.1, 0.3, 0.5, 0.6]
      trial_pred = trial_regression_utils.TrialData(
          id=2,
          learning_rate=0.1,
          final_objective=values[-1],
          steps=steps,
          objective_values=values,
      )
      pytrial_pred = _create_trial_for_testing(
          learning_rate=0.1,
          steps=steps,
          seconds=steps,
          values=values,
          trial_id=2,
          stop_reason=None,
          metric_name=_METRIC_NAME,
      )
      pred = gbm.predict(pytrial_pred)
      features = gbm._create_features_from_trial(trial_pred, end_index=2)
      features = np.array(features).reshape(1, -1)
      pred_expected = ideal_model.predict(features)[0]
      self.assertAlmostEqual(pred, pred_expected)

  def test_converter_in_trial_hallucinator(self):
    steps = [10, 20, 30, 40, 50]
    values = [0.1, 0.3, 0.5, 0.6, 0.66]
    pytrial = _create_trial_for_testing(
        learning_rate=0.1,
        steps=steps,
        seconds=steps,
        values=values,
        trial_id=1,
        stop_reason=None,
        metric_name=_METRIC_NAME,
    )
    options = trial_regression_utils.HallucinationOptions(max_steps=50)
    problem = pyvizier.ProblemStatement()
    problem.search_space.root.add_float_param(
        'learning_rate', min_value=0.0, max_value=1.0, default_value=0.5
    )
    problem.search_space.root.add_int_param(
        'num_layers', min_value=1, max_value=5
    )
    problem.metric_information = [
        pyvizier.MetricInformation(
            name=_METRIC_NAME, goal=pyvizier.ObjectiveMetricGoal.MINIMIZE
        )
    ]
    hallucinator = trial_regression_utils.GBMTrialHallucinator(
        study_config=problem, options=options
    )
    trial = trial_regression_utils.TrialData.from_trial(
        trial=pytrial,
        learning_rate_param_name='learning_rate',
        metric_name=_METRIC_NAME,
        converter=hallucinator._converter,
    )
    self.assertSequenceAlmostEqual(
        trial.objective_values, [0.1, 0.3, 0.5, 0.6, 0.66]
    )


if __name__ == '__main__':
  absltest.main()
