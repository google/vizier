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

"""Tests for gp_ucb_pe."""

import ast
import copy
from typing import Any, Tuple

import jax
import numpy as np
from vizier import pyvizier as vz
from vizier._src.algorithms.core import abstractions
from vizier._src.algorithms.designers import gp_ucb_pe
from vizier._src.algorithms.designers import quasi_random
from vizier._src.algorithms.optimizers import eagle_strategy as es
from vizier._src.algorithms.optimizers import vectorized_base as vb
from vizier.jax import optimizers
from vizier.pyvizier.converters import padding
from vizier.testing import test_studies

from absl.testing import absltest
from absl.testing import parameterized

ensemble_ard_optimizer = optimizers.default_optimizer()


def _extract_predictions(
    metadata: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
  pred = metadata.ns('prediction_in_warped_y_space')
  return (
      np.asarray(ast.literal_eval(pred['mean'])),
      np.asarray(ast.literal_eval(pred['stddev'])),
      np.asarray(ast.literal_eval(pred['stddev_from_all'])),
      float(pred['acquisition']),
      bool(pred['use_ucb'] == 'True'),
  )


class GpUcbPeTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(iters=3, batch_size=5, num_seed_trials=5),
      dict(iters=5, batch_size=1, num_seed_trials=2),
      dict(iters=5, batch_size=3, num_seed_trials=2, ensemble_size=3),
      dict(iters=3, batch_size=5, num_seed_trials=5, applies_padding=True),
      dict(iters=5, batch_size=1, num_seed_trials=2, pe_overwrite=True),
      dict(
          iters=3,
          batch_size=5,
          num_seed_trials=5,
          applies_padding=True,
          optimize_set_acquisition_for_exploration=True,
      ),
      dict(
          iters=3,
          batch_size=5,
          num_seed_trials=5,
          applies_padding=True,
          optimize_set_acquisition_for_exploration=True,
          search_space=test_studies.flat_categorical_space(),
      ),
      dict(
          iters=3,
          batch_size=5,
          num_seed_trials=5,
          applies_padding=True,
          ensemble_size=3,
          turns_on_high_noise_mode=True,
      ),
      dict(iters=3, batch_size=5, num_seed_trials=5, num_metrics=2),
      dict(
          iters=3,
          batch_size=3,
          num_metrics=2,
          applies_padding=True,
          multimetric_promising_region_penalty_type=(
              gp_ucb_pe.MultimetricPromisingRegionPenaltyType.UNION
          ),
      ),
      dict(
          iters=3,
          batch_size=3,
          num_metrics=2,
          applies_padding=True,
          ensemble_size=4,
          multimetric_promising_region_penalty_type=(
              gp_ucb_pe.MultimetricPromisingRegionPenaltyType.INTERSECTION
          ),
      ),
  )
  def test_on_flat_space(
      self,
      iters: int = 5,
      batch_size: int = 1,
      num_seed_trials: int = 1,
      ard_optimizer: str = 'default',
      ensemble_size: int = 1,
      applies_padding: bool = False,
      pe_overwrite: bool = False,
      optimize_set_acquisition_for_exploration: bool = False,
      search_space: vz.SearchSpace = (
          test_studies.flat_continuous_space_with_scaling()
      ),
      turns_on_high_noise_mode: bool = False,
      num_metrics: int = 1,
      multimetric_promising_region_penalty_type: (
          gp_ucb_pe.MultimetricPromisingRegionPenaltyType
      ) = gp_ucb_pe.MultimetricPromisingRegionPenaltyType.AVERAGE,
  ):
    # We use string names so that test case names are readable. Convert them
    # to objects.
    if ard_optimizer == 'default':
      ard_optimizer = optimizers.default_optimizer()
    problem = vz.ProblemStatement(search_space)
    for metric_idx in range(num_metrics):
      problem.metric_information.append(
          vz.MetricInformation(
              name=f'metric{metric_idx}',
              goal=vz.ObjectiveMetricGoal.MAXIMIZE
              if metric_idx % 2 == 0
              else vz.ObjectiveMetricGoal.MINIMIZE,
          )
      )
    vectorized_optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=es.VectorizedEagleStrategyFactory(),
        max_evaluations=100,
    )
    designer = gp_ucb_pe.VizierGPUCBPEBandit(
        problem,
        acquisition_optimizer_factory=vectorized_optimizer_factory,
        num_seed_trials=num_seed_trials,
        ard_optimizer=ard_optimizer,
        metadata_ns='gp_ucb_pe_bandit_test',
        config=gp_ucb_pe.UCBPEConfig(
            ucb_coefficient=10.0,
            explore_region_ucb_coefficient=0.5,
            # Sets the penalty coefficient to 0.0 so that the PE aquisition
            # value is exactly the standard deviation prediction based on all
            # trials.
            cb_violation_penalty_coefficient=0.0,
            ucb_overwrite_probability=0.0,
            pe_overwrite_probability=1.0 if pe_overwrite else 0.0,
            # In high noise mode, the PE acquisition function is always used.
            pe_overwrite_probability_in_high_noise=1.0,
            optimize_set_acquisition_for_exploration=(
                optimize_set_acquisition_for_exploration
            ),
            signal_to_noise_threshold=np.inf
            if turns_on_high_noise_mode
            else 0.0,
            multimetric_promising_region_penalty_type=(
                multimetric_promising_region_penalty_type
            ),
        ),
        ensemble_size=ensemble_size,
        padding_schedule=padding.PaddingSchedule(
            num_trials=padding.PaddingType.MULTIPLES_OF_10
            if applies_padding
            else padding.PaddingType.NONE,
        ),
        rng=jax.random.PRNGKey(1),
    )

    quasi_random_sampler = quasi_random.QuasiRandomDesigner(
        problem.search_space,
    )
    test_trials = quasi_random_sampler.suggest(count=3)

    all_active_trials = []
    all_trials = []
    trial_id = 1
    last_prediction = None
    last_samples = None
    label_rng = jax.random.PRNGKey(1)
    # Simulates batch suggestions with delayed feedback: the first two batches
    # are generated by the designer without any completed trials (but all with
    # active trials). Starting from the third batch, the oldest batch gets
    # completed and updated to the new designer with all the active trials, and
    # the designer then makes a new batch of suggestions. The last two batches
    # of suggestions are again made with only active trials being updated to
    # the designer.
    for idx in range(iters + 2):
      suggestions = designer.suggest(batch_size)
      self.assertLen(suggestions, batch_size)
      for suggestion in suggestions:
        problem.search_space.assert_contains(suggestion.parameters)
        all_active_trials.append(suggestion.to_trial(trial_id))
        all_trials.append(copy.deepcopy(all_active_trials[-1]))
        trial_id += 1
      completed_trials = []
      # Starting from the second until the last but two batch, complete the
      # oldest batch of suggestions.
      if idx > 0 and idx < iters:
        for _ in range(batch_size):
          measurement = vz.Measurement()
          for mi in problem.metric_information:
            label_rng, rng = jax.random.split(label_rng, 2)
            measurement.metrics[mi.name] = float(
                jax.random.uniform(
                    rng,
                    minval=mi.min_value_or(lambda: -10.0),
                    maxval=mi.max_value_or(lambda: 10.0),
                )
            )
          completed_trials.append(
              all_active_trials.pop(0).complete(measurement)
          )
      designer.update(
          completed=abstractions.CompletedTrials(completed_trials),
          all_active=abstractions.ActiveTrials(all_active_trials),
      )
      # After the designer is updated with completed trials, prediction and
      # sampling results are expected to change.
      if len(completed_trials) > 1:
        # test the sample method.
        samples = designer.sample(test_trials, num_samples=5)
        self.assertSequenceEqual(
            samples.shape, (5, 3) if num_metrics == 1 else (5, 3, num_metrics)
        )
        self.assertFalse(np.isnan(samples).any())
        # test the sample method with a different rng.
        samples_rng = designer.sample(
            test_trials, num_samples=5, rng=jax.random.PRNGKey(1)
        )
        self.assertFalse(np.isnan(samples_rng).any())
        self.assertFalse((np.abs(samples - samples_rng) <= 1e-6).all())
        # test the predict method.
        prediction = designer.predict(test_trials)
        self.assertSequenceEqual(
            prediction.mean.shape,
            (3,) if num_metrics == 1 else (3, num_metrics),
        )
        self.assertSequenceEqual(
            prediction.stddev.shape,
            (3,) if num_metrics == 1 else (3, num_metrics),
        )
        self.assertFalse(np.isnan(prediction.mean).any())
        self.assertFalse(np.isnan(prediction.stddev).any())
        if last_prediction is None:
          last_prediction = prediction
          last_samples = samples
        else:
          self.assertFalse(
              (np.abs(last_prediction.mean - prediction.mean) <= 1e-6).all()
          )
          self.assertFalse(
              (np.abs(last_prediction.stddev - prediction.stddev) <= 1e-6).all()
          )
          self.assertFalse((np.abs(last_samples - samples) <= 1e-6).all())

    self.assertLen(all_trials, (iters + 2) * batch_size)

    # The suggestions after the seeds up to the first two batches are expected
    # to be generated by the PE acquisition function.
    for jdx in range(2 * batch_size):
      # Before the designer was updated with enough trials, the suggested
      # batches were seeds, not from acquisition optimization.
      if (jdx // batch_size) * batch_size >= num_seed_trials:
        _, _, _, acq, use_ucb = _extract_predictions(
            all_trials[jdx].metadata.ns('gp_ucb_pe_bandit_test')
        )
        self.assertFalse(use_ucb)
        if not optimize_set_acquisition_for_exploration:
          self.assertGreaterEqual(acq, 0.0, msg=f'suggestion: {jdx}')

    for idx in range(2, iters + 2):
      # Skips seed trials, which are not generated by acquisition function
      # optimization.
      if idx * batch_size < num_seed_trials:
        continue
      set_acq_value = None
      stddev_from_all_list = []
      for jdx in range(batch_size):
        mean, _, stddev_from_all, acq, use_ucb = _extract_predictions(
            all_trials[idx * batch_size + jdx].metadata.ns(
                'gp_ucb_pe_bandit_test'
            )
        )
        if (
            jdx == 0
            and idx < (iters + 1)
            and not pe_overwrite
            and not turns_on_high_noise_mode
        ):
          # Except for the last batch of suggestions, the acquisition value of
          # the first suggestion in a batch is expected to be UCB, which
          # combines the predicted mean based only on completed trials and the
          # predicted standard deviation based on all trials. Only checks the
          # single-metric case because the acquisition value in the multi-metric
          # case is randomly scalarized.
          if num_metrics == 1:
            self.assertAlmostEqual(mean + 10.0 * stddev_from_all, acq)
          self.assertTrue(use_ucb)
          continue

        self.assertFalse(use_ucb)
        if optimize_set_acquisition_for_exploration:
          stddev_from_all_list.append(stddev_from_all)
          if set_acq_value is None:
            set_acq_value = acq
          else:
            self.assertAlmostEqual(set_acq_value, acq)
        else:
          # Because `ucb_overwrite_probability` is set to 0.0, when the designer
          # makes suggestions without seeing newer completed trials, it uses the
          # Pure-Exploration acquisition function. In this test, that happens
          # on the entire last batch and the second until the last suggestions
          # in every batch. The Pure-Exploration acquisition values are standard
          # deviation predictions based on all trials (completed and pending).
          self.assertAlmostEqual(
              acq,
              np.mean(stddev_from_all),
              msg=f'batch: {idx}, suggestion: {jdx}',
          )
      if optimize_set_acquisition_for_exploration:
        geometric_mean_of_pred_cov_eigs = np.exp(
            set_acq_value / (batch_size - 1)
        )
        arithmetic_mean_of_pred_cov_eigs = np.mean(
            np.square(stddev_from_all_list)
        )
        self.assertLessEqual(
            geometric_mean_of_pred_cov_eigs, arithmetic_mean_of_pred_cov_eigs
        )

  def test_ucb_overwrite(self):
    problem = vz.ProblemStatement(
        test_studies.flat_continuous_space_with_scaling()
    )
    problem.metric_information.append(
        vz.MetricInformation(
            name='metric', goal=vz.ObjectiveMetricGoal.MAXIMIZE
        )
    )
    vectorized_optimizer_factory = vb.VectorizedOptimizerFactory(
        strategy_factory=es.VectorizedEagleStrategyFactory(),
        max_evaluations=100,
    )
    designer = gp_ucb_pe.VizierGPUCBPEBandit(
        problem,
        acquisition_optimizer_factory=vectorized_optimizer_factory,
        metadata_ns='gp_ucb_pe_bandit_test',
        num_seed_trials=1,
        config=gp_ucb_pe.UCBPEConfig(
            ucb_coefficient=10.0,
            explore_region_ucb_coefficient=0.5,
            cb_violation_penalty_coefficient=10.0,
            ucb_overwrite_probability=1.0,
            pe_overwrite_probability=0.0,
            signal_to_noise_threshold=0.0,
        ),
        padding_schedule=padding.PaddingSchedule(
            num_trials=padding.PaddingType.MULTIPLES_OF_10
        ),
        rng=jax.random.PRNGKey(1),
    )

    trial_id = 1
    batch_size = 5
    iters = 3
    rng = jax.random.PRNGKey(1)
    all_trials = []
    # Simulates a batch suggestion loop that completes a full batch of
    # suggestions before asking for the next batch.
    for _ in range(iters):
      suggestions = designer.suggest(count=batch_size)
      self.assertLen(suggestions, batch_size)
      completed_trials = []
      for suggestion in suggestions:
        problem.search_space.assert_contains(suggestion.parameters)
        trial_id += 1
        measurement = vz.Measurement()
        for mi in problem.metric_information:
          measurement.metrics[mi.name] = float(
              jax.random.uniform(
                  rng,
                  minval=mi.min_value_or(lambda: -10.0),
                  maxval=mi.max_value_or(lambda: 10.0),
              )
          )
          rng, _ = jax.random.split(rng)
        completed_trials.append(
            suggestion.to_trial(trial_id).complete(measurement)
        )
      all_trials.extend(completed_trials)
      designer.update(
          completed=abstractions.CompletedTrials(completed_trials),
          all_active=abstractions.ActiveTrials(),
      )

    self.assertLen(all_trials, iters * batch_size)

    for idx, trial in enumerate(all_trials):
      if idx < batch_size:
        # Skips the first batch of suggestions, which are generated by the
        # seeding designer, not acquisition function optimization.
        continue
      # Because `ucb_overwrite_probability` is 1 and `pe_overwrite_probability`
      # is 0, all suggestions after the first batch are expected to be generated
      # by UCB. Within a batch, the first suggestion's UCB value is expected to
      # use predicted standard deviation based only on completed trials, while
      # the UCB values of the second to the last suggestions are expected to use
      # the predicted standard deviations based on completed and active trials.
      mean, stddev, stddev_from_all, acq, use_ucb = _extract_predictions(
          trial.metadata.ns('gp_ucb_pe_bandit_test')
      )
      self.assertAlmostEqual(
          mean + 10.0 * (stddev_from_all if idx % batch_size > 0 else stddev),
          acq,
      )
      self.assertTrue(use_ucb)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
