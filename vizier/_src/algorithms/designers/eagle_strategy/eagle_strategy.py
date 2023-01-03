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

"""Pythia Eagle Strategy Designer.

Implements a variation of Eagle Strategy without the Levy random walk, aka
Firefly Algorithm (FA).

Reference: Yang XS. (2009) Firefly Algorithms for Multimodal Optimization.
In: Stochastic Algorithms: Foundations and Applications (SAGA) 2009.
DOI: https://doi.org/10.1007/978-3-642-04944-6_14


Firefly Algorithm Summary
=========================
FA is a genetic algorithm that maintains a pool of fireflies. Each
firefly emits a light whose intensity is non-decreasing in (or simply equal
to) the objective value. Each iteration, a firefly chases after a brighter
firefly, but the brightness it perceives decreases in distance. This allows
multiple "clusters" to form, as opposed to all fireflies collapsing to a
single point. Not included in the original algorithm, we added "repulsion" which
in addition to the "attraction" forces, meaning fireflies move towards the
bright spots as well as away from the dark spots. We also support non-decimal
parameter types (categorical, discrete, integer), and treat them uniquely when
computing distance, adding pertrubation, and mutating fireflies.

For more details, see the linked paper.

OSS Vizier Implementation Summary
=================================
We maintain a pool of fireflies. Each firefly stores the best trial it created.
During 'Suggest' each firefly on its turn is used to suggest a new trial. The
best trial parameters are used to compute distances and to generate new
suggested parameters by calling the '_mutate' and '_perturb' methods. During
'Update' the trial results comeback and we create a new firefly if needed or
update/remove the associated firefly in the pool. To facilitate this
association, during 'Suggest', the newly created suggested trial stores in
its metadata the id of its parent firefly ('parent_fly_id'). Throughout the
study lifetime, the same firefly could be associated with multiple trials
created from it.

In our stateful implementation, across 'Suggest' calls, we persist the pool of
fireflies and update it using previously unseen COMPLETED trials. We also
persist the last firefly used to generate a suggestion so to not use
sequentially the same firefly even if there are ACTIVE trials. Lastly we persist
the maximum value of firefly id created, to ensure that each firefly has its own
unique id.
"""

import json
import logging
import time
from typing import Optional, Sequence

import attr
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.eagle_strategy import eagle_strategy_utils
from vizier._src.algorithms.designers.eagle_strategy import serialization
from vizier._src.algorithms.random import random_sample
from vizier.interfaces import serializable

EagleStrategyUtils = eagle_strategy_utils.EagleStrategyUtils
FireflyAlgorithmConfig = eagle_strategy_utils.FireflyAlgorithmConfig
Firefly = eagle_strategy_utils.Firefly
FireflyPool = eagle_strategy_utils.FireflyPool


class EagleStrategyDesigner(vza.PartiallySerializableDesigner):
  """The eagle strategy partially serializable designer."""

  def __init__(
      self,
      problem_statement: vz.ProblemStatement,
      *,
      config: Optional[FireflyAlgorithmConfig] = None,
      seed: Optional[int] = None,
      verbose: bool = True,
  ):
    """Initializes the Eagle Strategy desiger.

    Args:
      problem_statement: A problem description including the search space.
      config: The Firefly algorithm hyperparameters.
      seed: A seed to deterministically generate samples from random variables.
      verbose: Whether to display

    Raises:
      Exception: if the problem statement includes condional search space,
        mutli-objectives or safety metrics.
    """
    if seed is None:
      # When a key is not provided it will be set based on the current time to
      # ensure non-repeated behavior.
      seed = int(time.time())
      logging.info(
          ('A seed was not provided to Eagle Strategy designer constructor. '
           'Setting the seed to %s'),
          str(seed),
      )

    self._rng = np.random.default_rng(seed=seed)
    self.problem = problem_statement
    self.config = config or FireflyAlgorithmConfig()
    self._utils = EagleStrategyUtils(problem_statement, self.config, self._rng)
    self._firefly_pool = FireflyPool(
        utils=self._utils, capacity=self._utils.compute_pool_capacity())

    if problem_statement.search_space.is_conditional:
      raise ValueError(
          "Eagle Strategy designer doesn't support conditional parameters.")
    if not problem_statement.is_single_objective:
      raise ValueError(
          "Eagle Strategy designer doesn't support multi-objectives.")
    if problem_statement.is_safety_metric:
      raise ValueError(
          "Eagle Strategy designer doesn't support safety metrics.")
    if not self._utils.is_linear_scale:
      raise ValueError(
          "Eagle Strategy designer doesn't support non-linear scales.")

    logging.info(
        ('Eagle Strategy designer initialized. Pool capacity: %s. '
         'Eagle config:\n%s\nProblem statement:\n%s'),
        self._utils.compute_pool_capacity(),
        json.dumps(attr.asdict(self.config), indent=2),
        self.problem,
    )

  def dump(self) -> vz.Metadata:
    """Dumps the current state of the algorithm.

    Returns:
      Metadata with the current pool serialized.
    """
    metadata = vz.Metadata()
    metadata.ns('eagle')['rng'] = serialization.serialize_rng(self._rng)
    metadata.ns('eagle')['firefly_pool'] = (
        serialization.partially_serialize_firefly_pool(self._firefly_pool))
    metadata.ns('eagle')['serialization_version'] = 'v1'
    metadata.ns('eagle')['dump_timestamp'] = str(time.time())
    logging.info('Dump metadata:\n%s', metadata.ns('eagle'))
    return metadata

  def load(self, metadata: vz.Metadata) -> None:
    """Loads the current state of the algorithm run.

    The function is used as part of the `_restore_designer` method in
    PartiallySerializableDesignerPolicy to restore the state of the designer
    after it was already initialized with the problem statement.

    1. Set EagleStrategy, EagleStrategyUtils with the recovered random generator
    2. Recover the FireflyPool and populate it with the Fireflies.

    Args:
      metadata: Metadata
    """
    if metadata.ns('eagle').get('serialization_version', default=None) is None:
      # First time the designer is called, so the namespace doesn't exist yet.
      logging.info('Eagle designer was called for the first time. No state was'
                   ' recovered.')
    else:
      try:
        logging.info('Load metadata:\n%s', metadata.ns('eagle'))
        self._rng = serialization.restore_rng(metadata.ns('eagle')['rng'])
      except Exception as e:
        raise serializable.FatalDecodeError(
            "Couldn't load random generator from metadata.") from e
      self._utils.rng = self._rng
      logging.info('Restored rng:\n%s', self._rng)
      try:
        firefly_pool = metadata.ns('eagle')['firefly_pool']
        self._firefly_pool = serialization.restore_firefly_pool(
            self._utils, firefly_pool)
        logging.info('Restored firefly pool:\n%s', self._firefly_pool)
      except Exception as e:
        raise serializable.HarmlessDecodeError(
            "Couldn't load firefly pool from metadata.") from e
      logging.info(
          ('Eagle designer restored state from timestamp %s. Firefly pool'
           ' now contains %s fireflies.'),
          metadata.ns('eagle')['dump_timestamp'],
          self._firefly_pool.size,
      )

  def suggest(self,
              count: Optional[int] = None) -> Sequence[vz.TrialSuggestion]:
    """Suggests trials."""
    return [self._suggest_one() for _ in range(max(count, 1))]

  def _suggest_one(self) -> vz.TrialSuggestion:
    """Generates a single suggestion based on the current pool of flies.

    In order to generate a trial suggestion, we find the next fly that should be
    moved and create a copy of it. Then we mutate and perturb its trial
    parameters inplace and assign them to the suggested trial.

    Returns:
      The suggested trial
      The parent fly Id
    """
    suggested_trial = vz.TrialSuggestion()
    if self._firefly_pool.size < self._firefly_pool.capacity:
      # Pool is underpopulated. Generate a random trial parameters.
      # (b/243518714): Use random policy/designer to generate parameters.
      suggested_parameters = random_sample.sample_parameters(
          self._rng, self.problem.search_space)
      # Create a new parent fly id and assign it to the trial, this will be
      # used during Update to match the trial to its parent fly in the pool.
      parent_fly_id = self._firefly_pool.generate_new_fly_id()
      logging.info('Pool is underpopulated. Generated random trial parameters.')
    else:
      # The pool is full. Use a copy of the next fly in line to be moved.
      moving_fly = self._firefly_pool.get_next_moving_fly_copy()
      self._mutate_fly(moving_fly)
      self._perturb_fly(moving_fly)
      suggested_parameters = moving_fly.trial.parameters
      parent_fly_id = moving_fly.id_
      logging.info('Created a trial from parent fly ID: %s.', parent_fly_id)

    suggested_trial.parameters = suggested_parameters
    suggested_trial.metadata.ns('eagle')['parent_fly_id'] = str(parent_fly_id)
    logging.info(
        'Suggested trial: %s',
        self._utils.display_trial(suggested_trial.to_trial(-1)),
    )
    return suggested_trial

  def _mutate_fly(self, moving_fly: Firefly) -> None:
    """Mutates fly's trial parameters inplace.

    Apply pulls from the rest of the pool's flies on `moving_fly` to mutate
    its trial's parameters.

    Args:
      moving_fly: the fire from the pool to mutate its trial parameters.
    """
    # Access the moving fly's trial parameters to be modified.
    mutated_parameters = moving_fly.trial.parameters
    # Shuffle the ordering, so to apply the attracting and repelling forces
    # in random order every time. Creates a deep copy of the pool members.
    shuffled_flies = self._firefly_pool.get_shuffled_flies(self._rng)
    for other_fly in shuffled_flies:
      is_other_fly_better = self._utils.is_better_than(other_fly.trial,
                                                       moving_fly.trial)
      # Compute 'other_fly' pull weights by parameter type. In the paper
      # this is called "attractivness" and is denoted by beta(r).
      pull_weights = self._utils.compute_pull_weight_by_type(
          other_fly.trial.parameters, mutated_parameters, is_other_fly_better)
      # Apply the pulls from 'other_fly' on the moving fly's parameters.
      for param_config in self.problem.search_space.parameters:
        pull_weight = pull_weights[param_config.type]
        # Accentuate 'other_fly' pull using 'exploration_rate'.
        if pull_weight > 0.5:
          explore_pull_weight = (
              self.config.explore_rate * pull_weight +
              (1 - self.config.explore_rate) * 1.0)
        else:
          explore_pull_weight = self.config.explore_rate * pull_weight
        # Update the parameters using 'other_fly' and 'explore_pull_rate'.
        mutated_parameters[param_config.name] = (
            self._utils.combine_two_parameters(
                param_config,
                other_fly.trial.parameters,
                mutated_parameters,
                explore_pull_weight,
            ))

  def _perturb_fly(self, moving_fly: Firefly) -> None:
    """Perturbs the fly's trial parameters inplace.

    Apply random perturbation to the fly's parameter values based on the
    moving_fly's 'perturbation' value.

    Args:
      moving_fly: the fire from the pool to mutate its trial parameters.
    """
    suggested_parameters = moving_fly.trial.parameters
    perturbations = self._utils.create_perturbations(moving_fly.perturbation)
    for i, param_config in enumerate(self.problem.search_space.parameters):
      perturbed_value = self._utils.perturb_parameter(
          param_config,
          suggested_parameters[param_config.name].value,
          perturbations[i],
      )
      suggested_parameters[param_config.name] = perturbed_value

  def update(self, delta: vza.CompletedTrials) -> None:
    """Update the pool.

    Iterate over new completed trials and update the firefly pool. Every trial
    that was suggested from Eagle Strategy has a metadata data containing the
    parent fly id. For trials that were added to the study externally we assign
    a new parent fly id.

    Args:
      delta:
    """
    for trial in delta.completed:
      # Replaces trial metric name with a canonical metric name, which makes the
      # serialization and deserialization simpler.
      trial = self._utils.standardize_trial_metric_name(trial)
      if not trial.metadata.ns('eagle').get('parent_fly_id'):
        # Trial was not generated from Eagle Strategy. Set a new parent fly id.
        trial.metadata.ns('eagle')['parent_fly_id'] = str(
            self._firefly_pool.generate_new_fly_id())
        logging.info('Received a trial without parent_fly_id.')
      self._update_one(trial)

  def _update_one(self, trial: vz.Trial) -> None:
    """Update the pool using a single trial."""
    parent_fly_id = int(trial.metadata.ns('eagle').get('parent_fly_id'))
    parent_fly = self._firefly_pool.find_parent_fly(parent_fly_id)
    if parent_fly is None:
      logging.info('Parent fly ID: %s is not in pool.', parent_fly_id)
      if trial.infeasible:
        # Ignore infeasible trials without parent fly.
        logging.info('Got infeasible trial without a parent fly in the pool.')
      elif self._firefly_pool.size < self._firefly_pool.capacity:
        # Pool is below capacity. Create a new firefly or update existing one.
        logging.info(
            'Pool is below capacity (size=%s). Invoke create or update pool.',
            self._firefly_pool.size,
        )
        self._firefly_pool.create_or_update_fly(trial, parent_fly_id)
        return
      else:
        # Pool is at capacity. Try assigning a parent to utilize the trial info.
        parent_fly = self._assign_closest_parent(trial)

    if parent_fly is None:
      # Parent fly wasn't established. No need to continue.
      logging.info('Parent fly was not established.')
      return

    elif not trial.infeasible and self._utils.is_better_than(
        trial, parent_fly.trial):
      # If there's an improvement, we update the parent with the new trial.
      logging.info(
          'Good step:\nParent trial (val=%s): %s\nChild trial (val=%s): %s\n',
          self._utils.get_metric(parent_fly.trial),
          self._utils.display_trial(parent_fly.trial),
          self._utils.get_metric(trial),
          self._utils.display_trial(trial),
      )
      parent_fly.trial = trial
      parent_fly.generation += 1
    else:
      # If there's no improvement, we penalize the parent by decreasing its
      # exploration capability and potenitally remove it from the pool.
      logging.info(
          'Bad step:\nParent trial (val=%s): %s\nChild trial (val=%s): %s\n',
          self._utils.get_metric(parent_fly.trial),
          self._utils.display_trial(parent_fly.trial),
          self._utils.get_metric(trial),
          self._utils.display_trial(trial),
      )
      self._penalize_parent_fly(parent_fly, trial)

  def _assign_closest_parent(self, trial: vz.Trial) -> Optional[Firefly]:
    """Finds the closest parent fly and checks that the trial improves on it.

    Note that the trial's `parent_fly_id` won't exist in the pool when:

    1. The trial was randomly generated during 'Suggest', and therefore is not
    associated with any existing fly in the pool.

    2. The fly was removed from the pool but not all associated trials were
    processed yet.

    3. The trial was added to the study independently of Eagle Strategy
    suggestions.

    Args:
      trial:

    Returns:
      None or a fly from the pool that is closest to the trial.
    """
    closest_parent_fly = self._firefly_pool.find_closest_parent(trial)
    if self._utils.is_better_than(trial, closest_parent_fly.trial):
      # Only returns the closest fly if there's improvement. Otherwise, we don't
      # return it to not count it as a failure, as the closest parent is not
      # reponsible for it.
      return closest_parent_fly

  def _penalize_parent_fly(self, parent_fly: Firefly, trial: vz.Trial) -> None:
    """Penalizes a parent fly.

    The method is called on a fly after its generated trial didn't improve the
    objective function. The fly is penalized by decreasing its exploration
    capability and potentially removing it from the pool.

    Args:
      parent_fly: The parent fly with to be penalized.
      trial: The generated trial from the parent fly that didn't imporove.
    """
    if trial.parameters == parent_fly.trial.parameters:
      # If the new trial is identical to the parent trial, it means that the
      # fly is stuck, and so we increase its perturbation.
      parent_fly.perturbation = min(parent_fly.perturbation * 10,
                                    self.config.max_perturbation)
      logging.info(
          ('Penalize Parent Id: %s. Parameters are stuck. '
           'New perturbation factor: %s'),
          parent_fly.id_,
          parent_fly.perturbation,
      )
    else:
      # Otherwise, penalize the parent by decreasing its perturbation factor.
      parent_fly.perturbation *= 0.9
      logging.info(
          ('Penalize Parent Id: %s. Decrease perturbation factor. '
           'New perturbation factor: %s'),
          parent_fly.id_,
          parent_fly.perturbation,
      )
    if parent_fly.perturbation < self.config.perturbation_lower_bound:
      # If the perturbation factor is too low we attempt to eliminate the
      # unsuccessful parent fly from the pool.
      if self._firefly_pool.size == self._firefly_pool.capacity:
        # Only remove if the pool is at capacity. This is critical in studies
        # with few feasible/safe trials to retain the feasible trials.
        if not self._firefly_pool.is_best_fly(parent_fly):
          # Check that the fly is not the best one we have thus far.
          self._firefly_pool.remove_fly(parent_fly)
          logging.info('Removed fly ID: %s from pool.', parent_fly.id_)
