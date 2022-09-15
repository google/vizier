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

import codecs
import copy
import pickle
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from absl import logging
import attr
from jax import random
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.eagle_strategy import utils
from vizier._src.algorithms.random import random_sample

PRNGKey = Any


@attr.define
class FireflyAlgorithmConfig:
  """Configuration hyperparameters for Eagle Strategy / Firefly Algorithm."""
  perturbation: float = 1e-2
  categorical_perturbation_factor: float = 25
  pure_categorical_perturbation: float = 0.1
  discrete_perturbation_factor: float = 10.0
  perturbation_lower_bound: float = 1e-3
  gravity: float = 1.0
  visibility: float = 1.0
  categorical_visibility: float = 0.2
  discrete_visibility: float = 1.0
  negative_gravity: float = 0.02
  firefly_pool_size_factor: float = 1.2
  explore_rate: float = 1.0
  max_perturbation: float = 0.5


@attr.define
class _Firefly:
  """The Firefly class represents a single firefly in the pool.

  Attributes:
    id_: A unique firefly identifier. This is used to associate trials with
      their parent fireflies.
    perturbation_factor: Controls the exploration. Signifies the amount of
      perturbation to add to the generated trial parameters. The value of
      'perturbation_factor' keeps decreasing if the generatd trial doesn't
      improve the objective function until it reaches 'perturbation_lower_bound'
      in which case we remove the firefly from the pool.
    generation: The number of "successful" (better than the last trial) trials
      suggested from the firefly
    trial: The best trial associated with the firefly.
  """
  id_: int = attr.field(validator=attr.validators.instance_of(int))
  perturbation_factor: float = attr.field(
      validator=attr.validators.instance_of(float))
  generation: int = attr.field(validator=attr.validators.instance_of(int))
  trial: vz.Trial = attr.field(validator=attr.validators.instance_of(vz.Trial))


@attr.define
class _FireflyPool:
  """The class maintains the Firefly pool and relevent operations.

  Attributes:
    _problem_statement: A description of the optimization problem.
    _config: A firefly algorithm configuration hyperparameters.
    _capacity: The maximum number of flies that the pool could store.
    _pool: A dictionary of Firefly objects organized by firefly id.
    _last_id: The last firefly id used to generate a suggestion. It's persistent
      across calls to ensure we don't use the same fly repeatedly.
    _max_pool_id: The maximum value of any fly id ever created. It's persistent
      persistent accross calls to ensure unique ids even if trails were deleted.
    size: The current number of flies in the pool.
    capacity: The maximum number of flies that the pool could store.
  """
  _problem_statement: vz.ProblemStatement = attr.field(
      validator=attr.validators.instance_of(vz.ProblemStatement))

  _config: FireflyAlgorithmConfig = attr.field(
      validator=attr.validators.instance_of(FireflyAlgorithmConfig))

  _capacity: int = attr.field(validator=attr.validators.instance_of(int))

  _pool: Dict[int, _Firefly] = attr.field(
      init=False, default=attr.Factory(dict))

  _last_id: int = attr.field(init=False, default=0)

  _max_pool_id: int = attr.field(init=False, default=0)

  @property
  def size(self) -> int:
    return len(self._pool)

  @property
  def capacity(self) -> int:
    return self._capacity

  def remove_fly(self, fly: _Firefly):
    """Removes a fly from the pool."""
    del self._pool[fly.id_]

  def get_shuffled_flies(self, key: PRNGKey) -> Tuple[PRNGKey, List[_Firefly]]:
    """Shuffles the fireflies and returns them as a list."""
    return random_sample.shuffle_list(key, list(self._pool.values()))

  def generate_new_fly_id(self) -> int:
    """Generates a unique fly id to identify a fly in the pool."""
    self._max_pool_id += 1
    return self._max_pool_id - 1

  def get_next_moving_fly_copy(self) -> _Firefly:
    """Finds the next fly, creates a copy of it and updates '_last_id'.

    To find the next moving fly, we start from '_last_id' + 1 and incremently
    check whether the index has fly id in the pool. When reaching 'max_pool_id'
    we reset the index. We return a copy of the first fly we find an index for.
    Before returning we also set '_last_id' to the next moving fly id.

    Note that we don't assume the existance of '_last_id' in the pool, as the
    fly with `_last_id` might be removed from the pool.

    Returns:
      A copy of the next moving fly.
    """
    current_fly_id = self._last_id + 1
    while current_fly_id != self._last_id:
      if current_fly_id > self._max_pool_id:
        # Passed the maximum id. Start from the first one as ids are monotonic.
        current_fly_id = next(iter(self._pool))
      if current_fly_id in self._pool:
        self._last_id = current_fly_id
        return copy.deepcopy(self._pool[current_fly_id])
      current_fly_id += 1

    logging.info("Couldn't find another fly in the pool to move.")
    return copy.deepcopy(self._pool[self._last_id])

  def is_best_fly(self, fly: _Firefly) -> bool:
    """Checks if the 'fly' has the best final measurement in the pool."""
    for other_fly_id, other_fly in self._pool.items():
      if other_fly_id != fly.id_ and utils.better_than(
          self._problem_statement, other_fly.trial, fly.trial):
        return False
    return True

  def find_parent_fly(self, trial: vz.Trial) -> Optional[_Firefly]:
    """Obtains the parent firefly associated with the trial.

    Extract the associated parent id from the trial's metadata and attempt
    to find the parent firefly in the pool. If it doesn't exist return None.

    Args:
      trial:

    Returns:
      Firefly or None.
    """
    parent_fly_id = trial.metadata.ns('eagle').get('parent_fly_id', cls=int)
    parent_fly = self._pool.get(parent_fly_id, None)
    return parent_fly

  def find_closest_parent(self, trial: vz.Trial) -> _Firefly:
    """Finds the closest fly in the pool to a given trial."""
    if not self._pool:
      raise Exception('Pool was empty when searching for closest parent.')

    min_dist, closest_parent = float('inf'), next(iter(self._pool.values()))
    for other_fly in self._pool.values():
      curr_dist = utils.compute_cononical_distance(
          other_fly.trial.parameters, trial.parameters,
          self._problem_statement.search_space)
      if curr_dist < min_dist:
        min_dist = curr_dist
        closest_parent = other_fly

    return closest_parent

  def create_or_update_fly(self, trial: vz.Trial) -> None:
    """Creates a new fly in the pool, or update an existing one.

    The newly created fly's 'id_' is assigned with the 'parent_fly_id' taken
    from the trial's metadata, as the fly's id is determined during the
    'Suggest' method and stored in the metadata.

    If the 'parent_fly_id' already exists in the pool (which is less common) we
    update the associated fly with the better trial. This scenario could happen
    if for example a batch larger than the pool capacity was suggested and
    trials associated with the same fly were reported as COMPLETED sooner than
    the other trials, so when the pool is constructed during 'Update' we
    encounter trials from the same fly before the pool is at capactiy.

    Args:
      trial:
    """
    # Extract the parent fly id from the trial. Trials that were added to the
    # study but not suggested from Eagle Strategy will be assigned a
    # 'parent_fly_id' during 'Update'.
    parent_fly_id = trial.metadata.ns('eagle').get('parent_fly_id', cls=int)
    if parent_fly_id in self._pool:
      # Parent fly id already in pool. Update trial if there was improvement.
      if utils.better_than(self._problem_statement, trial,
                           self._pool[parent_fly_id].trial):
        self._pool[parent_fly_id].trial = trial
    else:
      # Create a new Firefly in pool.
      new_fly = _Firefly(
          id_=parent_fly_id,
          generation=1,
          perturbation_factor=self._config.perturbation,
          trial=trial)
      self._pool[parent_fly_id] = new_fly


class EagleStrategyDesiger(vza.PartiallySerializableDesigner):
  """The eagle strategy partially serializable designer."""

  def __init__(self,
               problem_statement: vz.ProblemStatement,
               *,
               config: Optional[FireflyAlgorithmConfig] = None,
               key: Optional[PRNGKey] = None,
               verbose: int = 0):
    """Initializes the Eagle Strategy desiger.

    When a key is not provided it will be set based on the current time to
    ensure non-repeated behavior.

    Args:
      problem_statement: A problem description including the search space.
      config: The Firefly algorithm hyperparameters.
      key: A key to deterministically generate samples from random variables.
      verbose: Controls logging verbosity. Higher values mean more logging.

    Raises:
      Exception: if the problem statement includes condional search space,
        mutli-objectives or safety metrics.
    """
    # TODO: We now assume linear scale. Need to address other types.
    # TODO: Add validation that all parameters are linear scale.

    if problem_statement.search_space.is_conditional:
      raise ValueError(
          "Eagle Strategy designer doesn't support conditional parameters.")
    if not problem_statement.is_single_objective:
      raise ValueError(
          "Eagle Strategy designer doesn't support multi-objectives.")
    if problem_statement.is_safety_metric:
      raise ValueError(
          "Eagle Strategy designer doesn't support safety metrics.")

    self.problem = problem_statement
    self.config = config or FireflyAlgorithmConfig()
    pool_capacity = utils.compute_pool_capacity(
        self.n_parameters, self.config.firefly_pool_size_factor)
    self._firefly_pool = _FireflyPool(self.problem, self.config, pool_capacity)
    # Jax key to generate samples from random distributions.
    if key is None:
      key = random.PRNGKey(int(time.time()))
      logging.info(
          'A key was not provided to Eagle Strategy designer constructor. '
          'Setting the key to %s', str(key))
    self._key = key
    self._verbose = verbose

  @property
  def n_parameters(self) -> int:
    return len(self.problem.search_space.parameters)

  @property
  def search_space(self) -> vz.SearchSpace:
    return self.problem.search_space

  def dump(self) -> vz.Metadata:
    """Dumps the current state of the algorithm run.

    Returns:
      Metadata with the current pool serialized.
    """
    # TODO: Replace pickle serialization.
    # Serialize the pool to bytes, encode it in base64 and convert to string.
    serialized_pool = codecs.encode(pickle.dumps(self._firefly_pool),
                                    'base64').decode()
    serialized_key = codecs.encode(pickle.dumps(self._key), 'base64').decode()
    metadata = vz.Metadata()
    metadata.ns('eagle')['pool'] = serialized_pool
    metadata.ns('eagle')['key'] = serialized_key
    return metadata

  def load(self, metadata: vz.Metadata) -> None:
    """Load the current state of the algorithm run.

    During the first time the designer run we haven't had a chance to create
    the metadata and so we assign default values.

    Args:
      metadata: Metadata
    """
    # TODO: Replace pickle serialization.
    if metadata.namespaces():
      serialized_pool: str = metadata.ns('eagle').get('pool')
      serialized_key: str = metadata.ns('eagle').get('key')
      # Convert string to bytes, decode it in base64 and convert to object.
      self._firefly_pool = pickle.loads(
          codecs.decode(serialized_pool.encode(), 'base64'))
      self._key = pickle.loads(codecs.decode(serialized_key.encode(), 'base64'))
    else:
      # First time the designer is called, so the namespace doesn't exist yet.
      logging.info(
          'Eagle Strategy was called for the first time. No state was recovered.'
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
      Trial with the modified parameters.
    """
    suggested_trial = vz.TrialSuggestion()
    if self._firefly_pool.size < self._firefly_pool.capacity:
      # If the pool is underpopulated attempt to randomize a trial.
      # TODO: Use random policy/designer to generate parameters.
      self._key, explore = random_sample.sample_bernoulli(
          self._key, self.config.explore_rate)
      if self._firefly_pool.size == 0 or explore:
        self._key, suggested_trial.parameters = random_sample.sample_input_parameters(
            self._key, self.problem.search_space)
        # Create a new parent fly id and assign it to the trial, this will be
        # used during Update to match the trial to its parent fly in the pool.
        suggested_trial.metadata.ns('eagle')['parent_fly_id'] = str(
            self._firefly_pool.generate_new_fly_id())
        return suggested_trial
    # The pool is full. Use a copy of the next fly in line to be moved.
    moving_fly = self._firefly_pool.get_next_moving_fly_copy()
    moving_fly_id = moving_fly.id_
    # Attach the moving fly id as the trial's parent fly id.
    suggested_trial.metadata.ns('eagle')['parent_fly_id'] = str(moving_fly_id)
    # Modify the moving fly's trial parameters. Assign to the suggested trial.
    self._mutate_fly(moving_fly)
    self._perturb_fly(moving_fly)
    suggested_trial.parameters = moving_fly.trial.parameters
    # TODO: Add more detailed logging for debugging purposes.
    suggested_trial.metadata.ns('eagle')['description'] = str(moving_fly_id)
    return suggested_trial

  def _mutate_fly(self, moving_fly: _Firefly) -> None:
    """Mutates fly's trial parameters.

    Apply pulls from the rest of the pool's flies on `moving_fly` to mutate
    its trial's parameters.

    Args:
      moving_fly: the fire from the pool to mutate its trial parameters.
    """
    # Access the moving fly's trial parameters to be modified.
    mutated_parameters = moving_fly.trial.parameters

    # Initialize a dictionary to store the total pull weights for debugging.
    total_pull_weights = {
        vz.ParameterType.DOUBLE: 0.0,
        vz.ParameterType.DISCRETE: 0.0,
        vz.ParameterType.INTEGER: 0.0,
        vz.ParameterType.CATEGORICAL: 0.0
    }
    # Shuffle the ordering, so to apply the attracting and repelling forces
    # in random order every time. Creates a deep copy of the pool members.
    self._key, shuffled_flies = self._firefly_pool.get_shuffled_flies(self._key)
    for other_fly in shuffled_flies:
      is_other_fly_better = utils.better_than(self.problem, other_fly.trial,
                                              moving_fly.trial)
      # Compute the pull weights by parameter type of 'other_fly'. In the paper
      # this is called "attractivness" and is denoted by beta(r).
      pull_weights = utils.compute_pull_weight_by_type(
          self.config.gravity, self.config.negative_gravity,
          self.config.visibility, self.config.categorical_visibility,
          self.config.discrete_visibility, self.problem.search_space,
          other_fly.trial.parameters, mutated_parameters, is_other_fly_better)

      for param_type, type_pull_weight in pull_weights.items():
        total_pull_weights[param_type] += type_pull_weight

      # Apply the pulls from 'other_fly' on the moving fly's parameters.
      for param_config in self.problem.search_space.parameters:
        type_pull_weight = pull_weights[param_config.type]
        # Accentuate 'other_fly' pull using 'exploration_rate' such that if the
        # pull weight favors 1.0, push pull weight closer to 1.0 and vice versa.
        if type_pull_weight > 0.5:
          explore_pull_rate = self.config.explore_rate * type_pull_weight + (
              1 - self.config.explore_rate) * 1.0
        else:
          explore_pull_rate = self.config.explore_rate * type_pull_weight
        # Update the parameters using 'other_fly' and 'explore_pull_rate'.
        self._key, mutated_parameters[
            param_config.name] = utils.combine_two_parameters(
                self._key, param_config, other_fly.trial.parameters,
                mutated_parameters, explore_pull_rate)

  def _perturb_fly(self, moving_fly: _Firefly) -> None:
    """Perturbs the fly's trial parameters.

    Apply random perturbation to the fly's parameter values based on the
    moving_fly's 'perturbation' value.

    Args:
      moving_fly: the fire from the pool to mutate its trial parameters.
    """
    # Access the moving fly's trial parameters to be modified.
    suggested_parameters = moving_fly.trial.parameters
    # Generate array of parameter perturbations in [-1,1]^D.
    perturbations = self._create_perturbations(moving_fly.perturbation_factor)
    # Iterate over parameters and add perturbations.
    for i, param_config in enumerate(self.problem.search_space.parameters):
      if param_config.type == vz.ParameterType.CATEGORICAL:
        # For CATEGORICAL parameters, multiply the perturbation by the
        # pre-configured factor. This perturbation is interpreted as a
        # probability of replacing the value with a uniformly random category.
        if utils.is_pure_categorical(self.problem.search_space):
          perturbations[i] = self.config.pure_categorical_perturbation
        else:
          perturbations[i] *= self.config.categorical_perturbation_factor

      if param_config.type == vz.ParameterType.DISCRETE:
        # For DISCRETE parameters, multiply the perturbation by the
        # pre-configured factor divided by the number of feasible points.
        perturbations[i] *= self.config.discrete_perturbation_factor / (
            param_config.num_feasible_values * self.config.perturbation)

      self._key, suggested_parameters[
          param_config.name] = utils.perturb_parameter(
              self._key, param_config,
              suggested_parameters[param_config.name].value, perturbations[i])

  def _create_perturbations(self, perturbation_factor: float) -> List[float]:
    """"Creates perturbations vector."""
    # Sample vector from Laplace distribution.
    self._key, subkey = random.split(self._key)
    perturbations = [
        float(x) for x in random.laplace(subkey, (self.n_parameters,))
    ]
    # Normalize pertubations and scale by `perturbation_factor`.
    perturbation_direction = perturbations / np.max(np.abs(perturbations))
    perturbations = perturbation_direction * perturbation_factor
    return list(perturbations)

  def update(self, delta: vza.CompletedTrials) -> None:
    """Constructs the pool.

    Iterate over new completed trials and update the firefly pool. Every trial
    that was suggested from Eagle Strategy has a metadata data containing the
    parent fly id. For trials that were added to the study externally we assign
    a new parent fly id.

    Args:
      delta:
    """
    for trial in delta.completed:
      if not trial.metadata.ns('eagle').get('parent_fly_id'):
        # Trial was not generated from Eagle Strategy. Set a new parent fly id.
        trial.metadata.ns('eagle')[
            'parent_fly_id'] = self._firefly_pool.generate_new_fly_id()
      self._update_one(trial)

  def _update_one(self, trial: vz.Trial) -> None:
    """Update the pool using a single trial."""
    # Try finding the parent fly in the pool associated with the trial.
    parent_fly = self._firefly_pool.find_parent_fly(trial)
    if parent_fly is None:
      if trial.infeasible:
        # Ignore infeasible trials without parent fly.
        logging.log_if(
            logging.INFO,
            'Got infeasible trial without a parent fly in the pool.',
            self._verbose >= 1)
      elif self._firefly_pool.size < self._firefly_pool.capacity:
        # Pool is below capacity. Create a new firefly or update existing one.
        self._firefly_pool.create_or_update_fly(trial)
      else:
        # Pool is at capacity. Try to assign a parent to utilize the trial.
        parent_fly = self._assign_closest_parent(trial)

    if parent_fly is None:
      # Parent fly wasn't established. No need to continue.
      return

    elif utils.better_than(self.problem, trial, parent_fly.trial):
      # If there's an improvement, we update the parent with the new trial.
      logging.log_if(logging.INFO,
                     'Good step.\nParent trial: %s\nChild trial: %s',
                     self._verbose >= 2, str(parent_fly.trial), str(trial))
      parent_fly.trial = trial
      parent_fly.generation += 1
    else:
      # If there's no improvement, we penalize the parent by decreasing its
      # exploration capability and potenitally remove it from the pool.
      logging.log_if(logging.INFO,
                     'Bad step.\nParent trial: %s\nChild trial: %s',
                     self._verbose >= 2, str(parent_fly.trial), str(trial))
      self._penalize_parent_fly(parent_fly, trial)

  def _assign_closest_parent(self, trial: vz.Trial) -> Optional[_Firefly]:
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
    if utils.better_than(self.problem, trial, closest_parent_fly.trial):
      # Only returns the closest fly if there's improvement. Otherwise, we don't
      # return it to not count it as a failure, as the closest parent is not
      # reponsible for it.
      return closest_parent_fly

  def _penalize_parent_fly(self, parent_fly: _Firefly, trial: vz.Trial) -> None:
    """Penalizes the parent fly.

    The method is called on a fly after its generated trial didn't improve the
    objective function. The fly is penalized by decreasing its exploration
    capability and potentially removing it from the pool.

    Args:
      parent_fly: The Firefly objects to be penalized.
      trial: The newly received trial generated from 'parent_fly'.
    """
    if trial.parameters == parent_fly.trial.parameters:
      # If the new trial is identical to the parent trial, it means that the
      # fly is stuck, and so we increase its perturbation.
      parent_fly.perturbation_factor = min(parent_fly.perturbation_factor * 10,
                                           self.config.max_perturbation)
    else:
      # Otherwise, penalize the parent by decreasing its perturbation factor.
      parent_fly.perturbation_factor *= 0.9

    if parent_fly.perturbation_factor < self.config.perturbation_lower_bound:
      # If the perturbation factor is too low we attempt to eliminate the
      # unsuccessful parent fly from the pool.
      if self._firefly_pool.size == self._firefly_pool.capacity:
        # Only remove if the pool is at capacity. This is critical in studies
        # with few feasible/safe trials to retain the feasible trials.
        if not self._firefly_pool.is_best_fly(parent_fly):
          # Check that the fly is not the best one we have thus far.
          self._firefly_pool.remove_fly(parent_fly)
