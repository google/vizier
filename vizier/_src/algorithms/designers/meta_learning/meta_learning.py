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

"""Meta Learning Designer.

The 'meta-learning' designer attempts to find the optimal hyper-parameters
values of the 'tuned' designer through a meta-learning process. In essensce, it
searches for the hyper-parameters values that yield the best result for the
given problem as oppose to blindly using pre-defined fixed values.

Notes
-----

1) The 'meta-learning' designer's search space is the hyper-parameters of
the 'tuned' designer which we seek to tune.

2) Before tuning starts, a tuned designer instantiated with the default
parameter values defined by the search space is used. This means that
default configuration defined in the designer level will be overridden.

3) The 'tuned_designer_factory' should accept the hyper-parameters as arguments.
This means for example, that if the 'tuned_designer' relies on an internal
configuration class the 'tuned_designer_factory' function would have to handle
the instantiation of that class (see 'eagle_meta_learning_convergence_test.py'
for an example).

4) Each instance of 'tuned' designer is updated with all trials seen thus-far,
therefore hyper-parameters that were created later in the process will benefit
from being instantiated with larger trajectory. This violates the assumption
of the meta-learning designer which the meta-trial metrics are derived from the
same objective function. For now we don't address this issue directly, though
in the future we could consider applying techniques such as RegEvo.
"""

import enum
from typing import Optional, Sequence
import attrs
import numpy as np
from vizier import algorithms as vza
from vizier import pyvizier as vz
from vizier._src.algorithms.designers.meta_learning import meta_learning_utils as utils


@attrs.define
class MetaLearningConfig:
  """Meta learning configuration (with pre-defined default values).

  To balance exploration and exploitation in the meta-learning process it's
  recommended to stop the meta-learner after a certain number of iterations so
  to take advantage of the thus-far best hyper-parameters and not to waste
  suggestion trials on further exploration.

  With the default configuration the number of meta-learning iterations is
  (10000-3000) / 100 = 70. In order to not terminate the meta-learning process,
  set the `tuning_max_num_trials` to sufficiently large value (e.g. 1e6).
  """

  # Number of trials to use per tuning iteration.
  num_trials_per_tuning: int = 100

  # Tuning starts when number of completed trials is at least this threshold.
  tuning_min_num_trials: int = 3000

  # Once the number of trials exceeds this threshold meta-learning stops.
  tuning_max_num_trials: int = 10000


class MetaLearningState(enum.Enum):
  """Meta learning state."""

  # The meta-learning process hasn't started yet, accumulating a sufficient
  # number of trials.
  INITIALIZE = 1

  # The meta-learning progress is performed to tune the designer.
  TUNE = 2

  # A tuned designer with the best hyper-params is used to generate suggestons.
  # Instantiate a tuned designer
  USE_BEST_PARAMS = 3


# TODO: support serialization.
@attrs.define
class MetaLearningDesigner(vza.Designer):
  """Meta learning designer."""

  # The problem associated with the 'tuned' designer.
  problem: vz.ProblemStatement

  # Factory of the 'tuned' designer.
  tuned_designer_factory: vza.DesignerFactory[vza.Designer]

  # Factory of the 'meta-learning' designer.
  meta_designer_factory: vza.DesignerFactory[vza.Designer]

  # The 'tuned' designer hyper-parameters to apply meta-learning on (which
  # constitute the search space of the 'meta-learning' designer).
  tuning_hyperparams: vz.SearchSpace

  # The meta-learner configuration.
  config: Optional[MetaLearningConfig] = None

  # A random seed used for instantiating the designers.
  seed: Optional[int] = None

  # ----------------------------------------------------------------------------
  # Internal Attributes
  # ----------------------------------------------------------------------------

  # The metric name used by the meta-learning designer.
  _meta_designer_metric_name: str = attrs.field(default='score', init=False)

  # The current 'tuned' designer instance with assigned hyper-parameter.
  _curr_tuned_designer: vza.Designer = attrs.field(init=False)

  # The 'meta' designer instance (only instantiated once).
  _meta_designer: vza.Designer = attrs.field(init=False)

  # The current state of the meta-learner.
  _state: MetaLearningState = attrs.field(init=False)

  # All the completed meta-trials suggested by the 'meta' designer.
  _meta_trials: list[vz.Trial] = attrs.field(factory=list, init=False)

  # The current 'tuned' hyper-parameter suggested by the meta-learner designer.
  _curr_tuned_hyperparams: vz.TrialSuggestion = attrs.field(init=False)

  # All the completed trials, suggested by the 'tuned' designers, seen since the
  # beginning of the meta-learner run.
  _trials: list[vz.Trial] = attrs.field(factory=list, init=False)

  # The completed trials suggested by the current 'tuned' designer which was
  # instantiated with the current hyper-parameter values.
  _curr_trials: list[vz.Trial] = attrs.field(factory=list, init=False)

  def __attrs_post_init__(self):
    """Initializes the meta learning desiger."""
    if len(self.problem.metric_information) != 1:
      raise ValueError(f'Expected exactly one metric. {self.problem}')

    if self.seed is None:
      # JAX random seed doesn't accept None, so generating random integer.
      self.seed = np.random.randint(low=0, high=1e6)

    # Instantiate an MetaLearningUtils.
    self._utils = utils.MetaLearningUtils(
        goal=self.problem.metric_information.item().goal,
        tuned_metric_name=self.problem.metric_information.item().name,
        meta_metric_name=self._meta_designer_metric_name,
        tuning_params=self.tuning_hyperparams,
    )
    # Instantiated 'tuned' designer the with default hyper-parameters.
    self._curr_tuned_hyperparams = self._utils.get_default_hyperparameters()
    self._curr_tuned_designer = self.tuned_designer_factory(
        self.problem,
        seed=self.seed,
        **self._curr_tuned_hyperparams.parameters.as_dict(),
    )
    self._meta_designer = self.meta_designer_factory(
        self._utils.meta_problem, seed=self.seed
    )
    self.config = self.config or MetaLearningConfig()
    self._state = MetaLearningState.INITIALIZE

  def suggest(self, count: int = 1) -> Sequence[vz.TrialSuggestion]:
    """Suggests trials."""
    return self._curr_tuned_designer.suggest(count)

  def update(
      self, completed: vza.CompletedTrials, all_active: vza.ActiveTrials
  ) -> None:
    """Incorporates completed trials into the meta-learner designer's state.

    1) At the beginning of the run the state is INITIALIZE, during which a
    'tune' designer with default hyper-parameter is used to suggest trials.

    2) After accumulating `tuning_min_num_trials` completed trials the state
    transitions to TUNE, and then the tuning process starts.

    3) After accumulating `num_trials_per_tuning` completd trials, the meta
    iteration is summarized, during which the best trial is selected to be used
    to update the 'meta' desinger.

    4) Once reaching maximum number of total completed meta-learn trials
    (`tuning_max_num_trials`), the meta-learning process is finalized during
    which the best hyper-parameter values are selected to instantiate a new
    'tuned' designer which is then updated with all seen trials.

    Args:
      completed:
      all_active:
    """
    self._trials.extend(completed.trials)
    self._curr_trials.extend(completed.trials)
    # Update curr_designer with newly completed trials (applies to all states).
    self._curr_tuned_designer.update(
        vza.CompletedTrials(completed.trials), vza.ActiveTrials()
    )

    # Check if enough trials already accumulated to start meta learning.
    if len(self._trials) < self.config.tuning_min_num_trials:
      return

    # Check if the meta-learning process should be terminated.
    elif len(self._trials) >= self.config.tuning_max_num_trials:
      # Check if the meta-learning has just terminated. If so, finalize it.
      if self._state == MetaLearningState.TUNE:
        # Find the best meta-learn result.
        best_meta_trial = self._utils.get_best_meta_trial(self._meta_trials)
        self._curr_tuned_designer = self.tuned_designer_factory(
            self.problem, seed=self.seed, **best_meta_trial.parameters.as_dict()
        )
        # Update the newly created designer with all completed trials.
        self._curr_tuned_designer.update(
            vza.CompletedTrials(self._trials), vza.ActiveTrials()
        )
        self._state = MetaLearningState.USE_BEST_PARAMS

    else:
      self._state = MetaLearningState.TUNE
      # Check if there's enough trials to summarize meta iteration.
      if len(self._curr_trials) >= self.config.num_trials_per_tuning:
        # Get best score for the current iteration.
        meta_trial = self._utils.complete_meta_suggestion(
            meta_suggestion=self._curr_tuned_hyperparams,
            score=self._utils.get_best_tuned_trial_score(self._curr_trials),
        )
        self._meta_designer.update(
            vza.CompletedTrials([meta_trial]), vza.ActiveTrials()
        )
        # Store the iteration results to be used during meta-learn finalization.
        self._meta_trials.append(meta_trial)
        # Get new tuned params suggestion and initialize a new curr_designer.
        self._curr_tuned_hyperparams = self._meta_designer.suggest(1)[0]
        self._curr_tuned_designer = self.tuned_designer_factory(
            self.problem,
            seed=self.seed,
            **self._curr_tuned_hyperparams.parameters.as_dict(),
        )
        # Update the newly created designer with all completed trials.
        self._curr_tuned_designer.update(
            vza.CompletedTrials(self._trials), vza.ActiveTrials()
        )
        # Reset the trials associated with current hyper-parameters.
        self._curr_trials = []
