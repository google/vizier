"""Grid Search Pythia Policy which searches over a discretized grid of Trial parameter values."""
from typing import List, Mapping, Sequence
import numpy as np
from vizier import pythia
from vizier import pyvizier

GRID_RESOLUTION = 100  # For double parameters.


def grid_points_from_parameter_config(
    parameter_config: pyvizier.ParameterConfig
) -> List[pyvizier.ParameterValue]:
  """Produces grid points from a parameter_config."""
  if parameter_config.type == pyvizier.ParameterType.DOUBLE:
    min_value, max_value = parameter_config.bounds

    if min_value == max_value:
      return [pyvizier.ParameterValue(value=min_value)]

    distance = (max_value - min_value) / (GRID_RESOLUTION - 1)
    grid_scalars = np.arange(min_value, max_value + distance, distance)
    return [pyvizier.ParameterValue(value=value) for value in grid_scalars]

  elif parameter_config.type == pyvizier.ParameterType.INTEGER:
    min_value, max_value = parameter_config.bounds
    return [
        pyvizier.ParameterValue(value=value)
        for value in range(min_value, max_value + 1)
    ]

  elif parameter_config.type == pyvizier.ParameterType.CATEGORICAL:
    return [
        pyvizier.ParameterValue(value=value)
        for value in parameter_config.feasible_values
    ]

  elif parameter_config.type == pyvizier.ParameterType.DISCRETE:
    return [
        pyvizier.ParameterValue(value=value)
        for value in parameter_config.feasible_values
    ]
  else:
    raise ValueError(
        f'ParameterConfig type is not one of the supported primitives for ParameterConfig: {parameter_config}'
    )


def make_grid_values(
    study_config: pyvizier.StudyConfig
) -> Mapping[str, List[pyvizier.ParameterValue]]:
  """Makes the grid values for every parameter."""
  grid_values = {}
  for parameter_config in study_config.search_space.parameters:
    grid_values[parameter_config.name] = grid_points_from_parameter_config(
        parameter_config)
  return grid_values


def make_grid_search_parameters(
    indices: Sequence[int],
    study_config: pyvizier.StudyConfig) -> List[pyvizier.ParameterDict]:
  """Selects the specific parameters from an index and study_spec based on the natural ordering over a Cartesian Product.

  This is looped over a sequence of indices. For a given `index`, this is
  effectively equivalent to itertools.product(list_of_lists)[index].

  Args:
    indices: Index over Cartesian Product.
    study_config: StudyConfig to produce the Cartesian Product. Ordering decided
      alphabetically over the parameter names.

  Returns:
    ParameterDict for a trial suggestion.
  """
  # TODO: Add conditional sampling case.
  for index in indices:
    if index < 0:
      raise ValueError('Indices can only be non-negative.')
  grid_values = make_grid_values(study_config)
  parameter_dicts = []
  for index in indices:
    parameter_dict = pyvizier.ParameterDict()
    temp_index = index
    for p_name in grid_values:
      p_length = len(grid_values[p_name])
      p_index = temp_index % p_length
      parameter_dict[p_name] = grid_values[p_name][p_index]
      temp_index = temp_index // p_length
    parameter_dicts.append(parameter_dict)
  return parameter_dicts


class GridSearchPolicy(pythia.Policy):
  """A policy that searches over a grid of hyper-parameter values."""

  def __init__(self, policy_supporter: pythia.PolicySupporter):
    self._policy_supporter = policy_supporter

  def suggest(self, request: pythia.SuggestRequest) -> pythia.SuggestDecision:
    """Gets number of Trials to propose, and produces random Trials."""
    all_trial_ids = [t.id for t in self._policy_supporter.GetTrials()]
    if all_trial_ids:
      next_index = max(all_trial_ids)
    else:
      next_index = 0

    parameter_dicts = make_grid_search_parameters(
        range(next_index, next_index + request.count), request.study_config)
    suggest_decision_list = [
        pyvizier.TrialSuggestion(parameters=p_s) for p_s in parameter_dicts
    ]
    return pythia.SuggestDecision(suggest_decision_list,
                                  pyvizier.MetadataDelta())

  def early_stop(
      self, request: pythia.EarlyStopRequest) -> List[pythia.EarlyStopDecision]:
    """Selects ACTIVE/PENDING trial with lowest ID to stop from datastore."""
    early_stop_decisions = []

    all_active_trials = self._policy_supporter.GetTrials(
        study_guid=request.study_guid,
        status_matches=pyvizier.TrialStatus.ACTIVE)
    trial_to_stop_id = None
    if all_active_trials:
      trial_to_stop_id = min([t.id for t in all_active_trials])
      early_stop_decisions.append(
          pythia.EarlyStopDecision(
              id=trial_to_stop_id, reason='Grid Search early stopping.'))

    for trial_id in list(request.trial_ids):
      if trial_id != trial_to_stop_id:
        early_stop_decisions.append(
            pythia.EarlyStopDecision(
                id=trial_id, reason='Trial should not stop.',
                should_stop=False))

    return early_stop_decisions
