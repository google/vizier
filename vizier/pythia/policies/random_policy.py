"""Random Pythia Policy which produces uniform sampling of Trial parameter values.

Since this is a RandomPolicy (i.e. stateless), we don't use the PolicySupporter
when suggesting trials.
"""
import random
from typing import List

from vizier.pythia import base
from vizier.pyvizier import oss as pyvizier
from vizier.pyvizier import pythia


def sample_from_parameter_config(
    parameter_config: pyvizier.ParameterConfig) -> pyvizier.ParameterValue:
  """Generate random values uniformly from a parameter_config."""
  if parameter_config.type == pyvizier.ParameterType.DOUBLE:
    min_value, max_value = parameter_config.bounds
    parameter_value = pyvizier.ParameterValue(
        value=random.uniform(min_value, max_value))
  elif parameter_config.type == pyvizier.ParameterType.INTEGER:
    min_value, max_value = parameter_config.bounds
    parameter_value = pyvizier.ParameterValue(
        value=random.randint(min_value, max_value))
  elif parameter_config.type == pyvizier.ParameterType.CATEGORICAL:
    parameter_value = pyvizier.ParameterValue(
        value=random.choice(parameter_config.feasible_values))
  elif parameter_config.type == pyvizier.ParameterType.DISCRETE:
    parameter_value = pyvizier.ParameterValue(
        value=random.choice(parameter_config.feasible_values))
  else:
    raise ValueError(
        f'ParameterConfig type is not one of the supported primitives for ParameterConfig: {parameter_config}'
    )
  return parameter_value


def make_random_parameters(
    study_config: pythia.StudyConfig) -> pythia.ParameterDict:
  """Makes random parameters from study_spec."""
  # TODO: Add conditional sampling case.
  parameter_dict = pythia.ParameterDict()
  for parameter_config in study_config.search_space.parameters:
    p_value = sample_from_parameter_config(parameter_config)
    parameter_dict[parameter_config.name] = p_value
  return parameter_dict


class RandomPolicy(base.Policy):
  """A policy that picks random hyper-parameter values."""

  def __init__(self):
    pass

  def suggest(self, request: base.SuggestRequest) -> base.SuggestDecisions:
    """Gets number of Trials to propose, and produces random Trials."""
    suggest_decision_list = []
    for _ in range(request.count):
      parameters = make_random_parameters(request.study_config)
      suggest_decision_list.append(base.SuggestDecision(parameters=parameters))
    return base.SuggestDecisions(suggest_decision_list)

  def early_stop(
      self, request: base.EarlyStopRequest) -> List[base.EarlyStopDecision]:
    """Selects a random ACTIVE trial to stop."""
    trial_to_stop_id = random.choice(list(request.trial_ids))
    early_stop_decision = base.EarlyStopDecision(
        id=trial_to_stop_id, reason='Random early stopping.')
    return [early_stop_decision]
